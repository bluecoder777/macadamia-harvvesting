#!/usr/bin/env python3
"""Tree mapper — find the upright "trees" (pool noodles) from the 2D lidar and
publish their positions + a 50 cm acceptance zone.

Unlike the nuts (flat on the floor, invisible to the lidar), the trees are
vertical cylinders that the scan plane slices cleanly, so the lidar gives
accurate (~few cm) tree positions. Those define where nuts are allowed to be:
nut_tracker keeps a nut only if it lies within a tree's zone, which rejects
off-row phantoms / clutter.

Per scan:
  range-gated returns -> cluster by neighbour distance -> keep small
  (noodle-sized) clusters -> centroid -> push OUT by tree_radius (the lidar
  sees the near surface; the centre is one radius further) -> TF to
  target_frame -> persistent nearest-neighbour dedup -> confirm after min_hits.

Publishes:
    /trees           geometry_msgs/PoseArray (latched)  confirmed tree centres
    /trees/markers   visualization_msgs/MarkerArray     noodle + zone discs

Run:
    ros2 run macadamia_sweep tree_mapper
"""

import math
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Tree:
    __slots__ = ("x", "y", "hits")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.hits = 1


class TreeMapper(Node):
    def __init__(self):
        super().__init__("tree_mapper")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("target_frame", "map")
        # Lidar gating: ignore returns off the robot body, and only trust trees
        # within tree_max_range (point density falls off with distance).
        self.declare_parameter("lidar_min_range", 0.20)
        self.declare_parameter("tree_max_range", 2.50)
        # Clustering + noodle-size filter.
        self.declare_parameter("cluster_gap", 0.12)       # m between neighbours
        self.declare_parameter("tree_max_extent", 0.16)   # m, a noodle arc is small
        self.declare_parameter("min_points", 3)
        # Free-standing check: a real post has free space (background at least
        # this far behind, or no return at all) on BOTH angular sides. This is
        # the key clutter discriminator -- it rejects wall / furniture SURFACES,
        # which continue past the cluster instead of having gaps around them.
        self.declare_parameter("isolation_gap", 0.20)
        self.declare_parameter("tree_radius", 0.06)       # m, push centroid out
        # Persistence (trees are static -> accumulate).
        self.declare_parameter("merge_radius", 0.20)
        self.declare_parameter("min_hits", 4)
        # Acceptance-zone radius drawn in RViz (match tree_gate_radius in tracker).
        self.declare_parameter("zone_radius", 0.50)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.lidar_min_range = float(self.get_parameter("lidar_min_range").value)
        self.tree_max_range = float(self.get_parameter("tree_max_range").value)
        self.cluster_gap = float(self.get_parameter("cluster_gap").value)
        self.tree_max_extent = float(self.get_parameter("tree_max_extent").value)
        self.min_points = int(self.get_parameter("min_points").value)
        self.isolation_gap = float(self.get_parameter("isolation_gap").value)
        self.tree_radius = float(self.get_parameter("tree_radius").value)
        self.merge_radius = float(self.get_parameter("merge_radius").value)
        self.min_hits = int(self.get_parameter("min_hits").value)
        self.zone_radius = float(self.get_parameter("zone_radius").value)

        self.trees: List[Tree] = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched.history = HistoryPolicy.KEEP_LAST
        self.tree_pub = self.create_publisher(PoseArray, "/trees", latched)
        self.marker_pub = self.create_publisher(MarkerArray, "/trees/markers", 10)

        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_timer(0.5, self.publish_markers)

        self.get_logger().info(
            f"tree_mapper up. {self.scan_topic} -> {self.target_frame}, "
            f"zone_radius={self.zone_radius:.2f}m."
        )

    # -----------------------------
    # Scan -> tree detections
    # -----------------------------

    def scan_cb(self, msg: LaserScan):
        n = len(msg.ranges)
        rmin, rmax = msg.range_min, msg.range_max

        def valid(r):
            return (not (math.isnan(r) or math.isinf(r))) and rmin <= r <= rmax

        def in_window(r):
            return valid(r) and self.lidar_min_range <= r <= self.tree_max_range

        # Cluster CONSECUTIVE in-window returns (keep the index so we can look
        # at the neighbours just outside the cluster). A gap (invalid / out of
        # window) or a big Cartesian jump ends a cluster.
        clusters: List[List[Tuple[int, float, float, float]]] = []
        cur: List[Tuple[int, float, float, float]] = []
        for i in range(n):
            r = msg.ranges[i]
            if in_window(r):
                a = msg.angle_min + i * msg.angle_increment
                p = (i, r, r * math.cos(a), r * math.sin(a))
                if cur and math.hypot(p[2] - cur[-1][2], p[3] - cur[-1][3]) > self.cluster_gap:
                    clusters.append(cur); cur = [p]
                else:
                    cur.append(p)
            else:
                if cur:
                    clusters.append(cur); cur = []
        if cur:
            clusters.append(cur)

        centres_laser: List[Tuple[float, float]] = []
        for c in clusters:
            if len(c) < self.min_points:
                continue
            xs = [q[2] for q in c]; ys = [q[3] for q in c]
            extent = max(max(xs) - min(xs), max(ys) - min(ys))
            if extent > self.tree_max_extent:
                continue

            # FREE-STANDING test: the returns just outside the cluster on each
            # angular side must be empty or far behind (background). A wall /
            # furniture surface keeps a similar-range neighbour and is rejected.
            mean_r = sum(q[1] for q in c) / len(c)

            def neighbour_free(idx):
                if not (0 <= idx < n):
                    return True
                rr = msg.ranges[idx]
                if not valid(rr):
                    return True                       # no return -> free space
                return rr > mean_r + self.isolation_gap   # far background -> free
            if not (neighbour_free(c[0][0] - 1) and neighbour_free(c[-1][0] + 1)):
                continue

            mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
            d = math.hypot(mx, my)
            if d > 1e-3:                       # push outward by the noodle radius
                mx += self.tree_radius * mx / d
                my += self.tree_radius * my / d
            centres_laser.append((mx, my))
        if not centres_laser:
            return

        # TF laser -> target_frame (once for the scan).
        src = msg.header.frame_id or "laser"
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, src, Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.1))
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(self.target_frame, src, Time())
            except Exception:
                return
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_R(q.x, q.y, q.z, q.w)
        origin = np.array([t.x, t.y, t.z])

        for (mx, my) in centres_laser:
            p = R @ np.array([mx, my, 0.0]) + origin
            self.associate(float(p[0]), float(p[1]))

    def associate(self, x: float, y: float):
        best = None
        best_d = self.merge_radius
        for tr in self.trees:
            d = math.hypot(tr.x - x, tr.y - y)
            if d < best_d:
                best_d = d; best = tr
        if best is None:
            self.trees.append(Tree(x, y))
        else:
            w = best.hits
            best.x = (best.x * w + x) / (w + 1)
            best.y = (best.y * w + y) / (w + 1)
            best.hits += 1
            if best.hits == self.min_hits:       # just confirmed -> push the list
                self.publish_trees()

    # -----------------------------
    # Outputs
    # -----------------------------

    def confirmed(self) -> List[Tree]:
        return [t for t in self.trees if t.hits >= self.min_hits]

    def publish_trees(self):
        pa = PoseArray()
        pa.header.frame_id = self.target_frame
        pa.header.stamp = self.get_clock().now().to_msg()
        for t in self.confirmed():
            p = Pose()
            p.position.x = t.x
            p.position.y = t.y
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.tree_pub.publish(pa)

    def publish_markers(self):
        arr = MarkerArray()
        for i, t in enumerate(self.confirmed()):
            noodle = Marker()
            noodle.header.frame_id = self.target_frame
            noodle.header.stamp = self.get_clock().now().to_msg()
            noodle.ns = "trees"
            noodle.id = i
            noodle.type = Marker.CYLINDER
            noodle.action = Marker.ADD
            noodle.pose.position.x = t.x
            noodle.pose.position.y = t.y
            noodle.pose.position.z = 0.15
            noodle.pose.orientation.w = 1.0
            noodle.scale.x = noodle.scale.y = 2.0 * self.tree_radius
            noodle.scale.z = 0.30
            noodle.color.r, noodle.color.g, noodle.color.b, noodle.color.a = 0.9, 0.4, 0.4, 0.9
            arr.markers.append(noodle)

            zone = Marker()
            zone.header.frame_id = self.target_frame
            zone.header.stamp = noodle.header.stamp
            zone.ns = "tree_zones"
            zone.id = i
            zone.type = Marker.CYLINDER
            zone.action = Marker.ADD
            zone.pose.position.x = t.x
            zone.pose.position.y = t.y
            zone.pose.position.z = 0.01
            zone.pose.orientation.w = 1.0
            zone.scale.x = zone.scale.y = 2.0 * self.zone_radius
            zone.scale.z = 0.02
            zone.color.r, zone.color.g, zone.color.b, zone.color.a = 0.2, 0.8, 0.2, 0.20
            arr.markers.append(zone)

        self.marker_pub.publish(arr)
        self.publish_trees()


def main(args=None):
    rclpy.init(args=args)
    node = TreeMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
