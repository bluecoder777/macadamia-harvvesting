#!/usr/bin/env python3
"""Nut tracker — WORLD MODEL + bookkeeping layer.

Consumes the stateless per-frame detections from nut_detector and maintains
the persistent truth about nuts:
  * De-duplicates re-observations of the same nut (nearest-neighbour data
    association within merge_radius -- the standard landmark-association test;
    see Suenderhauf et al., object-oriented semantic mapping).
  * Requires a few hits before a candidate is "confirmed" (rejects one-frame
    colour flickers).
  * Marks a nut COLLECTED when base_link passes within collection_radius of it
    -- your rule: "if the rosbot goes over a nut in the sweeping area it is
    collected". The detector can't see under the robot, but we already stored
    the nut's map position when it was ahead, so the drive-over check is purely
    geometric and needs no live sighting.
  * Publishes spheres to RViz: RED = uncollected, GREEN = collected.
  * Persists uncollected nut locations (latched topic + CSV) so a later phase
    can plan one sweep to collect them all (a TSP / coverage problem).

This is the deliberative half of the three-layer architecture: perception
(nut_detector) feeds it, and it owns the map of the world + the mission state.

Subscribes:
    /nuts/detections        geometry_msgs/PoseArray   (frame = map)

Publishes:
    /nuts/markers           visualization_msgs/MarkerArray   spheres + count
    /nuts/uncollected       geometry_msgs/PoseArray (latched) for the picker
    /snc_status             std_msgs/String   "Nuts: collected X / total Y"

Run:
    ros2 run macadamia_sweep nut_tracker
    # then in RViz add a MarkerArray display on /nuts/markers (Fixed Frame: map)
"""

import csv
import math
import os
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


class Nut:
    __slots__ = ("id", "x", "y", "hits", "collected", "collected_t")

    def __init__(self, nid: int, x: float, y: float):
        self.id = nid
        self.x = x
        self.y = y
        self.hits = 1
        self.collected = False
        self.collected_t: Optional[float] = None


class NutTracker(Node):
    def __init__(self):
        super().__init__("nut_tracker")

        # ---- Frames ----
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")

        # ---- Association / confirmation ----
        # Two detections within this distance are the SAME nut. Set wide enough
        # to absorb the projected-position jitter of far/foreshortened nuts
        # (which otherwise spawn a cloud of phantom nuts), but below the real
        # nut spacing so distinct nuts stay distinct.
        self.declare_parameter("merge_radius", 0.25)
        # Hits before a candidate becomes a confirmed (shown) nut.
        self.declare_parameter("min_hits", 3)

        # ---- Collection ----
        # Robot "sweeps up" a nut when base_link passes within this radius.
        # ~ half the robot footprint / brush width. Tune to your sweeper.
        self.declare_parameter("collection_radius", 0.25)

        # ---- Visualisation ----
        self.declare_parameter("marker_diameter", 0.08)   # bigger than 3 cm so it's visible
        self.declare_parameter("marker_namespace", "nuts")

        # ---- Persistence ----
        self.declare_parameter(
            "save_path", os.path.expanduser("~/nut_locations.csv")
        )

        self.map_frame = self.get_parameter("map_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.merge_radius = float(self.get_parameter("merge_radius").value)
        self.min_hits = int(self.get_parameter("min_hits").value)
        self.collection_radius = float(self.get_parameter("collection_radius").value)
        self.marker_diameter = float(self.get_parameter("marker_diameter").value)
        self.marker_ns = self.get_parameter("marker_namespace").value
        self.save_path = self.get_parameter("save_path").value

        # ---- State ----
        self.nuts: List[Nut] = []
        self.next_id = 0

        # ---- TF (to read robot pose for the drive-over check) ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Pub / Sub ----
        self.marker_pub = self.create_publisher(MarkerArray, "/nuts/markers", 10)
        self.status_pub = self.create_publisher(String, "/snc_status", 10)
        # Latched so a picker node that starts LATER still receives the list.
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched.history = HistoryPolicy.KEEP_LAST
        self.uncollected_pub = self.create_publisher(
            PoseArray, "/nuts/uncollected", latched
        )

        self.create_subscription(
            PoseArray, "/nuts/detections", self.detections_callback, 10
        )

        # Collection check + marker publish on a steady timer (independent of
        # detection rate; the drive-over event happens when the nut is under
        # the robot and therefore NOT being detected).
        self.create_timer(0.2, self.collection_tick)
        self.create_timer(0.5, self.publish_markers)

        self.get_logger().info(
            f"nut_tracker up. merge_radius={self.merge_radius:.2f}m "
            f"min_hits={self.min_hits} collection_radius={self.collection_radius:.2f}m. "
            f"RViz: MarkerArray on /nuts/markers (Fixed Frame {self.map_frame})."
        )

    # -----------------------------
    # Data association
    # -----------------------------

    def detections_callback(self, msg: PoseArray):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().warn(
                f"detections are in '{msg.header.frame_id}', expected "
                f"'{self.map_frame}'. Ignoring.", throttle_duration_sec=5.0,
            )
            return
        for pose in msg.poses:
            self.associate(pose.position.x, pose.position.y)

    def associate(self, x: float, y: float):
        """Nearest-neighbour: fold the detection into the closest existing nut
        within merge_radius, else start a new candidate."""
        best: Optional[Nut] = None
        best_d = self.merge_radius
        for n in self.nuts:
            d = math.hypot(n.x - x, n.y - y)
            if d < best_d:
                best_d = d
                best = n
        if best is None:
            self.nuts.append(Nut(self.next_id, x, y))
            self.next_id += 1
            return
        # Running average — but freeze the position once collected so a stray
        # later sighting can't drag a collected nut around.
        if not best.collected:
            w = best.hits
            best.x = (best.x * w + x) / (w + 1)
            best.y = (best.y * w + y) / (w + 1)
        best.hits += 1

    # -----------------------------
    # Collection (drive-over) check
    # -----------------------------

    def robot_xy(self) -> Optional[tuple]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, Time()
            )
        except Exception:
            return None
        return (tf.transform.translation.x, tf.transform.translation.y)

    def collection_tick(self):
        rxy = self.robot_xy()
        if rxy is None:
            return
        rx, ry = rxy
        changed = False
        now = self.get_clock().now().nanoseconds * 1e-9
        for n in self.nuts:
            if n.collected or n.hits < self.min_hits:
                continue
            if math.hypot(n.x - rx, n.y - ry) <= self.collection_radius:
                n.collected = True
                n.collected_t = now
                changed = True
                self.get_logger().info(
                    f"Nut #{n.id} collected at ({n.x:+.2f},{n.y:+.2f})."
                )
        if changed:
            self.publish_uncollected()
        self.publish_status()

    # -----------------------------
    # Outputs
    # -----------------------------

    def confirmed(self) -> List[Nut]:
        return [n for n in self.nuts if n.hits >= self.min_hits]

    def publish_status(self):
        conf = self.confirmed()
        collected = sum(1 for n in conf if n.collected)
        m = String()
        m.data = f"Nuts: collected {collected} / total {len(conf)}"
        self.status_pub.publish(m)

    def publish_uncollected(self):
        conf = self.confirmed()
        pa = PoseArray()
        pa.header.frame_id = self.map_frame
        pa.header.stamp = self.get_clock().now().to_msg()
        for n in conf:
            if n.collected:
                continue
            p = Pose()
            p.position.x = n.x
            p.position.y = n.y
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.uncollected_pub.publish(pa)

    def publish_markers(self):
        arr = MarkerArray()
        conf = self.confirmed()
        for n in conf:
            mk = Marker()
            mk.header.frame_id = self.map_frame
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = self.marker_ns
            mk.id = n.id
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = n.x
            mk.pose.position.y = n.y
            mk.pose.position.z = self.marker_diameter / 2.0  # rest on the floor
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = self.marker_diameter
            if n.collected:
                mk.color.r, mk.color.g, mk.color.b = 0.1, 0.85, 0.1  # green
            else:
                mk.color.r, mk.color.g, mk.color.b = 0.9, 0.1, 0.1   # red
            mk.color.a = 0.9
            arr.markers.append(mk)

        # A floating text marker = the live "loss estimate" (RMIT-style).
        collected = sum(1 for n in conf if n.collected)
        txt = Marker()
        txt.header.frame_id = self.map_frame
        txt.header.stamp = self.get_clock().now().to_msg()
        txt.ns = self.marker_ns + "_label"
        txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.z = 0.6
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.12
        txt.color.r = txt.color.g = txt.color.b = 1.0
        txt.color.a = 1.0
        txt.text = f"nuts collected {collected}/{len(conf)}"
        arr.markers.append(txt)

        self.marker_pub.publish(arr)

    # -----------------------------
    # Persistence
    # -----------------------------

    def save_csv(self):
        try:
            with open(self.save_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "x_map", "y_map", "collected", "hits"])
                for n in self.nuts:
                    if n.hits < self.min_hits:
                        continue
                    w.writerow([n.id, f"{n.x:.4f}", f"{n.y:.4f}",
                                int(n.collected), n.hits])
            self.get_logger().info(f"Saved nut map to {self.save_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save nut map: {e}")

    def destroy_node(self):
        self.save_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NutTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
