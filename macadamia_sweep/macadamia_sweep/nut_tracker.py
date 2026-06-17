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
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException

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
        self.declare_parameter("merge_radius", 0.30)
        # Hits before a candidate becomes a confirmed (shown) nut.
        self.declare_parameter("min_hits", 3)

        # ---- Tree (sweep-area) gate ----
        # A nut is kept only if it lies within tree_gate_radius of a known tree
        # (from tree_mapper on /trees). Rejects off-row phantoms / clutter that
        # the colour+shape gates can't. If no trees are known yet, gating is
        # skipped (accept all) so it degrades gracefully without tree_mapper.
        self.declare_parameter("use_tree_gate", True)
        self.declare_parameter("tree_gate_radius", 0.50)
        self.declare_parameter("trees_topic", "/trees")

        # ---- Collection ----
        # The sweeper is offset to the hug side of the robot (it sweeps the side
        # the robot hugs the tree line with), so a nut is collected when it
        # passes within collection_radius of the SWEEPER point, not the robot
        # centre. sweeper point = robot + sweep_offset * (hug-side direction).
        # sweep_offset = 0 reverts to a plain centre drive-over.
        self.declare_parameter("collection_radius", 0.25)
        self.declare_parameter("sweep_offset", 0.40)      # m to the hug side
        self.declare_parameter("sweep_side", "right")     # side the sweeper is on

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
        self.use_tree_gate = bool(self.get_parameter("use_tree_gate").value)
        self.tree_gate_radius = float(self.get_parameter("tree_gate_radius").value)
        self.trees_topic = self.get_parameter("trees_topic").value
        self.collection_radius = float(self.get_parameter("collection_radius").value)
        self.sweep_offset = float(self.get_parameter("sweep_offset").value)
        self.sweep_side = str(self.get_parameter("sweep_side").value).lower()
        self.marker_diameter = float(self.get_parameter("marker_diameter").value)
        self.marker_ns = self.get_parameter("marker_namespace").value
        self.save_path = self.get_parameter("save_path").value

        # ---- State ----
        self.nuts: List[Nut] = []
        self.next_id = 0
        self.tree_pts: List[Tuple[float, float]] = []   # tree centres (sweep-area gate)

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
        # Tree positions (latched) for the sweep-area gate.
        tree_qos = QoSProfile(depth=1)
        tree_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        tree_qos.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(
            PoseArray, self.trees_topic, self.trees_callback, tree_qos
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

    def trees_callback(self, msg: PoseArray):
        self.tree_pts = [(p.position.x, p.position.y) for p in msg.poses]

    def associate(self, x: float, y: float):
        """Nearest-neighbour: fold the detection into the closest existing nut
        within merge_radius, else start a new candidate."""
        # Sweep-area gate: a real nut is near a tree. Reject detections that are
        # not within tree_gate_radius of any known tree (off-row clutter). Skip
        # the gate until at least one tree is known.
        if self.use_tree_gate and self.tree_pts:
            nearest = min(math.hypot(tx - x, ty - y) for (tx, ty) in self.tree_pts)
            if nearest > self.tree_gate_radius:
                return

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
        """Robot centre in the map frame, or None if TF is unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, Time()
            )
        except Exception:
            return None
        return (tf.transform.translation.x, tf.transform.translation.y)

    def sweeper_xy(self) -> Optional[tuple]:
        """The sweeper point: robot centre + sweep_offset to the hug side.
        Uses the robot's heading so the offset is in the correct world
        direction. Returns None if TF is unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, Time()
            )
        except Exception:
            return None
        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # Right of the robot = (sin yaw, -cos yaw); left = (-sin yaw, cos yaw).
        if self.sweep_side == "left":
            ux, uy = -math.sin(yaw), math.cos(yaw)
        else:
            ux, uy = math.sin(yaw), -math.cos(yaw)
        return (rx + self.sweep_offset * ux, ry + self.sweep_offset * uy)

    def collection_tick(self):
        # A nut is collected if it passes under the SWEEPER (offset to the hug
        # side) OR directly under the ROBOT body - both sweep it up.
        points = [p for p in (self.robot_xy(), self.sweeper_xy()) if p is not None]
        if not points:
            return
        changed = False
        now = self.get_clock().now().nanoseconds * 1e-9
        for n in self.nuts:
            if n.collected or n.hits < self.min_hits:
                continue
            if any(math.hypot(n.x - px, n.y - py) <= self.collection_radius
                   for (px, py) in points):
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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()        # saves the CSV
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
