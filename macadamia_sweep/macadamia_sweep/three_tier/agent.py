#!/usr/bin/env python3
"""Agent: the ROS2 node that hosts the 3T architecture.

This is the only ROS-aware module. It owns the publishers/subscribers/
parameters/timer and the TF buffer, fills the world model from sensor
callbacks, and -- crucially -- acts as the ``io`` interface every tier uses for
actuation, timing, logging, perception enable and TF lookups. Once per 10 Hz
timer cycle it ticks the executive (``Sequencer.tick``), which in turn drives
the active skill and, at mission forks, the Deliberator.

The ROS surface (topics, QoS, parameters, the 0.1 s timer and the published
status text) is identical to the original ``simple_row_follower`` node, so the
rest of the system (perception nodes, RViz, operator triggers) is unaffected.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from geometry_msgs.msg import TwistStamped, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
import tf2_ros

from .world_model import WorldModel
from .perception import Perception
from .skills import Skills
from .sequencer import Sequencer
from .deliberator import Deliberator


class RowFollower3T(Node):
    def __init__(self):
        super().__init__("row_follower_3t")

        # The shared blackboard.
        self.wm = WorldModel()
        wm = self.wm

        # ---- Parameters (declared + read exactly as the original node) ------
        self.declare_parameter("start_side", "right")
        wm.start_side = str(self.get_parameter("start_side").value).lower()
        if wm.start_side not in ("right", "left"):
            self.get_logger().warn(
                f"Unknown start_side '{wm.start_side}', defaulting to 'right'"
            )
            wm.start_side = "right"
        wm.current_side = wm.start_side

        self.declare_parameter("lidar_yaw_offset_deg", 180.0)
        wm.lidar_yaw_offset = math.radians(
            float(self.get_parameter("lidar_yaw_offset_deg").value)
        )

        self.declare_parameter("max_rows", 0)
        wm.max_rows = int(self.get_parameter("max_rows").value)
        self.declare_parameter("next_row_min_hits", 8)
        wm.next_row_min_hits = int(self.get_parameter("next_row_min_hits").value)
        self.declare_parameter("next_row_max_dist", 0.60)
        wm.next_row_max_dist = float(self.get_parameter("next_row_max_dist").value)

        self.declare_parameter("collect_before_home", True)
        self.declare_parameter("nuts_topic", "/nuts/uncollected")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("collect_sweep_offset", 0.40)
        self.declare_parameter("collect_sweep_through", 0.60)
        self.declare_parameter("collect_arrive_tol", 0.15)
        self.declare_parameter("collect_visit_timeout", 45.0)
        self.declare_parameter("collect_max_avoid", 3)
        self.declare_parameter("collect_max_consec_skips", 3)
        wm.collect_before_home = bool(self.get_parameter("collect_before_home").value)
        wm.nuts_topic = self.get_parameter("nuts_topic").value
        wm.map_frame = self.get_parameter("map_frame").value
        wm.odom_frame = self.get_parameter("odom_frame").value
        wm.collect_sweep_offset = float(self.get_parameter("collect_sweep_offset").value)
        wm.collect_sweep_through = float(self.get_parameter("collect_sweep_through").value)
        wm.collect_arrive_tol = float(self.get_parameter("collect_arrive_tol").value)
        wm.collect_visit_timeout = float(self.get_parameter("collect_visit_timeout").value)
        wm.collect_max_avoid = int(self.get_parameter("collect_max_avoid").value)
        wm.collect_max_consec_skips = int(self.get_parameter("collect_max_consec_skips").value)
        self.declare_parameter("collect_side_clearance", 0.25)
        self.declare_parameter("collect_turn_away", 0.30)
        wm.collect_side_clearance = float(self.get_parameter("collect_side_clearance").value)
        wm.collect_turn_away = float(self.get_parameter("collect_turn_away").value)
        self.declare_parameter("path_sample_spacing", 0.15)
        self.declare_parameter("path_max_points", 4000)
        self.declare_parameter("path_wp_tol", 0.20)
        self.declare_parameter("path_lookahead", 1)
        wm.path_sample_spacing = float(self.get_parameter("path_sample_spacing").value)
        wm.path_max_points = int(self.get_parameter("path_max_points").value)
        wm.path_wp_tol = float(self.get_parameter("path_wp_tol").value)
        wm.path_lookahead = int(self.get_parameter("path_lookahead").value)

        self.declare_parameter("tree_max_range", 1.20)
        wm.tree_max_range = float(self.get_parameter("tree_max_range").value)

        self.declare_parameter("row_group_gap", 0.35)
        wm.row_group_gap = float(self.get_parameter("row_group_gap").value)

        self.declare_parameter("lateral_align_tol", 0.05)
        self.declare_parameter("lateral_align_max_duration", 20.0)
        wm.lateral_align_tol = float(self.get_parameter("lateral_align_tol").value)
        wm.lateral_align_max_duration = float(
            self.get_parameter("lateral_align_max_duration").value)

        self.declare_parameter("clear_end_distance", 0.15)
        wm.clear_end_distance = float(self.get_parameter("clear_end_distance").value)

        self.declare_parameter("arc_radius", 0.40)
        wm.arc_radius = float(self.get_parameter("arc_radius").value)
        # Derived from arc params (mirrors the original computation).
        nominal_omega = wm.arc_linear_speed / wm.arc_radius
        wm.arc_max_duration = wm.arc_max_yaw / nominal_omega + 5.0

        self.declare_parameter("avoid_front_distance", 0.28)
        wm.avoid_front_distance = float(self.get_parameter("avoid_front_distance").value)

        self.declare_parameter("return_home_enabled", True)
        wm.return_home_enabled = bool(self.get_parameter("return_home_enabled").value)
        self.declare_parameter("return_goal_tolerance", 0.12)
        wm.return_goal_tolerance = float(self.get_parameter("return_goal_tolerance").value)
        self.declare_parameter("return_yaw_tolerance_deg", 12.0)
        wm.return_yaw_tolerance = math.radians(
            float(self.get_parameter("return_yaw_tolerance_deg").value)
        )
        self.declare_parameter("return_max_duration", 120.0)
        wm.return_max_duration = float(self.get_parameter("return_max_duration").value)
        self.declare_parameter("return_exit_margin", 0.60)
        wm.return_exit_margin = float(self.get_parameter("return_exit_margin").value)

        # ---- Publishers -----------------------------------------------------
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel_nav", 10)
        self.status_pub = self.create_publisher(String, "/snc_status", 10)
        _enable_qos = QoSProfile(depth=1)
        _enable_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        _enable_qos.history = HistoryPolicy.KEEP_LAST
        self.detect_enable_pub = self.create_publisher(
            Bool, "/nuts/detect_enable", _enable_qos)
        self.tree_enable_pub = self.create_publisher(
            Bool, "/trees/enable", _enable_qos)

        # ---- Tiers (wired together) -----------------------------------------
        self.perc = Perception(wm)
        self.skills = Skills(wm, self, self.perc)
        self.seq = Sequencer(wm, self, self.perc, self.skills)
        self.delib = Deliberator(wm, self, self.seq)
        self.seq.deliberator = self.delib
        self.skills.enter_avoid = self.seq.enter_avoid_front

        # ---- Subscribers ----------------------------------------------------
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 20)
        self.create_subscription(Empty, "/sweep_start", self.start_callback, 10)
        self.create_subscription(Empty, "/sweep_stop", self.stop_callback, 10)
        self.create_subscription(Empty, "/return_home", self.return_home_callback, 10)

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(
            PoseArray, wm.nuts_topic, self.uncollected_callback, latched)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initial clock stamps (the WAITING state never moves, but the helpers
        # subtract from these, so they must be real Time objects).
        now = self.get_clock().now()
        wm.state_start_time = now
        wm.last_row_seen_time = now
        wm.avoid_phase_start_time = now

        self.timer = self.create_timer(0.1, self.control_loop)

        self.publish_status("Simple row follower ready. Publish /sweep_start to begin.")
        self.get_logger().info(
            f"Ready (3T, multi-row). CLEAR={wm.clear_end_distance:.2f}m, "
            f"ARC r={wm.arc_radius:.2f}m v={wm.arc_linear_speed:.2f}m/s, "
            f"start_side={wm.start_side}, "
            f"max_rows={wm.max_rows or 'unlimited'}, "
            f"row_group_gap={wm.row_group_gap:.2f}m, "
            f"lidar_yaw_offset={math.degrees(wm.lidar_yaw_offset):+.0f}deg, "
            f"return_home={wm.return_home_enabled}, manual_topic=/return_home, "
            f"avoid_front={wm.avoid_front_distance:.2f}m"
        )

    # -----------------------------
    # Sensor / trigger callbacks (fill the world model)
    # -----------------------------
    def scan_callback(self, msg: LaserScan):
        self.wm.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        self.wm.odom_x = msg.pose.pose.position.x
        self.wm.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.wm.odom_yaw = math.atan2(siny, cosy)

    def uncollected_callback(self, msg: PoseArray):
        self.wm.uncollected_map = [(p.position.x, p.position.y) for p in msg.poses]

    def start_callback(self, _msg: Empty):
        self.delib.on_sweep_start()

    def stop_callback(self, _msg: Empty):
        self.delib.on_sweep_stop()

    def return_home_callback(self, _msg: Empty):
        self.delib.on_return_home()

    def control_loop(self):
        self.seq.tick()

    # -----------------------------
    # io interface used by every tier
    # -----------------------------
    def now(self):
        return self.get_clock().now()

    def elapsed_since(self, t) -> float:
        return (self.get_clock().now() - t).nanoseconds / 1e9

    def log_info(self, text: str):
        self.get_logger().info(text)

    def log_warn(self, text: str):
        self.get_logger().warn(text)

    def publish_status(self, text: str):
        m = String()
        m.data = text
        self.status_pub.publish(m)

    def set_perception(self, enabled: bool):
        """Enable/pause BOTH perception nodes together: nut_detector (latched
        /nuts/detect_enable) and tree_mapper (latched /trees/enable)."""
        m = Bool()
        m.data = bool(enabled)
        self.detect_enable_pub.publish(m)
        self.tree_enable_pub.publish(m)

    def publish_cmd(self, linear: float, angular: float, lateral: float = 0.0):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.twist.linear.x = float(linear)
        m.twist.linear.y = float(lateral)   # mecanum strafe (+y = left); 0 otherwise
        m.twist.angular.z = float(angular)
        self.cmd_pub.publish(m)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def map_to_odom(self):
        """(tx, ty, cos, sin) of the odom<-map transform, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.wm.odom_frame, self.wm.map_frame, Time())
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, math.cos(yaw), math.sin(yaw))


def main(args=None):
    rclpy.init(args=args)
    node = RowFollower3T()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()       # only publish while the context is alive
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
