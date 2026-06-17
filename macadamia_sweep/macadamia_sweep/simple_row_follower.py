#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from rclpy.time import Time
import tf2_ros


class SimpleRowFollower(Node):
    def __init__(self):
        super().__init__("simple_row_follower")

        # Parameter: starting side the row is on. "right" or "left".
        # Override at launch: --ros-args -p start_side:=left
        self.declare_parameter("start_side", "right")
        self.start_side = str(self.get_parameter("start_side").value).lower()
        if self.start_side not in ("right", "left"):
            self.get_logger().warn(
                f"Unknown start_side '{self.start_side}', defaulting to 'right'"
            )
            self.start_side = "right"

        # Parameter: rotation (radians) to add to raw lidar angles so that
        # the code's "0 deg = forward" convention matches the physical
        # orientation of the lidar on this robot. If the lidar is mounted
        # rotated 180 deg (an object physically in front of the robot
        # reports at +/-180 deg in raw scan angles), use math.pi here.
        # Verify with the lidar_check diagnostic.
        self.declare_parameter("lidar_yaw_offset_deg", 180.0)
        self.lidar_yaw_offset = math.radians(
            float(self.get_parameter("lidar_yaw_offset_deg").value)
        )

        # ---------------------------------------------------------------
        # Multi-row sweep.
        #
        # Mission: sweep one side of row 1, U-turn at the far end, sweep
        # its other side back. During that return pass the NEXT row sits on
        # the opposite side of the robot; if it is seen consistently, the
        # robot clears the row start, turns 180 deg in place toward the
        # next row, and repeats the identical sweep on it. Every row is
        # swept exactly twice (once per side). Nothing about row spacing
        # or row count is hardcoded.
        #
        # max_rows > 0 -> FIXED COUNT (selected behaviour): sweep exactly
        #                 that many rows, turning to the next row after each
        #                 one REGARDLESS of what perception sees. Set with
        #                 --ros-args -p max_rows:=4.
        # max_rows = 0 -> AUTO: continue only while a next row is detected on
        #                 the opposite side during the return pass.
        # ---------------------------------------------------------------
        self.declare_parameter("max_rows", 0)
        self.max_rows = int(self.get_parameter("max_rows").value)
        # How many control ticks (10 Hz) the opposite-side row fit must
        # succeed during FOLLOW_BACK before we commit to a next row.
        self.declare_parameter("next_row_min_hits", 8)
        self.next_row_min_hits = int(self.get_parameter("next_row_min_hits").value)
        # A genuine NEXT row sits (row spacing - follow offset) away on the
        # opposite side during the return pass (~0.30 m for 0.70 m rows).
        # Anything farther than this is NOT the next row (e.g. an already
        # swept row seen across a gap) - prevents re-sweeping forever.
        self.declare_parameter("next_row_max_dist", 0.60)
        self.next_row_max_dist = float(
            self.get_parameter("next_row_max_dist").value
        )
        self.next_row_hits = 0
        self.rows_completed = 0

        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel_nav", 10)
        self.status_pub = self.create_publisher(String, "/snc_status", 10)

        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 20)
        self.create_subscription(Empty, "/sweep_start", self.start_callback, 10)
        self.create_subscription(Empty, "/sweep_stop", self.stop_callback, 10)
        self.create_subscription(Empty, "/return_home", self.return_home_callback, 10)

        # ---- Tree-aware collection of uncollected nuts (before return-home) ----
        # For each missed nut we drive to a sweep-entry point ON the nut's sweep
        # line and a little before it, align to the row direction, then sweep
        # the aisle past the nut. The sweeper is on the hug side (start_side) at
        # collect_sweep_offset; the nut is collected by nut_tracker when it
        # passes that sweeper point.
        #
        # The approach point is CLAMPED to the region the robot actually swept
        # (provably free, because it drove there). This stops the run-in being
        # staged in unexplored space behind/beside the start, where indoor walls
        # live - the cause of the "drive into the wall, back up, repeat" loop.
        self.declare_parameter("collect_before_home", True)
        self.declare_parameter("nuts_topic", "/nuts/uncollected")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("collect_sweep_offset", 0.40)   # nut on hug side
        self.declare_parameter("collect_approach_back", 1.00)  # start sweep this far before nut
        self.declare_parameter("collect_sweep_through", 0.60)  # drive this far past the nut
        self.declare_parameter("collect_arrive_tol", 0.15)
        self.declare_parameter("collect_visit_timeout", 45.0)
        # Stuck-escape: abandon a nut after this many avoidance trips, and give
        # up on collection entirely after this many nuts skipped back-to-back.
        self.declare_parameter("collect_max_avoid", 3)
        self.declare_parameter("collect_max_consec_skips", 3)
        self.collect_before_home = bool(self.get_parameter("collect_before_home").value)
        self.nuts_topic = self.get_parameter("nuts_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.collect_sweep_offset = float(self.get_parameter("collect_sweep_offset").value)
        self.collect_approach_back = float(self.get_parameter("collect_approach_back").value)
        self.collect_sweep_through = float(self.get_parameter("collect_sweep_through").value)
        self.collect_arrive_tol = float(self.get_parameter("collect_arrive_tol").value)
        self.collect_visit_timeout = float(self.get_parameter("collect_visit_timeout").value)
        self.collect_max_avoid = int(self.get_parameter("collect_max_avoid").value)
        self.collect_max_consec_skips = int(self.get_parameter("collect_max_consec_skips").value)
        # Inter-aisle moves follow the recorded swept polyline (provably clear)
        # instead of a straight line that could cut across a tree row.
        self.declare_parameter("path_sample_spacing", 0.15)  # record a point every Xm
        self.declare_parameter("path_max_points", 4000)
        self.declare_parameter("path_wp_tol", 0.20)          # progress-snap window
        self.declare_parameter("path_lookahead", 1)          # carrot this many samples ahead
        self.path_sample_spacing = float(self.get_parameter("path_sample_spacing").value)
        self.path_max_points = int(self.get_parameter("path_max_points").value)
        self.path_wp_tol = float(self.get_parameter("path_wp_tol").value)
        self.path_lookahead = int(self.get_parameter("path_lookahead").value)

        self.uncollected_map: List[Tuple[float, float]] = []   # nut positions (map frame)
        self._collect_skip = set()                             # unreachable nuts to ignore
        self._collect_target: Optional[Tuple[float, float]] = None
        self._collect_phase = "EXIT"
        self._collect_deadline: Optional[float] = None
        self._collect_avoid_count = 0                          # avoid trips for this nut
        self._collect_consec_skips = 0                         # nuts skipped back-to-back
        self._path_idx: Optional[int] = None                  # progress along the route
        self._path_goal: Optional[int] = None                 # route endpoint index

        # Bounding box of the swept region, in ROW-FRAME coords (along/across the
        # row). Updated live while sweeping; used to clamp the approach point.
        self._swept_long_min: Optional[float] = None
        self._swept_long_max: Optional[float] = None
        self._swept_lat_min: Optional[float] = None
        self._swept_lat_max: Optional[float] = None
        # The actual driven polyline (odom), recorded while sweeping. This is the
        # known-free corridor used to route inter-aisle collection moves.
        self._swept_path: List[Tuple[float, float]] = []
        # States in which the robot is actively sweeping the field (so its pose
        # marks free ground). Excludes COLLECT_NUTS/RETURN_HOME/AVOID/idle.
        self._SWEEP_STATES = (
            "FOLLOW_OUT", "CLEAR_END", "ARC_TURN", "ALIGN",
            "FOLLOW_BACK", "CLEAR_NEXT", "TURN_NEXT",
        )

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(
            PoseArray, self.nuts_topic, self.uncollected_callback, latched)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_scan: Optional[LaserScan] = None
        self.odom_x: Optional[float] = None
        self.odom_y: Optional[float] = None
        self.odom_yaw: Optional[float] = None

        # Return-home bookkeeping. These are saved at /sweep_start and used
        # after the row/nut mission so the robot drives back to where it began.
        self.home_x: Optional[float] = None
        self.home_y: Optional[float] = None
        self.home_yaw: Optional[float] = None

        self.started = False
        self.state = "WAITING"
        self.state_start_time = self.get_clock().now()
        self.last_row_seen_time = self.get_clock().now()
        self.seen_row_this_pass = False

        # Side the row is on for the CURRENT pass.
        # The arc U-turn LOOPS AROUND the last tree, crossing the row line.
        # After it, the row reappears on the SAME side of the robot - not
        # flipped.
        self.current_side = self.start_side

        # CLEAR_END / arc bookkeeping.
        self.clear_start_x: Optional[float] = None
        self.clear_start_y: Optional[float] = None
        self.arc_start_yaw: Optional[float] = None
        self.arc_last_yaw: Optional[float] = None
        self.arc_accumulated_yaw = 0.0

        # ---------------------------------------------------------------
        # Heading anchor.
        #
        # The robot is placed parallel to the row at /sweep_start. So the
        # odom yaw AT THAT INSTANT is the row's direction in the odom frame.
        # No EMA, no "only update when visible" rule, no drift accumulation
        # from steering wobble - it's a single snapshot.
        #
        # outbound_yaw: row direction when going OUT. Set once at sweep_start.
        # return_target_yaw: outbound_yaw + pi, the heading we want after
        #                    the arc. Computed deterministically.
        # ---------------------------------------------------------------
        self.outbound_yaw: Optional[float] = None
        self.return_target_yaw: Optional[float] = None

        # ALIGN gains
        self.k_align = 1.2
        self.min_align_angular = 0.08

        # -----------------------------
        # Row-following tuning
        # -----------------------------
        self.desired_side_distance = 0.40
        self.too_close_side = 0.25
        # 1.20 m (was 0.80). At a 0.40 m follow offset with 0.70 m tree
        # spacing, 0.80 m only catches a second tree marginally (worst case
        # 0.81 m), so the fit flickered - and the next-row check on the
        # return pass starved. 1.20 m sees 2-3 trees continuously. Cross-row
        # contamination is prevented by select_nearest_row, and
        # next_row_max_dist stops a far row from counting as "next".
        self.declare_parameter("tree_max_range", 1.20)
        self.tree_max_range = float(self.get_parameter("tree_max_range").value)
        self.lidar_min_range = 0.25

        self.emergency_stop_distance = 0.18

        self.forward_speed = 0.06
        self.search_speed = 0.03

        self.k_xtrack = 1.2
        self.k_heading = 1.4
        self.max_follow_angular = 0.30

        self.start_straight_duration = 1.5

        # -----------------------------
        # Row detection (cluster-then-fit)
        # -----------------------------
        self.cluster_gap = 0.15
        self.tree_max_extent = 0.15
        # With several rows in lidar range the robot can see MULTIPLE
        # candidate rows at once. Tree clusters are therefore grouped by
        # their offset perpendicular to the known row direction, and only
        # the NEAREST group on the requested side feeds the line fit - the
        # next row over can never contaminate it. The gap just has to be
        # smaller than the real row spacing (0.35 m suits 0.70 m rows and
        # anything wider).
        self.declare_parameter("row_group_gap", 0.35)
        self.row_group_gap = float(self.get_parameter("row_group_gap").value)
        # Ignore returns within this lateral band of the robot's row axis
        # (they can't be unambiguously assigned to a side).
        self.min_side_offset = 0.05
        self.min_trees_for_fit = 2
        self.front_width = math.radians(35)
        self.max_tree_points = 80

        self.min_pass_duration = 10.0
        self.max_pass_duration = 90.0
        self.row_lost_timeout = 2.0

        # CLEAR_END (0.15 puts the arc centre right at the row end with
        # ~0.25 m centre clearance to the last tree - see arc note below).
        self.declare_parameter("clear_end_distance", 0.15)
        self.clear_end_distance = float(
            self.get_parameter("clear_end_distance").value
        )
        self.clear_end_max_time = 8.0

        # Arc U-turn - TIGHT.
        # Geometry (following at 0.40 m from the row line):
        #   end lateral offset = 0.40 - 2*r from the row line.
        #   r = 0.55 (old) -> ends 0.70 m on the other side = ON the next
        #     row line for 0.70 m spacing. Too wide.
        #   r = 0.40 -> ends 0.40 m on the other side, max forward
        #     excursion clear_end + r = 0.55 m past the last tree, and the
        #     whole arc stays within +/-0.40 m of the row line. With
        #     clear_end = 0.15 the arc centre sits ~0.15 m from the last
        #     tree, so the robot orbits it with r - 0.15 = 0.25 m centre
        #     clearance (robot half-width 0.125 + noodle 0.06 = 0.19 m).
        self.declare_parameter("arc_radius", 0.40)
        self.arc_radius = float(self.get_parameter("arc_radius").value)
        self.arc_linear_speed = 0.06
        self.arc_max_yaw = math.radians(220.0)
        self.arc_distance_gain = 0.6

        nominal_omega = self.arc_linear_speed / self.arc_radius
        self.arc_max_duration = self.arc_max_yaw / nominal_omega + 5.0

        # ALIGN
        self.align_max_angular = 0.30
        self.align_parallel_tol = math.radians(12.0)
        self.align_max_duration = 8.0

        # TURN_NEXT (in-place 180 deg toward the next row). A full 180 at
        # align_max_angular takes ~10.5 s, so this needs its own, longer
        # cap than ALIGN (which only trims the small residual after the arc).
        self.turn_next_max_duration = 18.0

        self.recovery_angular = 0.20

        # -----------------------------
        # RETURN_HOME
        # -----------------------------
        # After the row sweep / nut collection run is finished, the robot
        # returns to the odometry position recorded at /sweep_start.
        self.declare_parameter("return_home_enabled", True)
        self.return_home_enabled = bool(
            self.get_parameter("return_home_enabled").value
        )
        self.declare_parameter("return_goal_tolerance", 0.12)
        self.return_goal_tolerance = float(
            self.get_parameter("return_goal_tolerance").value
        )
        self.declare_parameter("return_yaw_tolerance_deg", 12.0)
        self.return_yaw_tolerance = math.radians(
            float(self.get_parameter("return_yaw_tolerance_deg").value)
        )
        self.declare_parameter("return_max_duration", 120.0)
        self.return_max_duration = float(
            self.get_parameter("return_max_duration").value
        )
        # How far BEYOND the field's near (start) end to exit before crossing
        # to the start point. The robot first backs out of its aisle along the
        # row direction until it clears the tree rows, then drives to home in
        # the open ground - so the cross-over never cuts through a tree row.
        self.declare_parameter("return_exit_margin", 0.60)
        self.return_exit_margin = float(
            self.get_parameter("return_exit_margin").value
        )
        self.return_linear_speed = 0.07
        self.return_max_angular = 0.35
        self.return_heading_slowdown = math.radians(35.0)
        # RETURN_HOME runs in two phases: "OUT" (exit the field) then "GOAL"
        # (drive to the saved start pose).
        self._home_phase = "GOAL"

        # -----------------------------
        # FRONT OBSTACLE AVOIDANCE
        # -----------------------------
        # avoid_front_distance is the early warning distance. When the
        # front LiDAR sees something closer than this, the robot performs
        # a small recovery manoeuvre instead of continuing into the tree.
        self.declare_parameter("avoid_front_distance", 0.28)
        self.avoid_front_distance = float(
            self.get_parameter("avoid_front_distance").value
        )

        self.avoid_backup_speed = -0.04
        self.avoid_turn_speed = 0.28
        self.avoid_forward_speed = 0.05

        self.avoid_backup_duration = 1.2
        self.avoid_turn_duration = 1.8
        self.avoid_forward_duration = 1.8

        self.avoid_previous_state = "FOLLOW_OUT"
        self.avoid_phase = "BACKUP"
        self.avoid_phase_start_time = self.get_clock().now()

        # ================= NEW CODE BLOCK (obstacle recovery) =================
        # Distance-based, closed-loop recovery params, ported from the proven
        # straight_path_avoider. These drive the NEW avoid_front_obstacle():
        #   - reverse only if the REAR is clear (rear_min_clear)
        #   - reverse a measured distance (recover_backup_distance), not a timer
        #   - turn toward the side with more room
        #   - resume ONLY when the FRONT has actually opened up
        #     (avoid_resume_distance) - not after a fixed time
        # The old timed params above (avoid_*_duration, avoid_forward_speed) are
        # left in place but are no longer used by the new logic.
        self.declare_parameter("recover_backup_distance", 0.12)  # m, by odometry
        self.recover_backup_distance = float(
            self.get_parameter("recover_backup_distance").value)
        self.declare_parameter("rear_min_clear", 0.40)           # m rear gap to reverse
        self.rear_min_clear = float(self.get_parameter("rear_min_clear").value)
        self.declare_parameter("avoid_resume_distance", 0.55)    # m front gap to resume
        self.avoid_resume_distance = float(
            self.get_parameter("avoid_resume_distance").value)
        self.declare_parameter("avoid_timeout", 20.0)            # s whole-manoeuvre cap
        self.avoid_timeout = float(self.get_parameter("avoid_timeout").value)
        self._recover_turn = 1.0          # +1 left, -1 right; chosen on entry
        self._recover_backup_x = None     # odom pos where BACKUP started
        self._recover_backup_y = None
        self._avoid_start_time = self.get_clock().now()
        # =============== END NEW CODE BLOCK (obstacle recovery) ===============

        self.timer = self.create_timer(0.1, self.control_loop)

        self.publish_status("Simple row follower ready. Publish /sweep_start to begin.")
        self.get_logger().info(
            f"Ready (multi-row). CLEAR={self.clear_end_distance:.2f}m, "
            f"ARC r={self.arc_radius:.2f}m v={self.arc_linear_speed:.2f}m/s, "
            f"start_side={self.start_side}, "
            f"max_rows={self.max_rows or 'unlimited'}, "
            f"row_group_gap={self.row_group_gap:.2f}m, "
            f"lidar_yaw_offset={math.degrees(self.lidar_yaw_offset):+.0f}deg, "
            f"return_home={self.return_home_enabled}, manual_topic=/return_home, "
            f"avoid_front={self.avoid_front_distance:.2f}m"
        )

    # -----------------------------
    # Callbacks
    # -----------------------------

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny, cosy)

    def start_callback(self, _msg: Empty):
        # GUARD: `ros2 topic pub /sweep_start ...` WITHOUT `--once` keeps
        # publishing at 1 Hz; each repeat would re-snapshot the heading
        # anchor at the robot's CURRENT yaw (garbage mid-turn) and reset
        # the state machine. Over a multi-row mission that is fatal, so
        # ignore restarts while running.
        if self.started:
            self.get_logger().warn(
                "sweep_start ignored - sweep already running. "
                "Publish /sweep_stop first to restart."
            )
            return
        self.started = True
        self.rows_completed = 0
        self.next_row_hits = 0
        self.state = "FOLLOW_OUT"
        self.state_start_time = self.get_clock().now()
        self.last_row_seen_time = self.get_clock().now()
        self.seen_row_this_pass = False
        self.current_side = self.start_side
        self.clear_start_x = None
        self.clear_start_y = None
        self.arc_start_yaw = None
        self.arc_last_yaw = None
        self.arc_accumulated_yaw = 0.0
        # Fresh run: discard the previous sweep's free-space record.
        self._swept_path = []
        self._swept_long_min = self._swept_long_max = None
        self._swept_lat_min = self._swept_lat_max = None

        # Save the starting pose for RETURN_HOME. If odom is not ready yet,
        # control_loop will lazily fill this before the robot has moved far.
        if self.odom_x is not None and self.odom_y is not None:
            self.home_x = self.odom_x
            self.home_y = self.odom_y
            self.home_yaw = self.odom_yaw
            self.get_logger().info(
                f"Home pose saved: x={self.home_x:+.2f}, y={self.home_y:+.2f}, "
                f"yaw={math.degrees(self.home_yaw or 0.0):+.1f}deg"
            )
        else:
            self.home_x = None
            self.home_y = None
            self.home_yaw = None
            self.get_logger().warn(
                "Odom not ready at sweep_start - home pose will be saved on first odom tick."
            )

        # THE key change: snapshot the row direction NOW. The robot is
        # parallel to the row at this instant, so odom_yaw is the row's
        # direction in odom. Return heading is exactly the opposite.
        if self.odom_yaw is not None:
            self.outbound_yaw = self.odom_yaw
            self.return_target_yaw = self.normalize_angle(self.odom_yaw + math.pi)
            self.get_logger().info(
                f"Anchored row heading: outbound={math.degrees(self.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(self.return_target_yaw):+.1f}deg"
            )
        else:
            # Odom not ready yet - will be set lazily by control_loop on
            # the first tick odom arrives.
            self.outbound_yaw = None
            self.return_target_yaw = None
            self.get_logger().warn(
                "Odom not ready at sweep_start - heading anchor will be set on first odom tick."
            )

        self.publish_status(f"Started: line-fit following on {self.current_side.upper()}")
        self.get_logger().info(f"Sweep start received (side={self.current_side})")

    def stop_callback(self, _msg: Empty):
        self.started = False
        self.state = "STOPPED"
        self.stop_robot()
        self.publish_status("Manual stop received")
        self.get_logger().warn("Sweep stop received")

    def return_home_callback(self, _msg: Empty):
        """Manual trigger: publish /return_home to abandon the current sweep and drive home."""
        # If home was not saved at /sweep_start because odometry was late,
        # try to save it now only if the robot has not started moving yet.
        # Normally this will already be set by start_callback/control_loop.
        if self.home_x is None or self.home_y is None:
            self.get_logger().warn(
                "return_home requested, but no home pose has been saved. "
                "Publish /sweep_start first so the robot knows where home is."
            )
            self.publish_status("Cannot RETURN_HOME: no saved home pose. Publish /sweep_start first.")
            return

        self.started = True
        self.next_row_hits = 0
        self.clear_start_x = None
        self.clear_start_y = None
        self.arc_start_yaw = None
        self.arc_last_yaw = None
        self.arc_accumulated_yaw = 0.0
        self.stop_robot()
        self._home_phase = "OUT"
        self.set_state(
            "RETURN_HOME",
            f"Manual return-home requested. Exiting the field, then returning to "
            f"x={self.home_x:+.2f}, y={self.home_y:+.2f}."
        )
        self.get_logger().warn("Manual return-home requested")

    # -----------------------------
    # Helpers
    # -----------------------------

    def publish_status(self, text: str):
        m = String()
        m.data = text
        self.status_pub.publish(m)

    def elapsed_in_state(self) -> float:
        return (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9

    def time_since_row_seen(self) -> float:
        return (self.get_clock().now() - self.last_row_seen_time).nanoseconds / 1e9

    def set_state(self, new_state: str, status: str = ""):
        self.state = new_state
        self.state_start_time = self.get_clock().now()
        if status:
            self.publish_status(status)
        self.get_logger().info(f"State changed to {new_state}")

    def publish_cmd(self, linear: float, angular: float):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.twist.linear.x = float(linear)
        m.twist.angular.z = float(angular)
        self.cmd_pub.publish(m)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def normalize_angle(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def valid_range(self, r: float, scan: LaserScan) -> bool:
        return (
            not math.isnan(r)
            and not math.isinf(r)
            and scan.range_min <= r <= scan.range_max
            and r >= self.lidar_min_range
        )

    def collect_row_side_points(self, side: str) -> List[Tuple[float, float, float]]:
        scan = self.latest_scan
        if scan is None:
            return []
        if side == "right":
            min_a = math.radians(-170.0)
            max_a = math.radians(20.0)
        else:
            min_a = math.radians(-20.0)
            max_a = math.radians(170.0)

        pts: List[Tuple[float, float, float]] = []
        for i, r in enumerate(scan.ranges):
            # Apply lidar mounting offset: raw scan angle PLUS the offset
            # gives the angle in the robot's base_link frame (forward = 0).
            angle = scan.angle_min + i * scan.angle_increment + self.lidar_yaw_offset
            angle = self.normalize_angle(angle)
            if not (min_a <= angle <= max_a):
                continue
            if not self.valid_range(r, scan):
                continue
            if r > self.tree_max_range:
                continue
            pts.append((r * math.cos(angle), r * math.sin(angle), angle))
        pts.sort(key=lambda p: p[2])
        return pts

    def cluster_to_trees(
        self, pts: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, int]]:
        if not pts:
            return []
        clusters: List[List[Tuple[float, float, float]]] = []
        current: List[Tuple[float, float, float]] = [pts[0]]
        for prev, p in zip(pts, pts[1:]):
            d = math.hypot(p[0] - prev[0], p[1] - prev[1])
            if d <= self.cluster_gap:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)

        trees: List[Tuple[float, float, int]] = []
        for c in clusters:
            xs = [q[0] for q in c]
            ys = [q[1] for q in c]
            extent = max(max(xs) - min(xs), max(ys) - min(ys))
            if extent > self.tree_max_extent:
                continue
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            trees.append((mx, my, len(c)))
        return trees

    @staticmethod
    def opposite(side: str) -> str:
        return "left" if side == "right" else "right"

    def row_dir_robot_frame(self) -> Optional[float]:
        """Known row direction expressed in the robot frame.

        The heading anchor gives the row direction in odom; subtracting the
        current yaw moves it into base_link. A row is a LINE, so the value
        only matters mod pi - that is fine for projecting trees onto the
        across-row axis, and it stays valid in any robot orientation,
        including halfway around the U-turn.
        """
        if self.outbound_yaw is None or self.odom_yaw is None:
            return None
        return self.normalize_angle(self.outbound_yaw - self.odom_yaw)

    def select_nearest_row(
        self, trees: List[Tuple[float, float, int]], side: str
    ) -> List[Tuple[float, float, int]]:
        """Keep only the NEAREST row of trees on the requested side.

        With several rows in range the raw cluster list may contain trees
        from two (or more) parallel rows. Each tree is projected onto the
        axis perpendicular to the known row direction; trees are grouped
        along that axis (1-D clustering with row_group_gap) and the group
        with the smallest mean |offset| on the correct side wins. Trees of
        the next row sit a full row-spacing away on that axis, so they end
        up in a different group and never contaminate the fit.
        """
        if not trees:
            return []
        a = self.row_dir_robot_frame()
        if a is None:
            a = 0.0  # robot parallel to row (true at start, before anchor)
        # A row is a LINE - its direction is only defined mod pi. Use the
        # representative closest to the robot's forward axis so that
        # perp > 0 always means the ROBOT's left ('side' is a robot-frame
        # concept). Without this, the sign flips on the return pass and
        # every tree gets assigned to the wrong side.
        if a > math.pi / 2.0:
            a -= math.pi
        elif a < -math.pi / 2.0:
            a += math.pi
        sin_a = math.sin(a)
        cos_a = math.cos(a)

        tagged: List[Tuple[float, float, float, int]] = []
        for (tx, ty, n) in trees:
            # perp > 0 -> left of the row axis through the robot.
            perp = -tx * sin_a + ty * cos_a
            if side == "right" and perp > -self.min_side_offset:
                continue
            if side == "left" and perp < self.min_side_offset:
                continue
            tagged.append((perp, tx, ty, n))
        if not tagged:
            return []

        tagged.sort(key=lambda t: t[0])
        groups: List[List[Tuple[float, float, float, int]]] = [[tagged[0]]]
        for prev, cur in zip(tagged, tagged[1:]):
            if abs(cur[0] - prev[0]) <= self.row_group_gap:
                groups[-1].append(cur)
            else:
                groups.append([cur])

        best = min(
            groups,
            key=lambda g: abs(sum(t[0] for t in g) / len(g)),
        )
        return [(tx, ty, n) for (_, tx, ty, n) in best]

    def fit_row_line(self, side: str) -> Optional[Tuple[float, float, int]]:
        pts = self.collect_row_side_points(side)
        trees = self.cluster_to_trees(pts)
        trees = self.select_nearest_row(trees, side)
        if len(trees) < self.min_trees_for_fit:
            return None
        cx = [t[0] for t in trees]
        cy = [t[1] for t in trees]
        n = len(trees)
        sx = sum(cx)
        sy = sum(cy)
        sxx = sum(x * x for x in cx)
        sxy = sum(x * y for x, y in zip(cx, cy))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-6:
            mean_y = sy / n
            return (math.pi / 2.0, mean_y, n)
        m = (n * sxy - sx * sy) / denom
        c = (sy - m * sx) / n
        line_angle = math.atan(m)
        perp = c / math.sqrt(1.0 + m * m)
        return (line_angle, perp, n)

    def side_cone_points(self, side: str) -> List[Tuple[float, float]]:
        pts = self.collect_row_side_points(side)
        trees = self.cluster_to_trees(pts)
        # Same nearest-row gating as the fit, so "last tree abeam" is
        # judged against the CURRENT row only, not a neighbouring one.
        trees = self.select_nearest_row(trees, side)
        return [(t[0], t[1]) for t in trees]

    def get_front_distance(self) -> float:
        scan = self.latest_scan
        if scan is None:
            return 10.0
        half_w = self.front_width / 2.0
        vals = []
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment + self.lidar_yaw_offset
            if abs(self.normalize_angle(angle - 0.0)) > half_w:
                continue
            if not self.valid_range(r, scan):
                continue
            if r > 3.0:
                continue
            vals.append(r)
        return min(vals) if vals else 10.0

    def row_visible(self, side: str) -> Tuple[bool, Optional[Tuple[float, float, int]]]:
        fit = self.fit_row_line(side)
        return (fit is not None), fit

    def last_tree_abeam_or_behind(self, side: str) -> bool:
        if not self.seen_row_this_pass:
            return False
        pts = self.side_cone_points(side)
        if not pts:
            return False
        ahead_threshold = 0.05
        max_x = max(p[0] for p in pts)
        return max_x <= ahead_threshold

    def mark_row_seen_if_visible(self, visible: bool):
        if visible:
            self.last_row_seen_time = self.get_clock().now()
            self.seen_row_this_pass = True

    # -----------------------------
    # Line-fit follow
    # -----------------------------

    def follow_side(self, side: str):
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP: front={front:.2f} m")
            return

        if front < self.avoid_front_distance:
            self.enter_avoid_front(
                self.state,
                f"front={front:.2f}m"
            )
            return

        visible, fit = self.row_visible(side)
        self.mark_row_seen_if_visible(visible)

        if self.elapsed_in_state() < self.start_straight_duration:
            self.publish_cmd(self.forward_speed, 0.0)
            self.publish_status(
                f"START STRAIGHT | row_visible={visible} | front={front:.2f}m"
            )
            return

        if not visible:
            self.publish_cmd(self.search_speed, 0.0)
            self.publish_status(
                f"ROW NOT CONFIRMED {side.upper()} | creeping forward | "
                f"front={front:.2f}m"
            )
            return

        line_angle, perp, n = fit  # type: ignore[assignment]

        desired_perp = (-self.desired_side_distance
                        if side == "right" else self.desired_side_distance)
        move_left = perp - desired_perp
        heading_error = line_angle

        angular = self.k_xtrack * move_left + self.k_heading * heading_error
        angular = max(-self.max_follow_angular,
                      min(self.max_follow_angular, angular))
        linear = self.forward_speed

        actual_dist = abs(perp)
        if actual_dist < self.too_close_side:
            linear = self.search_speed
            status = f"TOO CLOSE {side.upper()} d={actual_dist:.2f}m"
        else:
            status = f"FOLLOWING {side.upper()} d={actual_dist:.2f}m"

        self.publish_cmd(linear, angular)
        self.publish_status(
            f"{status} | "
            f"perp={perp:+.2f} target={desired_perp:+.2f} | "
            f"line_ang={math.degrees(line_angle):+.0f}deg | "
            f"n={n} | front={front:.2f}m | ang={angular:+.2f}"
        )

    def should_end_pass(self) -> bool:
        elapsed = self.elapsed_in_state()
        if elapsed < self.min_pass_duration:
            return False
        if self.seen_row_this_pass and self.time_since_row_seen() >= self.row_lost_timeout:
            return True
        if elapsed >= self.max_pass_duration:
            return True
        return False

    # -----------------------------
    # CLEAR_END
    # -----------------------------

    def clear_straight(self, next_state: str, label: str):
        """Drive straight until clear_end_distance past the row end, then
        hand over to next_state. Used both at the FAR end (-> ARC_TURN)
        and at the START end before a row transition (-> TURN_NEXT)."""
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during {label}: front={front:.2f}m")
            return

        if self.clear_start_x is None and self.odom_x is not None:
            self.clear_start_x = self.odom_x
            self.clear_start_y = self.odom_y

        if self.clear_start_x is None or self.odom_x is None:
            self.publish_cmd(self.forward_speed, 0.0)
            self.publish_status(f"{label} (no odom yet)")
            return

        dx = self.odom_x - self.clear_start_x
        dy = self.odom_y - self.clear_start_y
        traveled = math.hypot(dx, dy)

        if traveled >= self.clear_end_distance:
            self.set_state(
                next_state,
                f"Cleared {traveled:.2f}m past row end. -> {next_state}."
            )
            return

        if self.elapsed_in_state() > self.clear_end_max_time:
            self.set_state(
                next_state,
                f"{label} time cap, -> {next_state} anyway."
            )
            return

        self.publish_cmd(self.forward_speed, 0.0)
        self.publish_status(
            f"{label} {traveled:.2f}/{self.clear_end_distance:.2f}m | front={front:.2f}m"
        )

    # -----------------------------
    # Arc U-turn
    # -----------------------------

    def arc_turn(self):
        sign = -1.0 if self.current_side == "right" else +1.0

        if self.odom_yaw is not None:
            if self.arc_last_yaw is None:
                self.arc_start_yaw = self.odom_yaw
                self.arc_last_yaw = self.odom_yaw
            else:
                # SIGNED accumulation in the commanded turn direction:
                # abs() would count yaw NOISE as progress (a "180 deg" arc
                # can exit after ~120 deg of true rotation with just 1 deg
                # of per-reading jitter). Signed deltas let jitter cancel.
                d = self.normalize_angle(self.odom_yaw - self.arc_last_yaw)
                self.arc_accumulated_yaw += sign * d
                self.arc_last_yaw = self.odom_yaw

        yaw_deg = math.degrees(self.arc_accumulated_yaw)

        v = self.arc_linear_speed
        base_omega = v / self.arc_radius

        # PURE ODOMETRY ARC - no lidar correction.
        # The old lidar distance-correction measured "distance to the row"
        # mid-arc, but halfway around the turn the robot straddles the row
        # line: its own row's trees are excluded (|perp| ~ 0) and the
        # NEIGHBOURING row becomes the nearest fit, so the correction
        # tightened the turn into the last tree (offline sim showed a
        # near-collision, 3 cm clearance). A constant-omega arc has clean,
        # provable geometry: with clear_end=0.15 and r=0.40 the robot
        # orbits the last tree with ~0.25 m centre clearance.
        omega = base_omega

        # PRIMARY exit: rotated 180 deg from arc start.
        arc_target_yaw = math.pi
        if self.arc_accumulated_yaw >= arc_target_yaw:
            self.set_state(
                "ALIGN",
                f"Arc 180 deg complete at +{yaw_deg:.0f}. Aligning to row."
            )
            return

        # Safety cap.
        if (self.arc_accumulated_yaw >= self.arc_max_yaw
                or self.elapsed_in_state() >= self.arc_max_duration):
            self.set_state(
                "ALIGN",
                f"Arc capped at {yaw_deg:.0f} deg. Aligning to row."
            )
            return

        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.publish_cmd(0.0, sign * base_omega)
            self.publish_status(
                f"ARC (front block {front:.2f}m) | yaw +{yaw_deg:.0f}deg"
            )
            return

        self.publish_cmd(v, sign * omega)
        self.publish_status(
            f"ARC | yaw +{yaw_deg:.0f}/180deg | r={self.arc_radius:.2f}m | "
            f"front={front:.2f}m"
        )

    # -----------------------------
    # ALIGN - heading-anchored. NO line-fit dependency.
    # -----------------------------

    def align_to_row(self):
        """Rotate in place until odom_yaw matches return_target_yaw.

        We ONLY use the heading anchor recorded at /sweep_start. No line fit,
        no perception ambiguity. Geometric guarantee:
          - At sweep_start, robot was parallel to row, facing along OUT.
          - return_target_yaw = sweep_start_yaw + pi exactly.
          - When odom_yaw matches return_target_yaw, robot is parallel to
            the row, facing along BACK. Period.

        Previously this state mixed in the line-fit angle for "fine tuning",
        but line_angle is in (-90, +90) and can't disambiguate the two
        parallel orientations - that's what made the robot settle facing
        the wrong way. Heading-only termination is unambiguous.
        """
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during ALIGN: front={front:.2f}m")
            return

        if self.return_target_yaw is None or self.odom_yaw is None:
            # Should not happen because we snapshot at sweep_start, but if
            # it does, abort gracefully.
            self.publish_cmd(0.0, 0.0)
            self.publish_status(
                "ALIGN cannot proceed: missing heading anchor or odom."
            )
            if self.elapsed_in_state() > self.align_max_duration:
                self._enter_follow_back("ALIGN aborted - no heading reference.")
            return

        # Pure heading control. err > 0 -> target is "more CCW" -> turn left.
        err = self.normalize_angle(self.return_target_yaw - self.odom_yaw)

        if abs(err) < self.align_parallel_tol:
            self._enter_follow_back(
                f"ALIGN done. odom_yaw={math.degrees(self.odom_yaw):+.1f} "
                f"target={math.degrees(self.return_target_yaw):+.1f} "
                f"err={math.degrees(err):+.1f}deg. Sweeping back."
            )
            return

        mag = min(self.align_max_angular,
                  max(self.min_align_angular, self.k_align * abs(err)))
        angular = math.copysign(mag, err)
        self.publish_cmd(0.0, angular)
        self.publish_status(
            f"ALIGN | odom={math.degrees(self.odom_yaw):+.0f}deg "
            f"target={math.degrees(self.return_target_yaw):+.0f}deg "
            f"err={math.degrees(err):+.0f}deg (tol "
            f"{math.degrees(self.align_parallel_tol):.0f}) | rot={angular:+.2f}"
        )

        if self.elapsed_in_state() > self.align_max_duration:
            self._enter_follow_back(
                f"ALIGN timeout. err={math.degrees(err):+.0f}deg."
            )

    def _enter_follow_back(self, status: str):
        self.last_row_seen_time = self.get_clock().now()
        self.seen_row_this_pass = False
        self.next_row_hits = 0  # fresh count for THIS return pass
        self.set_state("FOLLOW_BACK", status)

    # -----------------------------
    # TURN_NEXT - row transition (in-place 180 deg toward the next row)
    # -----------------------------

    def turn_to_next_row(self):
        """Rotate in place back to the OUTBOUND heading to start the next row.

        Geometry: at the end of FOLLOW_BACK the robot is just past the
        start end of the finished row, heading along the RETURN direction,
        with the finished row on current_side and the next row on the
        opposite side (~row spacing minus follow offset away). Rotating
        180 deg in place toward the next row leaves the robot heading
        OUTBOUND with the next row on current_side - exactly the start
        configuration of a normal row sweep. In place = zero turning
        radius, so nothing in the corridor can be hit.

        The turn direction is FORCED toward the next row's side (away from
        the finished row) so the maneuver is predictable.
        """
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during TURN_NEXT: front={front:.2f}m")
            return

        if self.outbound_yaw is None or self.odom_yaw is None:
            self.stop_robot()
            self.publish_status("TURN_NEXT cannot proceed: no heading reference.")
            if self.elapsed_in_state() > self.turn_next_max_duration:
                self._begin_next_row("TURN_NEXT aborted blind - trying next row.")
            return

        err = self.normalize_angle(self.outbound_yaw - self.odom_yaw)

        if abs(err) < self.align_parallel_tol:
            self._begin_next_row(
                f"TURN_NEXT done (err={math.degrees(err):+.1f}deg). "
                f"Starting row {self.rows_completed + 1}."
            )
            return

        if self.elapsed_in_state() > self.turn_next_max_duration:
            self._begin_next_row(
                f"TURN_NEXT timeout (err={math.degrees(err):+.0f}deg). "
                f"Starting row {self.rows_completed + 1} anyway."
            )
            return

        # Forced direction: next row is on the OPPOSITE side of the
        # finished row. Finished row on the right -> next row on the left
        # -> rotate CCW (+), and vice versa.
        turn_dir = +1.0 if self.current_side == "right" else -1.0
        mag = min(self.align_max_angular,
                  max(self.min_align_angular, self.k_align * abs(err)))
        self.publish_cmd(0.0, turn_dir * mag)
        self.publish_status(
            f"TURN_NEXT | odom={math.degrees(self.odom_yaw):+.0f}deg "
            f"target={math.degrees(self.outbound_yaw):+.0f}deg "
            f"err={math.degrees(err):+.0f}deg | rot={turn_dir * mag:+.2f}"
        )

    def _begin_next_row(self, status: str):
        """Reset per-row state and start the outbound pass on the next row."""
        self.next_row_hits = 0
        self.seen_row_this_pass = False
        self.last_row_seen_time = self.get_clock().now()
        self.clear_start_x = None
        self.clear_start_y = None
        self.arc_start_yaw = None
        self.arc_last_yaw = None
        self.arc_accumulated_yaw = 0.0
        # current_side is UNCHANGED: after the in-place 180 the next row
        # sits on the same commanded side as the previous one did.
        self.set_state("FOLLOW_OUT", status)

    # -----------------------------
    # FRONT OBSTACLE AVOIDANCE
    # -----------------------------

    def enter_avoid_front(self, from_state: str, reason: str = ""):
        # ================= NEW CODE BLOCK (enter recovery) =================
        # On entry: pick the turn direction toward the side with more room,
        # and only START in BACKUP if the rear is clear (else go straight to
        # turning). The actual motion is closed-loop in avoid_front_obstacle().
        self.avoid_previous_state = from_state
        self._avoid_start_time = self.get_clock().now()

        # choose turn direction toward the side with more free space
        left_clear = self.get_sector_distance(25, 90)
        right_clear = self.get_sector_distance(-90, -25)
        self._recover_turn = +1.0 if left_clear >= right_clear else -1.0

        # back up only if the rear is clear; otherwise just turn
        rear = self.rear_clearance()
        if rear >= self.rear_min_clear:
            self.avoid_phase = "BACKUP"
            self._recover_backup_x = self.odom_x
            self._recover_backup_y = self.odom_y
        else:
            self.avoid_phase = "TURN"

        self.avoid_phase_start_time = self.get_clock().now()
        self.stop_robot()
        side = "LEFT" if self._recover_turn > 0 else "RIGHT"
        self.set_state(
            "AVOID_FRONT",
            f"Front obstacle ({reason}). Recovering: "
            f"{'BACKUP then ' if self.avoid_phase == 'BACKUP' else ''}"
            f"TURN {side}, then resume {from_state}.")
        # =============== END NEW CODE BLOCK (enter recovery) ===============

    def elapsed_in_avoid_phase(self) -> float:
        return (self.get_clock().now() - self.avoid_phase_start_time).nanoseconds / 1e9

    def set_avoid_phase(self, phase: str):
        self.avoid_phase = phase
        self.avoid_phase_start_time = self.get_clock().now()
        self.get_logger().info(f"AVOID_FRONT phase changed to {phase}")

    def get_sector_distance(self, min_deg: float, max_deg: float) -> float:
        """Return nearest valid LiDAR distance in an angle sector."""
        scan = self.latest_scan
        if scan is None:
            return 10.0

        vals = []
        min_a = math.radians(min_deg)
        max_a = math.radians(max_deg)

        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment + self.lidar_yaw_offset
            angle = self.normalize_angle(angle)

            if min_a <= angle <= max_a and self.valid_range(r, scan):
                vals.append(r)

        return min(vals) if vals else 10.0

    # ================= NEW CODE BLOCK (rear clearance) =================
    # Closest valid return in the REAR cone (+/-30 deg around 180 deg).
    # Used to decide whether it's safe to reverse during recovery.
    def rear_clearance(self) -> float:
        scan = self.latest_scan
        if scan is None:
            return 10.0
        best = 10.0
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment + self.lidar_yaw_offset
            angle = self.normalize_angle(angle)
            if abs(self.normalize_angle(angle - math.pi)) > math.radians(30.0):
                continue
            if not self.valid_range(r, scan):
                continue
            if r > 3.0:
                continue
            best = min(best, r)
        return best
    # =============== END NEW CODE BLOCK (rear clearance) ===============

    def avoid_front_obstacle(self):
        # ================= NEW CODE BLOCK (closed-loop recovery) =================
        # Replaces the old timed BACKUP/TURN_AWAY/FORWARD_CLEAR sequence with the
        # proven straight_path_avoider logic:
        #   BACKUP  - reverse a measured distance, only while the rear stays clear
        #   TURN    - rotate in place toward the side with more room
        #   resume  - the moment the FRONT actually opens past avoid_resume_distance,
        #             return to the interrupted state (FOLLOW_OUT/BACK, COLLECT_NUTS,
        #             RETURN_HOME), which re-acquires its path and rejoins.
        #   timeout - whole manoeuvre capped by avoid_timeout -> STOPPED.
        front = self.get_front_distance()
        rear = self.rear_clearance()

        # whole-manoeuvre timeout
        elapsed = (self.get_clock().now() - self._avoid_start_time).nanoseconds / 1e9
        if elapsed > self.avoid_timeout:
            self.get_logger().warn("AVOID_FRONT timed out. Stopping.")
            self.set_state("STOPPED",
                           "AVOID_FRONT timeout - could not clear obstacle.")
            self.stop_robot()
            return

        # front actually clear -> resume whatever we were doing
        if front > self.avoid_resume_distance:
            self.last_row_seen_time = self.get_clock().now()
            self.seen_row_this_pass = False
            self.set_state(
                self.avoid_previous_state,
                f"Obstacle cleared (front={front:.2f}m). "
                f"Resuming {self.avoid_previous_state}.")
            return

        turn = self._recover_turn

        # ---- PHASE 1: BACKUP (only while the rear is clear) ----
        if self.avoid_phase == "BACKUP":
            if rear < self.rear_min_clear:
                self.set_avoid_phase("TURN")
                self.publish_status(
                    f"AVOID_FRONT rear blocked ({rear:.2f}m) -> TURN only")
                return
            backed = 0.0
            if self._recover_backup_x is not None and self.odom_x is not None:
                backed = math.hypot(self.odom_x - self._recover_backup_x,
                                    self.odom_y - self._recover_backup_y)
            if backed >= self.recover_backup_distance:
                self.set_avoid_phase("TURN")
                self.publish_status(f"AVOID_FRONT backed {backed:.2f}m -> TURN")
                return
            self.publish_cmd(-abs(self.avoid_backup_speed), 0.0)
            self.publish_status(
                f"AVOID_FRONT BACKUP {backed:.2f}/{self.recover_backup_distance:.2f}m "
                f"| front={front:.2f}m rear={rear:.2f}m")
            return

        # ---- PHASE 2: TURN toward the clearer side ----
        # rotate in place; the resume check at the top ends this phase when the
        # front opens up.
        self.publish_cmd(0.0, turn * self.avoid_turn_speed)
        side = "LEFT" if turn > 0 else "RIGHT"
        self.publish_status(
            f"AVOID_FRONT TURN {side} | front={front:.2f}m rear={rear:.2f}m "
            f"(resume when front>{self.avoid_resume_distance:.2f}m)")
        return
        # =============== END NEW CODE BLOCK (closed-loop recovery) ===============

    # -----------------------------
    # RETURN_HOME
    # -----------------------------

    def finish_or_return_home(self, reason: str):
        """Collect any uncollected nuts first (if enabled), then return home."""
        self.stop_robot()
        if self.collect_before_home and self.uncollected_map:
            self._collect_skip = set()
            self._collect_target = None
            self._collect_phase = "EXIT"
            self._collect_deadline = None
            self._collect_avoid_count = 0
            self._collect_consec_skips = 0
            self._path_idx = None
            self._path_goal = None
            self.set_state(
                "COLLECT_NUTS",
                reason + f" Collecting {len(self.uncollected_map)} uncollected nut(s) "
                f"before home."
            )
            return
        self._go_home(reason)

    def _go_home(self, reason: str):
        """Drive to the saved start pose if return-home is enabled, else finish."""
        self.stop_robot()
        if self.return_home_enabled and self.home_x is not None and self.home_y is not None:
            self._home_phase = "OUT"
            self.set_state(
                "RETURN_HOME",
                reason + f" Exiting the field, then returning to start "
                f"x={self.home_x:+.2f}, y={self.home_y:+.2f}."
            )
        else:
            self.set_state("DONE", reason + " Mission complete.")

    def return_home(self):
        """Drive back to the odometry pose recorded at /sweep_start."""
        if self.home_x is None or self.home_y is None:
            self.set_state("DONE", "No home pose saved. Robot stopped.")
            self.stop_robot()
            return

        if self.odom_x is None or self.odom_y is None or self.odom_yaw is None:
            self.stop_robot()
            self.publish_status("RETURN_HOME waiting for odometry")
            return

        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during RETURN_HOME: front={front:.2f}m")
            return

        if front < self.avoid_front_distance:
            self.enter_avoid_front(
                "RETURN_HOME",
                f"front={front:.2f}m while returning home"
            )
            return

        # Phase 1 (OUT): back out of the field along the row direction until the
        # robot is clear of the near (start) end, so the cross-over to the start
        # point happens in open ground instead of cutting across the tree rows.
        # Needs the row-direction anchor; without it, skip straight to GOAL.
        if self._home_phase == "OUT":
            if self.outbound_yaw is None:
                self._home_phase = "GOAL"
            else:
                ddx, ddy = math.cos(self.outbound_yaw), math.sin(self.outbound_yaw)
                home_long = self.home_x * ddx + self.home_y * ddy
                robot_long = self.odom_x * ddx + self.odom_y * ddy
                out_long = home_long - self.return_exit_margin
                if robot_long <= out_long + self.return_goal_tolerance:
                    # Already clear of the field's near end.
                    self._home_phase = "GOAL"
                else:
                    # Target: current position projected back to the exit line,
                    # so this phase drives purely along the aisle (no lateral
                    # correction that would steer into a row).
                    step = out_long - robot_long      # negative -> along -row dir
                    tx = self.odom_x + step * ddx
                    ty = self.odom_y + step * ddy
                    ex = tx - self.odom_x
                    ey = ty - self.odom_y
                    edist = math.hypot(ex, ey)
                    target_yaw = math.atan2(ey, ex)
                    heading_error = self.normalize_angle(target_yaw - self.odom_yaw)
                    if abs(heading_error) > self.return_heading_slowdown:
                        self.publish_cmd(
                            0.0, math.copysign(self.return_max_angular, heading_error))
                    else:
                        linear = min(self.return_linear_speed, max(0.03, 0.45 * edist))
                        angular = max(-self.return_max_angular,
                                      min(self.return_max_angular, 1.4 * heading_error))
                        self.publish_cmd(linear, angular)
                    self.publish_status(
                        f"RETURN_HOME exiting field | out_dist={edist:.2f}m "
                        f"| heading_err={math.degrees(heading_error):+.0f}deg "
                        f"| front={front:.2f}m"
                    )
                    if self.elapsed_in_state() > self.return_max_duration:
                        self.set_state(
                            "DONE",
                            f"RETURN_HOME timeout after {self.return_max_duration:.0f}s "
                            f"while exiting field. Robot stopped."
                        )
                        self.stop_robot()
                    return

        dx = self.home_x - self.odom_x
        dy = self.home_y - self.odom_y
        dist = math.hypot(dx, dy)

        # Phase 2 (GOAL): reach the saved x/y position.
        if dist > self.return_goal_tolerance:
            target_yaw = math.atan2(dy, dx)
            heading_error = self.normalize_angle(target_yaw - self.odom_yaw)

            # If facing far away from home, rotate first. Otherwise drive forward
            # with proportional heading correction. This avoids driving a big arc.
            if abs(heading_error) > self.return_heading_slowdown:
                linear = 0.0
                angular = math.copysign(
                    self.return_max_angular,
                    heading_error,
                )
                mode = "turning toward start"
            else:
                linear = min(self.return_linear_speed, max(0.03, 0.45 * dist))
                angular = max(
                    -self.return_max_angular,
                    min(self.return_max_angular, 1.4 * heading_error),
                )
                mode = "driving to start"

            self.publish_cmd(linear, angular)
            self.publish_status(
                f"RETURN_HOME {mode} | dist={dist:.2f}m "
                f"| heading_err={math.degrees(heading_error):+.0f}deg "
                f"| front={front:.2f}m"
            )
            if self.elapsed_in_state() > self.return_max_duration:
                self.set_state(
                    "DONE",
                    f"RETURN_HOME timeout after {self.return_max_duration:.0f}s. Robot stopped."
                )
                self.stop_robot()
            return

        # At x/y start. Rotate back to the original start heading if known.
        if self.home_yaw is not None:
            yaw_error = self.normalize_angle(self.home_yaw - self.odom_yaw)
            if abs(yaw_error) > self.return_yaw_tolerance:
                angular = max(
                    -self.return_max_angular,
                    min(self.return_max_angular, 1.2 * yaw_error),
                )
                if abs(angular) < self.min_align_angular:
                    angular = math.copysign(self.min_align_angular, yaw_error)
                self.publish_cmd(0.0, angular)
                self.publish_status(
                    f"RETURN_HOME at start, aligning yaw | "
                    f"yaw_err={math.degrees(yaw_error):+.0f}deg"
                )
                return

        self.stop_robot()
        self.set_state(
            "DONE",
            f"Returned to start. Final distance={dist:.2f}m. Mission complete."
        )

    # -----------------------------
    # COLLECT_NUTS - tree-aware pickup of missed nuts before home
    # -----------------------------

    def uncollected_callback(self, msg: PoseArray):
        self.uncollected_map = [(p.position.x, p.position.y) for p in msg.poses]

    def _map_to_odom(self):
        """(tx, ty, cos, sin) of the odom<-map transform, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(self.odom_frame, self.map_frame, Time())
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, math.cos(yaw), math.sin(yaw))

    @staticmethod
    def _apply_tf(mx, my, tf):
        tx, ty, c, s = tf
        return (tx + c * mx - s * my, ty + s * mx + c * my)

    def _uncollected_targets(self, tf):
        out = []
        for (mx, my) in self.uncollected_map:
            if (round(mx, 2), round(my, 2)) in self._collect_skip:
                continue
            out.append(((mx, my), self._apply_tf(mx, my, tf)))
        return out

    def _row_units(self):
        """(d, sweeper_unit) in odom: d along the row, sweeper_unit toward the
        hug side. None if the heading anchor isn't set."""
        if self.outbound_yaw is None:
            return None
        th = self.outbound_yaw
        d = (math.cos(th), math.sin(th))
        if self.start_side == "left":
            sw = (-math.sin(th), math.cos(th))   # left of d
        else:
            sw = (math.sin(th), -math.cos(th))   # right of d
        return d, sw

    def _update_swept_bounds(self):
        """Grow the swept-region box (row-frame) and append to the driven
        polyline. Both describe known-free ground for collection routing."""
        units = self._row_units()
        if units is None or self.odom_x is None:
            return
        (dx, dy), (sx, sy) = units
        lon = self.odom_x * dx + self.odom_y * dy
        lat = self.odom_x * sx + self.odom_y * sy
        if self._swept_long_min is None:
            self._swept_long_min = self._swept_long_max = lon
            self._swept_lat_min = self._swept_lat_max = lat
        else:
            self._swept_long_min = min(self._swept_long_min, lon)
            self._swept_long_max = max(self._swept_long_max, lon)
            self._swept_lat_min = min(self._swept_lat_min, lat)
            self._swept_lat_max = max(self._swept_lat_max, lat)
        # Decimate: only record when we have moved a sample's worth.
        if not self._swept_path:
            self._swept_path.append((self.odom_x, self.odom_y))
        else:
            lx, ly = self._swept_path[-1]
            if math.hypot(self.odom_x - lx, self.odom_y - ly) >= self.path_sample_spacing:
                self._swept_path.append((self.odom_x, self.odom_y))
                if len(self._swept_path) > self.path_max_points:
                    self._swept_path.pop(0)

    def _nearest_path_index(self, x: float, y: float) -> int:
        """Index of the swept-path point closest to (x, y)."""
        best_i, best_d = 0, float("inf")
        for i, (px, py) in enumerate(self._swept_path):
            d = (px - x) * (px - x) + (py - y) * (py - y)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _drive_to_via_path(self, ex: float, ey: float) -> bool:
        """Drive to (ex,ey) by following the recorded swept polyline, so the
        move never cuts across a tree row. Falls back to a straight drive if no
        path is recorded. Returns True when the final point is reached."""
        path = self._swept_path
        if len(path) < 2:
            return self._drive_to(ex, ey, self.collect_arrive_tol)
        if self._path_goal is None:
            self._path_goal = self._nearest_path_index(ex, ey)
            self._path_idx = self._nearest_path_index(self.odom_x, self.odom_y)
        i_goal = self._path_goal
        if self._path_idx == i_goal:
            # On the goal aisle now; final short hop to the precise entry is
            # collinear with the row, so it crosses nothing.
            return self._drive_to(ex, ey, self.collect_arrive_tol)
        step = 1 if i_goal > self._path_idx else -1
        ox, oy = self.odom_x, self.odom_y
        # Snap progress to the nearest UPCOMING path point (monotonic toward the
        # goal). Robust to corner-cutting: we track the closest point rather than
        # requiring the robot to pass exactly through each waypoint.
        nearest_i = self._path_idx
        nearest_d = math.hypot(path[nearest_i][0] - ox, path[nearest_i][1] - oy)
        probe = self._path_idx
        while probe != i_goal:
            probe += step
            d = math.hypot(path[probe][0] - ox, path[probe][1] - oy)
            if d < nearest_d:
                nearest_d, nearest_i = d, probe
            elif d > nearest_d + self.path_wp_tol:
                break    # we are clearly past the nearest point; stop scanning
        self._path_idx = nearest_i
        if self._path_idx == i_goal:
            return self._drive_to(ex, ey, self.collect_arrive_tol)
        # Steer toward a carrot a small number of samples ahead, clamped to the
        # goal. tol=0 so the low-level driver never stops on a carrot (smooth,
        # no stutter); the carrot is close enough that arc corner-cut is tiny.
        carrot = self._path_idx
        for _ in range(max(1, self.path_lookahead)):
            if carrot == i_goal:
                break
            carrot += step
        self._drive_to(path[carrot][0], path[carrot][1], 0.0)
        return False

    def _sweep_entry(self, nox, noy):
        """Entry point (odom) for sweeping nut at (nox,noy): on the nut's sweep
        line (so the nut ends up sweep_offset on the hug side), a little before
        the nut along the row. CLAMPED to the swept box so the run-in is always
        staged on ground the robot already drove - never out in the unexplored,
        possibly walled, space behind/beside the start."""
        units = self._row_units()
        if units is None:
            return None
        (dx, dy), (sx, sy) = units
        # Robot sweep-line point L: nut minus the sweeper offset.
        lx = nox - self.collect_sweep_offset * sx
        ly = noy - self.collect_sweep_offset * sy
        l_long = lx * dx + ly * dy
        l_lat = lx * sx + ly * sy
        # Stage the approach a bit before the nut, but never before the swept
        # region's near end (that is where indoor walls are). max() also keeps
        # it at/behind the nut's own longitudinal, so the forward sweep passes
        # the nut, except for the degenerate case of a nut behind the near end.
        target_long = l_long - self.collect_approach_back
        if self._swept_long_min is not None:
            target_long = max(target_long, self._swept_long_min)
            # Keep the lateral inside the swept columns too.
            l_lat = max(self._swept_lat_min, min(self._swept_lat_max, l_lat))
        elif self.home_x is not None:
            home_long = self.home_x * dx + self.home_y * dy
            target_long = max(target_long, home_long)         # fallback: not behind home
        # Reconstruct the odom point from (target_long, l_lat). d and the lateral
        # unit are orthonormal, so x = long*dx + lat*sx, y = long*dy + lat*sy.
        ex = target_long * dx + l_lat * sx
        ey = target_long * dy + l_lat * sy
        return (ex, ey)

    def _drive_to(self, tx: float, ty: float, tol: float) -> bool:
        """Go-to-point with the RETURN_HOME control style + AVOID_FRONT safety.
        Returns True when within tol."""
        if self.odom_x is None or self.odom_yaw is None:
            self.stop_robot()
            return False
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            return False
        if front < self.avoid_front_distance:
            self._collect_avoid_count += 1
            self.enter_avoid_front("COLLECT_NUTS", f"front={front:.2f}m during collect")
            return False
        dx = tx - self.odom_x
        dy = ty - self.odom_y
        dist = math.hypot(dx, dy)
        if dist <= tol:
            self.stop_robot()
            return True
        he = self.normalize_angle(math.atan2(dy, dx) - self.odom_yaw)
        if abs(he) > self.return_heading_slowdown:
            self.publish_cmd(0.0, math.copysign(self.return_max_angular, he))
        else:
            lin = min(self.return_linear_speed, max(0.03, 0.45 * dist))
            ang = max(-self.return_max_angular, min(self.return_max_angular, 1.4 * he))
            self.publish_cmd(lin, ang)
        return False

    def _sweep_forward(self, heading: float):
        """Drive forward at sweep speed holding `heading` (front-safe)."""
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            return
        if front < self.avoid_front_distance:
            self._collect_avoid_count += 1
            self.enter_avoid_front("COLLECT_NUTS", f"front={front:.2f}m during sweep")
            return
        he = self.normalize_angle(heading - self.odom_yaw)
        ang = max(-self.max_follow_angular, min(self.max_follow_angular, self.k_heading * he))
        self.publish_cmd(self.forward_speed, ang)

    def collect_nuts(self):
        if self.odom_x is None or self.odom_yaw is None:
            self.stop_robot()
            self.publish_status("COLLECT_NUTS waiting for odometry")
            return
        tf = self._map_to_odom()
        if tf is None:
            self.stop_robot()
            self.publish_status("COLLECT_NUTS waiting for map->odom TF")
            return
        if self._row_units() is None:
            self._go_home("COLLECT_NUTS: no row heading anchor.")
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Is the current target still an uncollected, non-skipped nut?
        valid = False
        if self._collect_target is not None:
            key = (round(self._collect_target[0], 2), round(self._collect_target[1], 2))
            valid = key not in self._collect_skip and any(
                (round(mx, 2), round(my, 2)) == key for (mx, my) in self.uncollected_map)
        if not valid:
            # If the previous target left the list WITHOUT us skipping it, it was
            # collected - real progress, so reset the back-to-back skip counter.
            if self._collect_target is not None:
                pkey = (round(self._collect_target[0], 2), round(self._collect_target[1], 2))
                if pkey not in self._collect_skip:
                    self._collect_consec_skips = 0
            targets = self._uncollected_targets(tf)
            if not targets:
                self._go_home("All reachable nuts collected.")
                return
            targets.sort(key=lambda t: math.hypot(t[1][0] - self.odom_x,
                                                  t[1][1] - self.odom_y))
            self._collect_target = targets[0][0]
            self._collect_phase = "EXIT"
            self._collect_deadline = now + self.collect_visit_timeout
            self._collect_avoid_count = 0
            self._path_idx = None            # recompute route for the new target
            self._path_goal = None

        # Stuck-escape: a blocked nut keeps tripping AVOID_FRONT and getting
        # nowhere. Abandon it after a few trips instead of ramming for the whole
        # timeout - then give up on collection if several go this way in a row.
        if self._collect_avoid_count > self.collect_max_avoid:
            self._skip_current_nut("blocked (avoidance limit)")
            return

        # Give up on a nut we can't service in time.
        if self._collect_deadline is not None and now > self._collect_deadline:
            self._skip_current_nut("visit timeout")
            return

        nox, noy = self._apply_tf(self._collect_target[0], self._collect_target[1], tf)
        (dx, dy), _ = self._row_units()

        if self._collect_phase == "EXIT":
            entry = self._sweep_entry(nox, noy)
            if entry is None:
                self._collect_target = None
                return
            if self._drive_to_via_path(entry[0], entry[1]):
                self._collect_phase = "ALIGN"
                self._collect_deadline = now + self.collect_visit_timeout
                self._collect_avoid_count = 0    # fresh budget per phase
            else:
                self.publish_status(
                    f"COLLECT EXIT -> ({entry[0]:+.2f},{entry[1]:+.2f}) | "
                    f"nut=({nox:+.2f},{noy:+.2f}) | "
                    f"route {self._path_idx}->{self._path_goal}"
                )
            return

        if self._collect_phase == "ALIGN":
            err = self.normalize_angle(self.outbound_yaw - self.odom_yaw)
            if abs(err) < self.align_parallel_tol:
                self.stop_robot()
                self._collect_phase = "SWEEP"
                self._collect_deadline = now + self.collect_visit_timeout
                self._collect_avoid_count = 0    # fresh budget per phase
            else:
                mag = min(self.align_max_angular,
                          max(self.min_align_angular, self.k_align * abs(err)))
                self.publish_cmd(0.0, math.copysign(mag, err))
                self.publish_status(
                    f"COLLECT ALIGN | err={math.degrees(err):+.0f}deg")
            return

        if self._collect_phase == "SWEEP":
            robot_long = self.odom_x * dx + self.odom_y * dy
            nut_long = nox * dx + noy * dy
            # The nut is collected (and drops off the list) when the sweeper
            # passes it; that flips `valid` False next tick and we pick the
            # next one. This is the fallback if we drive past without a pickup.
            if robot_long >= nut_long + self.collect_sweep_through:
                self._skip_current_nut("swept past without pickup")
                return
            self._sweep_forward(self.outbound_yaw)
            self.publish_status(
                f"COLLECT SWEEP | nut=({nox:+.2f},{noy:+.2f}) | "
                f"{(nut_long - robot_long):+.2f}m to abeam"
            )
            return

    def _skip_current_nut(self, reason: str):
        """Abandon the current collection target; give up entirely if too many
        nuts in a row turn out to be unreachable."""
        self.stop_robot()
        if self._collect_target is not None:
            self._collect_skip.add((round(self._collect_target[0], 2),
                                    round(self._collect_target[1], 2)))
            self.get_logger().warn(
                f"COLLECT_NUTS: skipping nut "
                f"({self._collect_target[0]:+.2f},{self._collect_target[1]:+.2f}) "
                f"- {reason}."
            )
        self._collect_target = None
        self._collect_consec_skips += 1
        if self._collect_consec_skips >= self.collect_max_consec_skips:
            self._go_home(
                f"COLLECT_NUTS: {self._collect_consec_skips} nuts unreachable "
                f"in a row; giving up on collection."
            )

    # -----------------------------
    # State machine
    # -----------------------------

    def control_loop(self):
        if not self.started:
            self.stop_robot()
            return
        if self.latest_scan is None:
            self.stop_robot()
            self.publish_status("Waiting for /scan")
            return

        # Lazy heading anchor: set on first odom tick if it wasn't ready at
        # sweep_start. The robot hasn't moved during start_straight_duration,
        # so this is still a valid "parallel to row" snapshot.
        if self.outbound_yaw is None and self.odom_yaw is not None:
            self.outbound_yaw = self.odom_yaw
            self.return_target_yaw = self.normalize_angle(self.odom_yaw + math.pi)
            self.get_logger().info(
                f"Lazy anchor: outbound={math.degrees(self.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(self.return_target_yaw):+.1f}deg"
            )

        if (self.home_x is None or self.home_y is None) and self.odom_x is not None and self.odom_y is not None:
            self.home_x = self.odom_x
            self.home_y = self.odom_y
            self.home_yaw = self.odom_yaw
            self.get_logger().info(
                f"Lazy home saved: x={self.home_x:+.2f}, y={self.home_y:+.2f}, "
                f"yaw={math.degrees(self.home_yaw or 0.0):+.1f}deg"
            )

        # Record where the robot has driven during the sweep - that swept box is
        # known-free ground and bounds where collection may stage its run-in.
        if self.state in self._SWEEP_STATES:
            self._update_swept_bounds()

        if self.state == "FOLLOW_OUT":
            self.follow_side(self.current_side)
            if self.state != "FOLLOW_OUT":
                # follow_side handed off to AVOID_FRONT; let it run instead of
                # overriding it with an end-of-row transition this tick.
                return
            abeam = self.last_tree_abeam_or_behind(self.current_side)
            lost = self.should_end_pass()
            if abeam or lost:
                trigger = "last tree abeam" if abeam else "row lost"
                self.clear_start_x = None
                self.clear_start_y = None
                self.arc_start_yaw = None
                self.arc_last_yaw = None
                self.arc_accumulated_yaw = 0.0
                self.seen_row_this_pass = False
                tgt = (math.degrees(self.return_target_yaw)
                       if self.return_target_yaw is not None else float("nan"))
                self.set_state(
                    "CLEAR_END",
                    f"End of row ({trigger}). Return heading={tgt:.0f}deg. "
                    f"Driving past last tree."
                )
                return

        elif self.state == "CLEAR_END":
            self.clear_straight("ARC_TURN", "CLEAR_END")

        elif self.state == "ARC_TURN":
            self.arc_turn()

        elif self.state == "ALIGN":
            self.align_to_row()

        elif self.state == "FOLLOW_BACK":
            self.follow_side(self.current_side)
            if self.state != "FOLLOW_BACK":
                # follow_side handed off to AVOID_FRONT; let it run instead of
                # overriding it with an end-of-row transition this tick.
                return

            # While sweeping back, the NEXT row (if any) sits on the
            # opposite side of the robot. Count confident fits so the
            # decision at the end of the pass is based on a whole pass of
            # evidence, not one noisy scan.
            opp = self.opposite(self.current_side)
            opp_visible, opp_fit = self.row_visible(opp)
            if opp_visible and abs(opp_fit[1]) <= self.next_row_max_dist:
                self.next_row_hits += 1

            if self.should_end_pass():
                self.rows_completed += 1
                next_row_seen = self.next_row_hits >= self.next_row_min_hits

                if self.max_rows > 0:
                    # FIXED COUNT (selected): proceed to the next row after
                    # each one until exactly max_rows are done, regardless of
                    # perception. next_row_seen is logged but not gating.
                    proceed = self.rows_completed < self.max_rows
                    done_reason = f"reached requested {self.max_rows} rows"
                else:
                    # AUTO: only continue while a next row is actually seen.
                    proceed = next_row_seen
                    done_reason = (f"no next row ({self.next_row_hits} hits "
                                   f"< {self.next_row_min_hits})")

                if proceed:
                    self.clear_start_x = None
                    self.clear_start_y = None
                    seen_str = (f"next row on {opp.upper()} "
                                f"({self.next_row_hits} hits)" if next_row_seen
                                else "next row not yet confirmed by lidar")
                    self.set_state(
                        "CLEAR_NEXT",
                        f"Row {self.rows_completed} done; {seen_str}. "
                        f"Clearing row start."
                    )
                else:
                    self.finish_or_return_home(
                        f"Row {self.rows_completed} done, {done_reason}."
                    )
                return

        elif self.state == "CLEAR_NEXT":
            self.clear_straight("TURN_NEXT", "CLEAR_NEXT")

        elif self.state == "TURN_NEXT":
            self.turn_to_next_row()

        elif self.state == "AVOID_FRONT":
            self.avoid_front_obstacle()

        elif self.state == "COLLECT_NUTS":
            self.collect_nuts()

        elif self.state == "RETURN_HOME":
            self.return_home()

        elif self.state == "DONE":
            self.stop_robot()
            self.started = False
            self.publish_status("Demo complete. Robot stopped.")
            return

        elif self.state == "STOPPED":
            self.stop_robot()
            return

        else:
            self.stop_robot()
            self.publish_status(f"Unknown state: {self.state}")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRowFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()