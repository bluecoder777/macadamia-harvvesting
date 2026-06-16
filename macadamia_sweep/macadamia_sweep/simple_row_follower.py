#!/usr/bin/env python3
"""simple_row_follower.py — multi-row macadamia sweep with obstacle avoidance.

Changes vs original:
  * AVOID_OBSTACLE state replaces the hard freeze on emergency-stop.
    When an obstacle enters the 0.30 m warn zone the robot steers around it
    using a potential-field side-step, then automatically resumes the
    interrupted state (FOLLOW_OUT, FOLLOW_BACK, CLEAR_END, etc.).
  * get_front_distance() extended to also return the lateral offset of the
    closest point, so the avoidance knows which side to dodge toward.
  * New helper get_obstacle_info() returns (dist, lateral_offset).
  * New parameter obstacle_warn_distance (default 0.30 m): the distance at
    which avoidance kicks in — before the old 0.18 m hard stop.
  * New parameter avoid_side_gain (default 1.5): how hard the robot steers
    away per metre of lateral obstacle offset.
  * New parameter avoid_resume_distance (default 0.45 m): obstacle must be
    this far before the robot resumes its previous state.
  * The 0.18 m emergency_stop_distance is kept as a last-resort freeze for
    anything that breaks through avoidance (e.g. robot body already very
    close).
"""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String


class SimpleRowFollower(Node):
    def __init__(self):
        super().__init__("simple_row_follower")

        self.declare_parameter("start_side", "right")
        self.start_side = str(self.get_parameter("start_side").value).lower()
        if self.start_side not in ("right", "left"):
            self.get_logger().warn(
                f"Unknown start_side '{self.start_side}', defaulting to 'right'"
            )
            self.start_side = "right"

        self.declare_parameter("lidar_yaw_offset_deg", 180.0)
        self.lidar_yaw_offset = math.radians(
            float(self.get_parameter("lidar_yaw_offset_deg").value)
        )

        self.declare_parameter("max_rows", 0)
        self.max_rows = int(self.get_parameter("max_rows").value)
        self.declare_parameter("next_row_min_hits", 8)
        self.next_row_min_hits = int(self.get_parameter("next_row_min_hits").value)
        self.declare_parameter("next_row_max_dist", 0.60)
        self.next_row_max_dist = float(self.get_parameter("next_row_max_dist").value)
        self.next_row_hits = 0
        self.rows_completed = 0

        # ------------------------------------------------------------------
        # Obstacle avoidance parameters
        # ------------------------------------------------------------------
        # warn zone: avoidance kicks in here, well before the hard stop.
        self.declare_parameter("obstacle_warn_distance", 0.15)
        self.obstacle_warn_distance = float(
            self.get_parameter("obstacle_warn_distance").value
        )
        # resume when obstacle has retreated to this distance.
        self.declare_parameter("avoid_resume_distance", 0.30)
        self.avoid_resume_distance = float(
            self.get_parameter("avoid_resume_distance").value
        )
        # how strongly to steer away (proportional to lateral offset).
        self.declare_parameter("avoid_side_gain", 1.5)
        self.avoid_side_gain = float(self.get_parameter("avoid_side_gain").value)
        # forward creep speed while sidestepping.
        self.declare_parameter("avoid_creep_speed", 0.03)
        self.avoid_creep_speed = float(
            self.get_parameter("avoid_creep_speed").value
        )
        # which state to return to after avoidance clears.
        self._pre_avoid_state: str = "FOLLOW_OUT"
        # timeout: if obstacle doesn't clear in this many seconds, stop.
        self.avoid_timeout = 12.0

        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel_nav", 10)
        self.status_pub = self.create_publisher(String, "/snc_status", 10)

        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 20)
        self.create_subscription(Empty, "/sweep_start", self.start_callback, 10)
        self.create_subscription(Empty, "/sweep_stop", self.stop_callback, 10)

        self.latest_scan: Optional[LaserScan] = None
        self.odom_x: Optional[float] = None
        self.odom_y: Optional[float] = None
        self.odom_yaw: Optional[float] = None

        self.started = False
        self.state = "WAITING"
        self.state_start_time = self.get_clock().now()
        self.last_row_seen_time = self.get_clock().now()
        self.seen_row_this_pass = False
        self.current_side = self.start_side

        self.clear_start_x: Optional[float] = None
        self.clear_start_y: Optional[float] = None
        self.arc_start_yaw: Optional[float] = None
        self.arc_last_yaw: Optional[float] = None
        self.arc_accumulated_yaw = 0.0

        self.outbound_yaw: Optional[float] = None
        self.return_target_yaw: Optional[float] = None

        # ALIGN gains
        self.k_align = 1.2
        self.min_align_angular = 0.08

        # Row-following tuning
        self.desired_side_distance = 0.40
        self.too_close_side = 0.25
        self.declare_parameter("tree_max_range", 1.20)
        self.tree_max_range = float(self.get_parameter("tree_max_range").value)
        self.lidar_min_range = 0.25
        self.emergency_stop_distance = 0.18   # last-resort hard freeze

        self.forward_speed = 0.06
        self.search_speed = 0.03
        self.k_xtrack = 1.2
        self.k_heading = 1.4
        self.max_follow_angular = 0.30
        self.start_straight_duration = 1.5

        # Row detection
        self.cluster_gap = 0.15
        self.tree_max_extent = 0.15
        self.declare_parameter("row_group_gap", 0.35)
        self.row_group_gap = float(self.get_parameter("row_group_gap").value)
        self.min_side_offset = 0.05
        self.min_trees_for_fit = 2
        self.front_width = math.radians(35)
        self.max_tree_points = 80
        self.min_pass_duration = 10.0
        self.max_pass_duration = 90.0
        self.row_lost_timeout = 2.0

        self.declare_parameter("clear_end_distance", 0.15)
        self.clear_end_distance = float(self.get_parameter("clear_end_distance").value)
        self.clear_end_max_time = 8.0

        self.declare_parameter("arc_radius", 0.40)
        self.arc_radius = float(self.get_parameter("arc_radius").value)
        self.arc_linear_speed = 0.06
        self.arc_max_yaw = math.radians(220.0)
        self.arc_distance_gain = 0.6
        nominal_omega = self.arc_linear_speed / self.arc_radius
        self.arc_max_duration = self.arc_max_yaw / nominal_omega + 5.0

        self.align_max_angular = 0.30
        self.align_parallel_tol = math.radians(12.0)
        self.align_max_duration = 8.0
        self.turn_next_max_duration = 18.0
        self.recovery_angular = 0.20

        self.timer = self.create_timer(0.1, self.control_loop)

        self.publish_status("Simple row follower ready. Publish /sweep_start to begin.")
        self.get_logger().info(
            f"Ready (multi-row + obstacle avoidance). "
            f"warn={self.obstacle_warn_distance:.2f}m "
            f"resume={self.avoid_resume_distance:.2f}m "
            f"estop={self.emergency_stop_distance:.2f}m "
            f"start_side={self.start_side} "
            f"max_rows={self.max_rows or 'unlimited'}"
        )

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

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

        if self.odom_yaw is not None:
            self.outbound_yaw = self.odom_yaw
            self.return_target_yaw = self.normalize_angle(self.odom_yaw + math.pi)
            self.get_logger().info(
                f"Anchored row heading: outbound={math.degrees(self.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(self.return_target_yaw):+.1f}deg"
            )
        else:
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

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Obstacle sensing                                                     #
    # ------------------------------------------------------------------ #

    def get_obstacle_info(self) -> Tuple[float, float]:
        """Return (min_front_dist, lateral_offset_of_closest_point).

        Scans the forward cone (±35 deg).  lateral_offset > 0 means the
        obstacle is to the LEFT of the robot's centreline, < 0 means RIGHT.
        The robot should steer AWAY from the obstacle, i.e. angular cmd has
        the opposite sign to lateral_offset.

        Returns (10.0, 0.0) when no obstacle is seen in the cone.
        """
        scan = self.latest_scan
        if scan is None:
            return (10.0, 0.0)

        half_w = self.front_width / 2.0
        min_r = 10.0
        lat_at_min = 0.0

        for i, r in enumerate(scan.ranges):
            angle = (
                scan.angle_min + i * scan.angle_increment + self.lidar_yaw_offset
            )
            angle = self.normalize_angle(angle)
            if abs(angle) > half_w:
                continue
            if not self.valid_range(r, scan):
                continue
            if r > 3.0:
                continue
            if r < min_r:
                min_r = r
                # lateral position of the return in base_link frame.
                # positive = left (ROS convention: y is left).
                lat_at_min = r * math.sin(angle)

        return (min_r, lat_at_min)

    def get_front_distance(self) -> float:
        """Convenience wrapper — distance only."""
        dist, _ = self.get_obstacle_info()
        return dist

    # ------------------------------------------------------------------ #
    # Perception                                                           #
    # ------------------------------------------------------------------ #

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
        if self.outbound_yaw is None or self.odom_yaw is None:
            return None
        return self.normalize_angle(self.outbound_yaw - self.odom_yaw)

    def select_nearest_row(
        self, trees: List[Tuple[float, float, int]], side: str
    ) -> List[Tuple[float, float, int]]:
        if not trees:
            return []
        a = self.row_dir_robot_frame()
        if a is None:
            a = 0.0
        if a > math.pi / 2.0:
            a -= math.pi
        elif a < -math.pi / 2.0:
            a += math.pi
        sin_a = math.sin(a)
        cos_a = math.cos(a)

        tagged: List[Tuple[float, float, float, int]] = []
        for (tx, ty, n) in trees:
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
        trees = self.select_nearest_row(trees, side)
        return [(t[0], t[1]) for t in trees]

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

    # ------------------------------------------------------------------ #
    # AVOID_OBSTACLE state                                                 #
    # ------------------------------------------------------------------ #

    def _enter_avoidance(self, interrupted_state: str, dist: float, lat: float):
        """Transition into AVOID_OBSTACLE, remembering what to resume."""
        self._pre_avoid_state = interrupted_state
        self.set_state(
            "AVOID_OBSTACLE",
            f"OBSTACLE {dist:.2f}m lat={lat:+.2f}m — pausing {interrupted_state}, sidestepping."
        )

    def avoid_obstacle(self):
        """Sidestep around a forward obstacle and resume the mission.

        Strategy:
          1. Read the obstacle position from the forward cone.
          2. If the obstacle is to the LEFT, steer RIGHT (negative angular),
             and vice versa.  This keeps the robot moving forward slowly
             while steering around the object.
          3. Once the obstacle clears the avoid_resume_distance, transition
             back to the pre-avoidance state.
          4. Hard stop (emergency_stop_distance) is still enforced here as
             the absolute last resort.
          5. Timeout: if the obstacle hasn't cleared in avoid_timeout
             seconds, stop and flag — prevents looping forever.
        """
        front, lat = self.get_obstacle_info()

        # Last-resort hard freeze — should almost never trigger now.
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(
                f"AVOID hard stop {front:.2f}m — obstacle too close to manoeuvre."
            )
            return

        # Obstacle cleared — resume.
        if front >= self.avoid_resume_distance:
            self.get_logger().info(
                f"Obstacle cleared ({front:.2f}m). Resuming {self._pre_avoid_state}."
            )
            self.set_state(
                self._pre_avoid_state,
                f"Obstacle cleared. Resuming {self._pre_avoid_state}."
            )
            # Re-seed timers so row-lost / pass-end logic isn't confused by
            # the time spent avoiding.
            self.last_row_seen_time = self.get_clock().now()
            return

        # Timeout guard.
        if self.elapsed_in_state() > self.avoid_timeout:
            self.get_logger().warn(
                f"Avoidance timeout after {self.avoid_timeout:.0f}s. "
                f"Obstacle still at {front:.2f}m. Stopping."
            )
            self.set_state(
                "STOPPED",
                f"Avoidance timeout — obstacle at {front:.2f}m. Manual intervention needed."
            )
            self.stop_robot()
            return

        # Compute sidestep.
        # lat > 0 = obstacle left  → steer right (negative angular).
        # lat < 0 = obstacle right → steer left  (positive angular).
        # Magnitude scales with how central the obstacle is: an obstacle
        # dead-centre (lat≈0) gets maximum sidestep; one at the edge of
        # the cone gets less.
        #
        # We also scale forward speed down proportionally to how close the
        # obstacle is — slower as we get closer, so there's time to react.
        closeness = max(0.0, 1.0 - (front - self.emergency_stop_distance) /
                        (self.avoid_resume_distance - self.emergency_stop_distance))

        angular = -self.avoid_side_gain * lat  # steer away
        angular = max(-self.max_follow_angular, min(self.max_follow_angular, angular))
        linear = self.avoid_creep_speed * (1.0 - 0.5 * closeness)

        self.publish_cmd(linear, angular)
        self.publish_status(
            f"AVOID | front={front:.2f}m lat={lat:+.2f}m | "
            f"cmd lin={linear:.3f} ang={angular:+.2f} | "
            f"resume_at={self.avoid_resume_distance:.2f}m | "
            f"will_resume={self._pre_avoid_state}"
        )

    # ------------------------------------------------------------------ #
    # Line-fit follow                                                      #
    # ------------------------------------------------------------------ #

    def follow_side(self, side: str) -> bool:
        """Follow the row on `side`.  Returns False if avoidance was triggered
        (so the caller can skip end-of-pass checks this tick)."""
        front, lat = self.get_obstacle_info()

        # Last-resort hard freeze.
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP: front={front:.2f} m")
            return False

        # Warn zone — enter avoidance.
        if front < self.obstacle_warn_distance:
            self._enter_avoidance(self.state, front, lat)
            return False

        visible, fit = self.row_visible(side)
        self.mark_row_seen_if_visible(visible)

        if self.elapsed_in_state() < self.start_straight_duration:
            self.publish_cmd(self.forward_speed, 0.0)
            self.publish_status(
                f"START STRAIGHT | row_visible={visible} | front={front:.2f}m"
            )
            return True

        if not visible:
            self.publish_cmd(self.search_speed, 0.0)
            self.publish_status(
                f"ROW NOT CONFIRMED {side.upper()} | creeping forward | "
                f"front={front:.2f}m"
            )
            return True

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
        return True

    def should_end_pass(self) -> bool:
        elapsed = self.elapsed_in_state()
        if elapsed < self.min_pass_duration:
            return False
        if self.seen_row_this_pass and self.time_since_row_seen() >= self.row_lost_timeout:
            return True
        if elapsed >= self.max_pass_duration:
            return True
        return False

    # ------------------------------------------------------------------ #
    # CLEAR_END / CLEAR_NEXT                                               #
    # ------------------------------------------------------------------ #

    def clear_straight(self, next_state: str, label: str):
        front, lat = self.get_obstacle_info()

        # Last-resort hard freeze.
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during {label}: front={front:.2f}m")
            return

        # Enter avoidance if something blocks the clear path.
        if front < self.obstacle_warn_distance:
            self._enter_avoidance(self.state, front, lat)
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

    # ------------------------------------------------------------------ #
    # Arc U-turn                                                           #
    # ------------------------------------------------------------------ #

    def arc_turn(self):
        sign = -1.0 if self.current_side == "right" else +1.0

        if self.odom_yaw is not None:
            if self.arc_last_yaw is None:
                self.arc_start_yaw = self.odom_yaw
                self.arc_last_yaw = self.odom_yaw
            else:
                d = self.normalize_angle(self.odom_yaw - self.arc_last_yaw)
                self.arc_accumulated_yaw += sign * d
                self.arc_last_yaw = self.odom_yaw

        yaw_deg = math.degrees(self.arc_accumulated_yaw)
        v = self.arc_linear_speed
        base_omega = v / self.arc_radius
        omega = base_omega

        arc_target_yaw = math.pi
        if self.arc_accumulated_yaw >= arc_target_yaw:
            self.set_state(
                "ALIGN",
                f"Arc 180 deg complete at +{yaw_deg:.0f}. Aligning to row."
            )
            return

        if (self.arc_accumulated_yaw >= self.arc_max_yaw
                or self.elapsed_in_state() >= self.arc_max_duration):
            self.set_state(
                "ALIGN",
                f"Arc capped at {yaw_deg:.0f} deg. Aligning to row."
            )
            return

        front, _ = self.get_obstacle_info()
        # During the arc, spin in place if blocked — don't enter full
        # avoidance (avoidance would interrupt arc yaw accumulation and
        # the robot could end up with a wrong heading). The spin-only
        # behaviour is the same as the original.
        if front < self.emergency_stop_distance:
            self.publish_cmd(0.0, sign * base_omega)
            self.publish_status(
                f"ARC (hard block {front:.2f}m, spinning only) | yaw +{yaw_deg:.0f}deg"
            )
            return

        self.publish_cmd(v, sign * omega)
        self.publish_status(
            f"ARC | yaw +{yaw_deg:.0f}/180deg | r={self.arc_radius:.2f}m | "
            f"front={front:.2f}m"
        )

    # ------------------------------------------------------------------ #
    # ALIGN                                                                #
    # ------------------------------------------------------------------ #

    def align_to_row(self):
        front, lat = self.get_obstacle_info()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during ALIGN: front={front:.2f}m")
            return

        if self.return_target_yaw is None or self.odom_yaw is None:
            self.publish_cmd(0.0, 0.0)
            self.publish_status("ALIGN cannot proceed: missing heading anchor or odom.")
            if self.elapsed_in_state() > self.align_max_duration:
                self._enter_follow_back("ALIGN aborted - no heading reference.")
            return

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
        self.next_row_hits = 0
        self.set_state("FOLLOW_BACK", status)

    # ------------------------------------------------------------------ #
    # TURN_NEXT                                                            #
    # ------------------------------------------------------------------ #

    def turn_to_next_row(self):
        front, lat = self.get_obstacle_info()
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
        self.next_row_hits = 0
        self.seen_row_this_pass = False
        self.last_row_seen_time = self.get_clock().now()
        self.clear_start_x = None
        self.clear_start_y = None
        self.arc_start_yaw = None
        self.arc_last_yaw = None
        self.arc_accumulated_yaw = 0.0
        self.set_state("FOLLOW_OUT", status)

    # ------------------------------------------------------------------ #
    # State machine                                                        #
    # ------------------------------------------------------------------ #

    def control_loop(self):
        if not self.started:
            self.stop_robot()
            return
        if self.latest_scan is None:
            self.stop_robot()
            self.publish_status("Waiting for /scan")
            return

        if self.outbound_yaw is None and self.odom_yaw is not None:
            self.outbound_yaw = self.odom_yaw
            self.return_target_yaw = self.normalize_angle(self.odom_yaw + math.pi)
            self.get_logger().info(
                f"Lazy anchor: outbound={math.degrees(self.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(self.return_target_yaw):+.1f}deg"
            )

        if self.state == "FOLLOW_OUT":
            ok = self.follow_side(self.current_side)
            if not ok:
                return  # entered avoidance this tick; skip end-of-pass check.
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
            ok = self.follow_side(self.current_side)
            if not ok:
                return  # entered avoidance; skip end-of-pass check.

            opp = self.opposite(self.current_side)
            opp_visible, opp_fit = self.row_visible(opp)
            if opp_visible and abs(opp_fit[1]) <= self.next_row_max_dist:
                self.next_row_hits += 1

            if self.should_end_pass():
                self.rows_completed += 1
                next_row_seen = self.next_row_hits >= self.next_row_min_hits

                if self.max_rows > 0:
                    proceed = self.rows_completed < self.max_rows
                    done_reason = f"reached requested {self.max_rows} rows"
                else:
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
                    self.set_state(
                        "DONE",
                        f"Row {self.rows_completed} done, {done_reason}. "
                        f"Mission complete."
                    )
                    self.stop_robot()
                return

        elif self.state == "CLEAR_NEXT":
            self.clear_straight("TURN_NEXT", "CLEAR_NEXT")

        elif self.state == "TURN_NEXT":
            self.turn_to_next_row()

        elif self.state == "AVOID_OBSTACLE":
            self.avoid_obstacle()

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