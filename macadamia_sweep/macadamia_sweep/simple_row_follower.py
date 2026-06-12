#!/usr/bin/env python3

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
        self.tree_max_range = 0.80
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
        self.min_trees_for_fit = 2
        self.front_width = math.radians(35)
        self.max_tree_points = 80

        self.min_pass_duration = 10.0
        self.max_pass_duration = 90.0
        self.row_lost_timeout = 2.0

        # CLEAR_END
        self.clear_end_distance = 0.20
        self.clear_end_max_time = 8.0

        # Arc U-turn
        self.arc_radius = 0.55
        self.arc_linear_speed = 0.06
        self.arc_max_yaw = math.radians(220.0)
        self.arc_distance_gain = 0.6

        nominal_omega = self.arc_linear_speed / self.arc_radius
        self.arc_max_duration = self.arc_max_yaw / nominal_omega + 5.0

        # ALIGN
        self.align_max_angular = 0.30
        self.align_parallel_tol = math.radians(12.0)
        self.align_max_duration = 8.0

        self.recovery_angular = 0.20

        self.timer = self.create_timer(0.1, self.control_loop)

        self.publish_status("Simple row follower ready. Publish /sweep_start to begin.")
        self.get_logger().info(
            f"Ready. CLEAR_END={self.clear_end_distance:.2f}m, "
            f"ARC r={self.arc_radius:.2f}m v={self.arc_linear_speed:.2f}m/s, "
            f"start_side={self.start_side}, "
            f"lidar_yaw_offset={math.degrees(self.lidar_yaw_offset):+.0f}deg"
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
        self.started = True
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

    def fit_row_line(self, side: str) -> Optional[Tuple[float, float, int]]:
        pts = self.collect_row_side_points(side)
        trees = self.cluster_to_trees(pts)
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

    def clear_end(self):
        front = self.get_front_distance()
        if front < self.emergency_stop_distance:
            self.stop_robot()
            self.publish_status(f"EMERGENCY STOP during CLEAR_END: front={front:.2f}m")
            return

        if self.clear_start_x is None and self.odom_x is not None:
            self.clear_start_x = self.odom_x
            self.clear_start_y = self.odom_y

        if self.clear_start_x is None or self.odom_x is None:
            self.publish_cmd(self.forward_speed, 0.0)
            self.publish_status("CLEAR_END (no odom yet)")
            return

        dx = self.odom_x - self.clear_start_x
        dy = self.odom_y - self.clear_start_y
        traveled = math.hypot(dx, dy)

        if traveled >= self.clear_end_distance:
            self.set_state(
                "ARC_TURN",
                f"Cleared {traveled:.2f}m past last tree. Starting arc."
            )
            return

        if self.elapsed_in_state() > self.clear_end_max_time:
            self.set_state(
                "ARC_TURN",
                "CLEAR_END time cap, starting arc anyway."
            )
            return

        self.publish_cmd(self.forward_speed, 0.0)
        self.publish_status(
            f"CLEAR_END {traveled:.2f}/{self.clear_end_distance:.2f}m | front={front:.2f}m"
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
                d = abs(self.normalize_angle(self.odom_yaw - self.arc_last_yaw))
                self.arc_accumulated_yaw += d
                self.arc_last_yaw = self.odom_yaw

        yaw_deg = math.degrees(self.arc_accumulated_yaw)

        v = self.arc_linear_speed
        base_omega = v / self.arc_radius

        fit = self.fit_row_line(self.current_side)
        if fit is not None:
            _, perp, n_pts = fit
            inside_dist = abs(perp)
            dist_error = inside_dist - self.arc_radius
            omega = base_omega + self.arc_distance_gain * dist_error
        else:
            inside_dist = float("nan")
            n_pts = 0
            omega = base_omega
        omega = max(0.3 * base_omega, min(2.0 * base_omega, omega))

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
        dist_str = f"{inside_dist:.2f}m" if not math.isnan(inside_dist) else "nan"
        self.publish_status(
            f"ARC | yaw +{yaw_deg:.0f}/180deg | "
            f"inside={dist_str} n={n_pts} | front={front:.2f}m"
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
        self.set_state("FOLLOW_BACK", status)

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

        if self.state == "FOLLOW_OUT":
            self.follow_side(self.current_side)
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
            self.clear_end()

        elif self.state == "ARC_TURN":
            self.arc_turn()

        elif self.state == "ALIGN":
            self.align_to_row()

        elif self.state == "FOLLOW_BACK":
            self.follow_side(self.current_side)
            if self.should_end_pass():
                self.set_state("DONE", "Return row end detected. Demo done.")
                self.stop_robot()
                return

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