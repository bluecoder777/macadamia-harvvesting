#!/usr/bin/env python3
"""Comprehensive diagnostic logger for the macadamia sweep follower.

This is a STANDALONE node that runs alongside simple_row_follower (or any
controller publishing on /cmd_vel_nav). It subscribes to every relevant
topic, captures ground truth for the trees once at startup, and writes a
single CSV plus a human-readable trace so a bad run can be analysed offline
without standing next to the robot.

What it captures, per tick (~10 Hz):
  * Time since logging started.
  * Robot pose in odom: x, y, yaw (deg).
  * Lidar summary: front-min range, left-min, right-min, front-avg,
    left-avg, right-avg, plus the angle of the closest return in each cone.
  * The controller's last commanded velocity (linear.x, angular.z) -
    captured from /cmd_vel_nav so we see what the follower actually decided.
  * The controller's last status string (from /snc_status) - tells us
    which behavioural branch is firing.
  * Distance from the robot to EACH ground-truth tree, and which tree is
    currently nearest. Lets us see "the robot is orbiting tree #2" in the
    data plainly.

Ground truth:
  * On the first scan after start, the logger snapshots every distinct
    near-range cluster (the trees, in the robot's start frame) and stores
    them as (x, y) tree positions in the odom frame.
  * Optional override: pass tree positions on the command line (see usage).

Run:
    ros2 run macadamia_sweep sweep_logger
    # or, to also overlay your measured tree positions:
    ros2 run macadamia_sweep sweep_logger --ros-args \
        -p tree_positions:="[0.5,-0.5, 1.0,-0.5, 1.5,-0.5, 2.0,-0.5]"

Output files (written to ~/sweep_logs/<timestamp>/):
    sweep.csv        one row per tick, machine-readable
    sweep.txt        human-readable trace, easy to skim
    trees.txt        ground-truth tree positions (auto-detected or supplied)

Stop logging cleanly with Ctrl-C - the files are flushed every tick so a
hard kill is still safe.
"""

import csv
import math
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class SweepLogger(Node):
    def __init__(self):
        super().__init__("sweep_logger")

        # ---- Parameters ----
        # Optional measured tree positions (m), as a flat list [x0,y0,x1,y1,...]
        # in the robot's start frame (odom at the moment logging starts).
        # If empty, the logger auto-detects trees from the first scan.
        self.declare_parameter("tree_positions", [])
        # Where to write logs.
        self.declare_parameter("log_dir", os.path.expanduser("~/sweep_logs"))
        # Lidar gates - match the follower so the perception view is the same.
        self.declare_parameter("lidar_min_range", 0.25)
        self.declare_parameter("side_max_range", 1.50)
        # Auto-tree detection thresholds.
        self.declare_parameter("tree_cluster_max_extent", 0.20)  # m
        self.declare_parameter("tree_cluster_gap", 0.15)  # m, point spacing
        # 2.0 m (was 3.0): a 360 deg x 3 m window indoors swallows walls,
        # chair legs and the robot's own chassis -> 18-19 "trees" from 4
        # noodles. Tighten the window to the actual rig.
        self.declare_parameter("tree_max_range", 2.0)  # m
        # A real noodle returns a dense arc of points; 1-2 stray returns
        # are noise. But density falls with range (a noodle spans ~12 points
        # at 1 m, ~2-3 at 3 m), so this is a FLOOR - the effective minimum is
        # scaled up at close range from the expected point count, so distant
        # real trees still pass. See _min_points_for_range.
        self.declare_parameter("tree_min_points", 2)
        self.declare_parameter("tree_radius", 0.06)         # m, noodle radius
        self.declare_parameter("tree_min_points_fraction", 0.35)
        # Minimum separation between accepted trees (merge anything closer;
        # noodles are ~0.7 m apart, so 0.3 m is safe).
        self.declare_parameter("tree_min_separation", 0.30)  # m

        self.tree_positions_param = list(
            self.get_parameter("tree_positions").value or []
        )
        self.log_dir_root = self.get_parameter("log_dir").value
        self.lidar_min_range = float(self.get_parameter("lidar_min_range").value)
        self.side_max_range = float(self.get_parameter("side_max_range").value)
        self.tree_cluster_max_extent = float(
            self.get_parameter("tree_cluster_max_extent").value
        )
        self.tree_cluster_gap = float(self.get_parameter("tree_cluster_gap").value)
        self.tree_max_range = float(self.get_parameter("tree_max_range").value)
        self.tree_min_points = int(self.get_parameter("tree_min_points").value)
        self.tree_radius = float(self.get_parameter("tree_radius").value)
        self.tree_min_points_fraction = float(
            self.get_parameter("tree_min_points_fraction").value
        )
        self.tree_min_separation = float(
            self.get_parameter("tree_min_separation").value
        )

        # ---- State ----
        self.start_time = self.get_clock().now()
        self.latest_scan: Optional[LaserScan] = None
        self.latest_cmd: Optional[TwistStamped] = None
        self.latest_status: str = ""
        self.odom_x: Optional[float] = None
        self.odom_y: Optional[float] = None
        self.odom_yaw: Optional[float] = None
        # Snapshot of odom at first scan - defines the "start frame" we
        # express ground-truth tree positions in.
        self.start_odom_x: Optional[float] = None
        self.start_odom_y: Optional[float] = None
        self.start_odom_yaw: Optional[float] = None
        # Ground-truth tree positions, in the ODOM frame.
        self.trees: List[Tuple[float, float]] = []
        self.trees_locked = False

        # ---- Subscribers ----
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 20)
        self.create_subscription(TwistStamped, "/cmd_vel_nav", self.cmd_callback, 10)
        self.create_subscription(String, "/snc_status", self.status_callback, 10)

        # ---- Set up log files ----
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.log_dir_root, stamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "sweep.csv")
        self.txt_path = os.path.join(self.log_dir, "sweep.txt")
        self.trees_path = os.path.join(self.log_dir, "trees.txt")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "t_s",
                "odom_x", "odom_y", "yaw_deg",
                "front_min", "front_min_angle_deg",
                "left_min", "left_min_angle_deg",
                "right_min", "right_min_angle_deg",
                "front_avg", "left_avg", "right_avg",
                "n_left", "n_right",
                "cmd_lin", "cmd_ang",
                "nearest_tree_idx", "nearest_tree_dist",
                "status",
            ]
        )
        self.csv_file.flush()

        self.txt_file = open(self.txt_path, "w")
        self.txt_file.write(
            f"sweep_logger started {stamp}\n"
            f"log_dir: {self.log_dir}\n"
            f"lidar_min_range={self.lidar_min_range}m, "
            f"side_max_range={self.side_max_range}m\n"
            f"---\n"
        )
        self.txt_file.flush()

        # ---- Apply measured tree positions if given ----
        if len(self.tree_positions_param) >= 2:
            self._apply_measured_trees()

        # ---- Periodic loop ----
        self.timer = self.create_timer(0.1, self.log_tick)

        self.get_logger().info(f"sweep_logger writing to {self.log_dir}")

    # ---- Callbacks ----

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        if not self.trees_locked and self.odom_x is not None:
            self._lock_ground_truth(msg)

    def odom_callback(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny, cosy)

    def cmd_callback(self, msg: TwistStamped):
        self.latest_cmd = msg

    def status_callback(self, msg: String):
        self.latest_status = msg.data

    # ---- Ground truth (auto-detect or supplied) ----

    def _apply_measured_trees(self):
        """Convert the supplied [x0,y0,x1,y1,...] (in robot start frame) to
        ODOM-frame positions, using the odom snapshot at startup. If odom
        isn't ready yet, the locking happens on the first scan instead.
        """
        if self.odom_x is None:
            return
        if self.start_odom_x is None:
            self.start_odom_x = self.odom_x
            self.start_odom_y = self.odom_y
            self.start_odom_yaw = self.odom_yaw
        pairs = self.tree_positions_param
        cx = math.cos(self.start_odom_yaw)
        sy = math.sin(self.start_odom_yaw)
        self.trees = []
        for i in range(0, len(pairs) - 1, 2):
            tx, ty = pairs[i], pairs[i + 1]
            # Start-frame -> odom-frame.
            ox = self.start_odom_x + cx * tx - sy * ty
            oy = self.start_odom_y + sy * tx + cx * ty
            self.trees.append((ox, oy))
        self.trees_locked = True
        self._write_trees_file(source="measured (user-supplied)")

    def _min_points_for_range(self, scan: LaserScan, r: float) -> int:
        """Range-adaptive minimum cluster size.

        A noodle (radius tree_radius) at range r subtends about
        2*atan(tree_radius / r) rad = that / angle_increment lidar returns.
        Require tree_min_points_fraction of that, floored at tree_min_points.
        This keeps the bar high enough to reject noise up close yet low
        enough that genuinely distant trees (which return few points) survive.
        """
        if scan.angle_increment <= 0.0:
            return self.tree_min_points
        subtended = 2.0 * math.atan2(self.tree_radius, max(r, 0.15))
        expected = subtended / scan.angle_increment
        return max(self.tree_min_points,
                   int(math.ceil(self.tree_min_points_fraction * expected)))

    def _lock_ground_truth(self, scan: LaserScan):
        """Auto-detect tree positions from the first scan after start.

        Clusters near-range returns (using the same near-range floor as the
        controller, so phantoms on the robot body are excluded), keeps only
        small clusters consistent with a noodle's size, and converts each
        cluster centroid from base_link to the odom frame using the current
        pose.
        """
        # Take the snapshot of the current pose as the "start frame".
        self.start_odom_x = self.odom_x
        self.start_odom_y = self.odom_y
        self.start_odom_yaw = self.odom_yaw

        # Convert scan to (angle, range) points in base_link, gated.
        pts: List[Tuple[float, float]] = []  # (angle_rad, range)
        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if not (scan.range_min <= r <= scan.range_max):
                continue
            if r < self.lidar_min_range or r > self.tree_max_range:
                continue
            ang = scan.angle_min + i * scan.angle_increment
            pts.append((ang, r))

        # Convert to Cartesian (in base_link).
        xy = [(r * math.cos(a), r * math.sin(a)) for (a, r) in pts]

        # Sort by angle order is already preserved; cluster by Cartesian
        # neighbour distance.
        clusters: List[List[Tuple[float, float]]] = []
        current: List[Tuple[float, float]] = []
        last_pt = None
        for p in xy:
            if last_pt is None:
                current = [p]
            else:
                dx = p[0] - last_pt[0]
                dy = p[1] - last_pt[1]
                if math.hypot(dx, dy) <= self.tree_cluster_gap:
                    current.append(p)
                else:
                    if len(current) >= 1:
                        clusters.append(current)
                    current = [p]
            last_pt = p
        if current:
            clusters.append(current)

        # Filter clusters: enough points (range-adaptive), small
        # (noodle-sized), within range.
        trees_base: List[Tuple[float, float]] = []
        for c in clusters:
            xs = [p[0] for p in c]
            ys = [p[1] for p in c]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            cluster_range = math.hypot(mx, my)
            if len(c) < self._min_points_for_range(scan, cluster_range):
                continue  # too few points for a real tree at this range
            extent = max(max(xs) - min(xs), max(ys) - min(ys))
            if extent > self.tree_cluster_max_extent:
                continue
            trees_base.append((mx, my))

        # Merge detections closer than the minimum tree separation (a
        # noodle partially split by occlusion produces 2+ clusters that are
        # really one tree).
        merged: List[Tuple[float, float]] = []
        for (tx, ty) in trees_base:
            dup = False
            for j, (mx, my) in enumerate(merged):
                if math.hypot(tx - mx, ty - my) < self.tree_min_separation:
                    merged[j] = ((mx + tx) / 2.0, (my + ty) / 2.0)
                    dup = True
                    break
            if not dup:
                merged.append((tx, ty))
        trees_base = merged

        # Convert to odom frame using the snapshot pose.
        cx = math.cos(self.start_odom_yaw)
        sy = math.sin(self.start_odom_yaw)
        self.trees = []
        for (tx, ty) in trees_base:
            ox = self.start_odom_x + cx * tx - sy * ty
            oy = self.start_odom_y + sy * tx + cx * ty
            self.trees.append((ox, oy))

        self.trees_locked = True
        self._write_trees_file(source="auto-detected from first scan")
        self.get_logger().info(
            f"Locked {len(self.trees)} ground-truth tree positions"
        )

    def _write_trees_file(self, source: str):
        with open(self.trees_path, "w") as f:
            f.write(f"# Ground-truth tree positions ({source})\n")
            f.write(
                f"# start pose (odom): x={self.start_odom_x:.3f} "
                f"y={self.start_odom_y:.3f} "
                f"yaw={math.degrees(self.start_odom_yaw):.1f}deg\n"
            )
            f.write("# columns: idx, x_odom, y_odom, x_start_frame, y_start_frame\n")
            cx = math.cos(-self.start_odom_yaw)
            sy = math.sin(-self.start_odom_yaw)
            for i, (tx, ty) in enumerate(self.trees):
                dx = tx - self.start_odom_x
                dy = ty - self.start_odom_y
                # odom -> start frame.
                sx = cx * dx - sy * dy
                sy_ = sy * dx + cx * dy
                f.write(f"{i}, {tx:.3f}, {ty:.3f}, {sx:.3f}, {sy_:.3f}\n")
        self.txt_file.write(f"Locked {len(self.trees)} trees ({source})\n")
        for i, (tx, ty) in enumerate(self.trees):
            self.txt_file.write(
                f"  tree[{i}] odom=({tx:+.2f},{ty:+.2f})\n"
            )
        self.txt_file.write("---\n")
        self.txt_file.flush()

    # ---- Lidar summary ----

    @staticmethod
    def _normalize_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _cone_stats(self, center: float, width: float, max_range: float):
        """Return (min_range, min_angle_deg, mean_range, count) for valid
        returns in the cone. min_angle_deg is the angle at which the closest
        return was found (deg in robot frame).
        """
        scan = self.latest_scan
        if scan is None:
            return (float("inf"), float("nan"), float("inf"), 0)
        min_r = float("inf")
        min_a = float("nan")
        ssum = 0.0
        cnt = 0
        n = len(scan.ranges)
        for i in range(n):
            r = scan.ranges[i]
            if math.isnan(r) or math.isinf(r):
                continue
            if not (scan.range_min <= r <= scan.range_max):
                continue
            if r < self.lidar_min_range or r > max_range:
                continue
            a = scan.angle_min + i * scan.angle_increment
            if abs(self._normalize_angle(a - center)) <= width / 2.0:
                if r < min_r:
                    min_r = r
                    min_a = math.degrees(a)
                ssum += r
                cnt += 1
        mean = (ssum / cnt) if cnt > 0 else float("inf")
        return (min_r, min_a, mean, cnt)

    def _nearest_tree(self) -> Tuple[int, float]:
        if not self.trees or self.odom_x is None:
            return (-1, float("inf"))
        best_i = -1
        best_d = float("inf")
        for i, (tx, ty) in enumerate(self.trees):
            d = math.hypot(tx - self.odom_x, ty - self.odom_y)
            if d < best_d:
                best_d = d
                best_i = i
        return (best_i, best_d)

    # ---- Tick ----

    def log_tick(self):
        if self.latest_scan is None or self.odom_x is None:
            return

        # If trees haven't been locked yet, try once now (handles measured
        # trees supplied before odom arrived).
        if not self.trees_locked and len(self.tree_positions_param) >= 2:
            self._apply_measured_trees()

        t_s = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        front_min, front_min_a, front_avg, _ = self._cone_stats(
            0.0, math.radians(35), 3.0
        )
        # Side cones span +/- 60 deg about +/-90, gated to side_max_range.
        left_min, left_min_a, left_avg, n_left = self._cone_stats(
            math.radians(90), math.radians(120), self.side_max_range
        )
        right_min, right_min_a, right_avg, n_right = self._cone_stats(
            math.radians(-90), math.radians(120), self.side_max_range
        )

        cmd_lin = 0.0
        cmd_ang = 0.0
        if self.latest_cmd is not None:
            cmd_lin = self.latest_cmd.twist.linear.x
            cmd_ang = self.latest_cmd.twist.angular.z

        near_i, near_d = self._nearest_tree()

        self.csv_writer.writerow(
            [
                f"{t_s:.2f}",
                f"{self.odom_x:.3f}", f"{self.odom_y:.3f}",
                f"{math.degrees(self.odom_yaw):.1f}",
                self._fmt(front_min), self._fmt(front_min_a),
                self._fmt(left_min), self._fmt(left_min_a),
                self._fmt(right_min), self._fmt(right_min_a),
                self._fmt(front_avg), self._fmt(left_avg), self._fmt(right_avg),
                n_left, n_right,
                f"{cmd_lin:.3f}", f"{cmd_ang:.3f}",
                near_i, self._fmt(near_d),
                self.latest_status.replace(",", ";"),
            ]
        )
        self.csv_file.flush()

        # Human trace - one line, easy to skim.
        self.txt_file.write(
            f"t={t_s:6.2f}s | "
            f"pose=({self.odom_x:+.2f},{self.odom_y:+.2f},{math.degrees(self.odom_yaw):+5.0f}deg) | "
            f"F={self._fmt(front_min):>5}m@{self._fmt(front_min_a):>5}deg | "
            f"L={self._fmt(left_min):>5}m(n={n_left}) avg={self._fmt(left_avg):>5} | "
            f"R={self._fmt(right_min):>5}m(n={n_right}) avg={self._fmt(right_avg):>5} | "
            f"cmd=(lin={cmd_lin:+.2f},ang={cmd_ang:+.2f}) | "
            f"near_tree={near_i}({self._fmt(near_d)}m) | "
            f"{self.latest_status}\n"
        )
        self.txt_file.flush()

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return "inf"
            return f"{v:.2f}"
        return v

    def destroy_node(self):
        try:
            self.csv_file.close()
            self.txt_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SweepLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Logs written to {node.log_dir}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()