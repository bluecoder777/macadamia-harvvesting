#!/usr/bin/env python3
"""harvest_dashboard.py — standalone ROS2 terminal dashboard for macadamia sweep.

Completely independent of simple_row_follower.py, sweep_logger.py, and test.py.
Does NOT import or modify any existing file.  Drop it in the same package folder
and add one entry_point line to setup.py (see bottom of this file).

What it subscribes to (read-only):
    /scan               sensor_msgs/LaserScan
    /odometry/filtered  nav_msgs/Odometry
    /cmd_vel_nav        geometry_msgs/TwistStamped
    /snc_status         std_msgs/String

What it does NOT do:
    - publish anything
    - write files
    - import from other nodes in this package

Metrics computed entirely from the live topic stream:
    1. Tree detection success rate
       "Each scan tick where row_visible() succeeds / all follow ticks"
       Uses the same cluster → line-fit logic as the follower (independent copy).

    2. Average approach distance to tree
       Each time the robot is the nearest it has been to any detected tree
       (distance is decreasing and then increases), record the closest approach.
       Average over all such events during the run.

    3. Coverage completion rate
       "What fraction of the orchard has the robot swept past?"
       Estimated from the robot's forward travel along the row compared to the
       max x-extent of detected trees.

    4. Navigation success rate
       Ticks where state is FOLLOW_OUT or FOLLOW_BACK and the row line fit
       succeeds, divided by all FOLLOW ticks.

    5. Obstacle avoidance success rate
       Counts AVOID entries (from /snc_status) and tracks how many resolved
       cleanly (state went back to FOLLOW) vs hit the hard stop / timeout.

    6. Traversal completion time
       Wall-clock seconds from first FOLLOW_OUT tick to DONE status.

    7. Nut collection map
       A 2-D occupancy sketch in the terminal.  Each detected tree is a
       candidate nut location.  A tree counts as "collected" if the robot
       passed within approach_threshold of it during a FOLLOW pass.
       Otherwise it counts as "missed".

Display: pure ANSI terminal (no curses, no GUI).  Refreshes every 2 s.
Works alongside sweep_logger — they subscribe to the same topics independently.

To add to the package:

1. Copy this file to:
       macadamia_sweep/macadamia_sweep/harvest_dashboard.py

2. Add ONE line to setup.py entry_points['console_scripts']:
       'harvest_dashboard = macadamia_sweep.harvest_dashboard:main',

3. Rebuild:
       colcon build --packages-select macadamia_sweep --symlink-install
       source install/setup.bash

4. Run (in a separate terminal alongside the follower):
       ros2 run macadamia_sweep harvest_dashboard

Optional ROS2 parameters:
    approach_threshold   float   0.55   m — distance to count a tree as "visited"
    display_rate         float   2.0    Hz — how often to redraw the terminal
    nut_yield_per_tree   int     4      simulated nuts per tree (for nut map)
"""

import math
import os
import re
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# ──────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ──────────────────────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_BLUE   = "\033[34m"
_MAGENTA = "\033[35m"
_WHITE  = "\033[37m"
_BG_DARK = "\033[40m"
_CLEAR_SCREEN = "\033[2J\033[H"


def _bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Return a coloured ASCII progress bar string."""
    pct = max(0.0, min(1.0, value))
    filled = round(pct * width)
    colour = _GREEN if pct >= 0.75 else (_YELLOW if pct >= 0.40 else _RED)
    return f"{colour}{fill * filled}{_DIM}{empty * (width - filled)}{_RESET}"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Inline perception — independent copy so we never import the follower
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _valid_range(r: float, rmin: float, rmax: float) -> bool:
    return (
        not math.isnan(r) and not math.isinf(r)
        and rmin <= r <= rmax
        and r >= 0.25
    )


def _scan_to_side_points(
    scan: LaserScan,
    side: str,
    lidar_yaw_offset: float,
    tree_max_range: float,
) -> List[Tuple[float, float]]:
    """Return (x, y) Cartesian points on the requested side in base_link."""
    if side == "right":
        min_a, max_a = math.radians(-170.0), math.radians(20.0)
    else:
        min_a, max_a = math.radians(-20.0), math.radians(170.0)
    pts = []
    for i, r in enumerate(scan.ranges):
        angle = scan.angle_min + i * scan.angle_increment + lidar_yaw_offset
        angle = _normalize_angle(angle)
        if not (min_a <= angle <= max_a):
            continue
        if not _valid_range(r, scan.range_min, scan.range_max):
            continue
        if r > tree_max_range:
            continue
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    pts.sort(key=lambda p: math.atan2(p[1], p[0]))
    return pts


def _cluster_to_centroids(
    pts: List[Tuple[float, float]],
    cluster_gap: float = 0.15,
    max_extent: float = 0.15,
) -> List[Tuple[float, float]]:
    """Euclidean 1-D clustering → (cx, cy) centroids of small clusters."""
    if not pts:
        return []
    clusters: List[List[Tuple[float, float]]] = [[pts[0]]]
    for prev, p in zip(pts, pts[1:]):
        if math.hypot(p[0] - prev[0], p[1] - prev[1]) <= cluster_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    out = []
    for c in clusters:
        xs = [q[0] for q in c]
        ys = [q[1] for q in c]
        if max(max(xs) - min(xs), max(ys) - min(ys)) <= max_extent:
            out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return out


def _fit_row_line(
    scan: LaserScan,
    side: str,
    lidar_yaw_offset: float,
    tree_max_range: float = 1.20,
) -> Optional[Tuple[float, float, int]]:
    """Return (line_angle_rad, perp_dist_m, n_trees) or None."""
    pts = _scan_to_side_points(scan, side, lidar_yaw_offset, tree_max_range)
    trees = _cluster_to_centroids(pts)
    if len(trees) < 2:
        return None
    cx = [t[0] for t in trees]
    cy = [t[1] for t in trees]
    n = len(trees)
    sx, sy = sum(cx), sum(cy)
    sxx = sum(x * x for x in cx)
    sxy = sum(x * y for x, y in zip(cx, cy))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-6:
        return (math.pi / 2.0, sy / n, n)
    m = (n * sxy - sx * sy) / denom
    c = (sy - m * sx) / n
    return (math.atan(m), c / math.sqrt(1.0 + m * m), n)


def _detect_trees_global(
    scan: LaserScan,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    lidar_yaw_offset: float,
    tree_max_range: float = 2.0,
) -> List[Tuple[float, float]]:
    """Detect trees in any direction and return odom-frame (x, y) positions."""
    pts_base = []
    for i, r in enumerate(scan.ranges):
        angle = scan.angle_min + i * scan.angle_increment + lidar_yaw_offset
        angle = _normalize_angle(angle)
        if not _valid_range(r, scan.range_min, scan.range_max):
            continue
        if r > tree_max_range:
            continue
        pts_base.append((r * math.cos(angle), r * math.sin(angle)))

    centroids_base = _cluster_to_centroids(pts_base, cluster_gap=0.15, max_extent=0.20)

    cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
    odom_trees = []
    for (bx, by) in centroids_base:
        ox = robot_x + cos_y * bx - sin_y * by
        oy = robot_y + sin_y * bx + cos_y * by
        odom_trees.append((ox, oy))
    return odom_trees


def _merge_trees(
    known: List[Tuple[float, float]],
    new: List[Tuple[float, float]],
    min_sep: float = 0.30,
) -> List[Tuple[float, float]]:
    """Merge new detections into the known list, ignoring duplicates."""
    result = list(known)
    for (nx, ny) in new:
        if all(math.hypot(nx - kx, ny - ky) > min_sep for (kx, ky) in result):
            result.append((nx, ny))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Parse helpers for /snc_status strings
# ──────────────────────────────────────────────────────────────────────────────

_STATE_KEYWORDS = [
    "FOLLOW_OUT", "FOLLOW_BACK", "CLEAR_END", "ARC_TURN", "ALIGN",
    "CLEAR_NEXT", "TURN_NEXT", "AVOID_OBSTACLE", "DONE", "STOPPED", "WAITING",
]

def _extract_state(status: str) -> str:
    for kw in _STATE_KEYWORDS:
        if kw in status:
            return kw
    if "FOLLOWING" in status:
        return "FOLLOW_OUT"   # follower doesn't repeat state name every tick
    if "EMERGENCY STOP" in status:
        return "EMERGENCY_STOP"
    if "AVOID" in status:
        return "AVOID_OBSTACLE"
    return "UNKNOWN"


def _extract_perp(status: str) -> Optional[float]:
    m = re.search(r"perp=([+-]?\d+\.\d+)", status)
    return float(m.group(1)) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Nut map renderer
# ──────────────────────────────────────────────────────────────────────────────

def _render_nut_map(
    trees: List[Tuple[float, float]],
    visited: set,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    trail: List[Tuple[float, float]],
    width: int = 60,
    height: int = 16,
) -> List[str]:
    """Return a list of terminal lines showing the top-down nut map."""
    if not trees and not trail:
        return [f"  {_DIM}(no tree detections yet){_RESET}"]

    all_x = [t[0] for t in trees] + [p[0] for p in trail] + [robot_x]
    all_y = [t[1] for t in trees] + [p[1] for p in trail] + [robot_y]
    xmin, xmax = min(all_x) - 0.3, max(all_x) + 0.3
    ymin, ymax = min(all_y) - 0.3, max(all_y) + 0.3
    xrange = max(xmax - xmin, 0.5)
    yrange = max(ymax - ymin, 0.5)

    def to_cell(x, y):
        col = int((x - xmin) / xrange * (width - 1))
        row = int((1.0 - (y - ymin) / yrange) * (height - 1))
        return max(0, min(width - 1, col)), max(0, min(height - 1, row))

    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Trail
    for (px, py) in trail[::3]:
        c, r = to_cell(px, py)
        if grid[r][c] == " ":
            grid[r][c] = f"{_DIM}.{_RESET}"

    # Trees
    for i, (tx, ty) in enumerate(trees):
        c, r = to_cell(tx, ty)
        if i in visited:
            grid[r][c] = f"{_GREEN}T{_RESET}"   # collected
        else:
            grid[r][c] = f"{_RED}X{_RESET}"      # missed

    # Robot
    rc, rr = to_cell(robot_x, robot_y)
    arrow = {
        (d, True): ch
        for d, ch in [
            (0, "→"), (45, "↗"), (90, "↑"), (135, "↖"),
            (180, "←"), (225, "↙"), (270, "↓"), (315, "↘"),
        ]
    }
    yaw_deg = math.degrees(robot_yaw) % 360
    bucket = round(yaw_deg / 45) * 45 % 360
    ch = {0: "→", 45: "↗", 90: "↑", 135: "↖",
          180: "←", 225: "↙", 270: "↓", 315: "↘"}.get(bucket, "●")
    grid[rr][rc] = f"{_CYAN}{_BOLD}{ch}{_RESET}"

    lines = []
    border = f"{_DIM}+" + "-" * width + f"+{_RESET}"
    lines.append(border)
    for row in grid:
        lines.append(f"{_DIM}|{_RESET}" + "".join(row) + f"{_DIM}|{_RESET}")
    lines.append(border)
    lines.append(
        f"  {_GREEN}T{_RESET}=collected  "
        f"{_RED}X{_RESET}=missed  "
        f"{_CYAN}arrow{_RESET}=robot  "
        f"{_DIM}.{_RESET}=trail"
    )
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard node
# ──────────────────────────────────────────────────────────────────────────────

class HarvestDashboard(Node):

    def __init__(self):
        super().__init__("harvest_dashboard")

        # Parameters
        self.declare_parameter("approach_threshold", 0.55)
        self.approach_threshold = float(
            self.get_parameter("approach_threshold").value
        )
        self.declare_parameter("display_rate", 2.0)
        display_rate = float(self.get_parameter("display_rate").value)
        self.declare_parameter("nut_yield_per_tree", 4)
        self.nut_yield = int(self.get_parameter("nut_yield_per_tree").value)

        # Lidar convention — must match the follower default.
        self.lidar_yaw_offset = math.pi   # 180 deg; override with param if needed
        self.declare_parameter("lidar_yaw_offset_deg", 180.0)
        self.lidar_yaw_offset = math.radians(
            float(self.get_parameter("lidar_yaw_offset_deg").value)
        )

        # ── subscriptions (read-only) ──
        self.create_subscription(LaserScan, "/scan", self._cb_scan, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self._cb_odom, 20)
        self.create_subscription(TwistStamped, "/cmd_vel_nav", self._cb_cmd, 10)
        self.create_subscription(String, "/snc_status", self._cb_status, 10)

        # ── runtime state ──
        self._scan: Optional[LaserScan] = None
        self._odom_x: float = 0.0
        self._odom_y: float = 0.0
        self._odom_yaw: float = 0.0
        self._odom_ready: bool = False
        self._cmd_lin: float = 0.0
        self._cmd_ang: float = 0.0
        self._status: str = ""
        self._current_state: str = "WAITING"

        # ── metrics accumulators ──
        # 1. Tree detection
        self._detect_ticks: int = 0       # follow ticks where fit succeeded
        self._follow_ticks: int = 0       # total follow ticks
        self._detect_history: deque = deque(maxlen=200)  # rolling window

        # 2. Approach distance
        self._approach_events: List[float] = []  # closest-approach distances
        self._prev_nearest_dist: float = 9999.0
        self._nearest_dist: float = 9999.0

        # 3. Coverage
        self._max_x_seen: float = 0.0     # max forward extent of trees seen
        self._max_x_reached: float = 0.0  # max x the robot actually reached

        # 4. Navigation success
        self._nav_success_ticks: int = 0
        self._nav_total_ticks: int = 0

        # 5. Obstacle avoidance
        self._avoid_entries: int = 0
        self._avoid_resolved: int = 0    # transitioned back to FOLLOW
        self._avoid_hard_stops: int = 0  # hit emergency stop / timeout
        self._in_avoidance: bool = False
        self._estop_count: int = 0

        # 6. Traversal time
        self._mission_start_time: Optional[float] = None
        self._mission_end_time: Optional[float] = None
        self._run_start_wall: float = time.time()

        # 7. Nut map
        self._trees: List[Tuple[float, float]] = []   # odom-frame detections
        self._visited_tree_ids: set = set()            # indices into _trees
        self._trail: List[Tuple[float, float]] = []   # robot path
        self._per_tree_closest: Dict[int, float] = {} # closest the robot got

        # ── display timer ──
        self.create_timer(1.0 / max(0.1, display_rate), self._draw)

        self.get_logger().info(
            "harvest_dashboard started. Subscribing to /scan /odometry/filtered "
            "/cmd_vel_nav /snc_status"
        )

    # ── callbacks ──────────────────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        self._scan = msg
        if not self._odom_ready:
            return
        # Update global tree map
        new_trees = _detect_trees_global(
            msg,
            self._odom_x, self._odom_y, self._odom_yaw,
            self.lidar_yaw_offset,
        )
        self._trees = _merge_trees(self._trees, new_trees)
        # Update max x extent
        for (tx, ty) in self._trees:
            self._max_x_seen = max(self._max_x_seen, tx)

        # Per-tree closest approach
        for i, (tx, ty) in enumerate(self._trees):
            d = math.hypot(tx - self._odom_x, ty - self._odom_y)
            prev = self._per_tree_closest.get(i, 9999.0)
            self._per_tree_closest[i] = min(prev, d)
            if d <= self.approach_threshold and self._current_state in (
                "FOLLOW_OUT", "FOLLOW_BACK"
            ):
                self._visited_tree_ids.add(i)

        # Nearest-tree approach event detection (local minimum tracking)
        if self._trees:
            nearest_d = min(
                math.hypot(tx - self._odom_x, ty - self._odom_y)
                for (tx, ty) in self._trees
            )
            if nearest_d < self._nearest_dist:
                self._nearest_dist = nearest_d
            elif self._nearest_dist < self._prev_nearest_dist:
                # Just passed a local minimum — record approach distance
                if self._nearest_dist < 2.0:  # ignore far background detections
                    self._approach_events.append(self._nearest_dist)
                self._nearest_dist = nearest_d
            self._prev_nearest_dist = nearest_d

        # Row-line fit for detection rate
        if self._current_state in ("FOLLOW_OUT", "FOLLOW_BACK"):
            self._follow_ticks += 1
            side = "right"  # dashboard doesn't know which side; check both
            fit = _fit_row_line(msg, "right", self.lidar_yaw_offset) or \
                  _fit_row_line(msg, "left",  self.lidar_yaw_offset)
            success = fit is not None
            self._detect_history.append(1 if success else 0)
            if success:
                self._detect_ticks += 1

        # Navigation success (fit available AND state is a follow state)
        if self._current_state in ("FOLLOW_OUT", "FOLLOW_BACK"):
            self._nav_total_ticks += 1
            fit = _fit_row_line(msg, "right", self.lidar_yaw_offset) or \
                  _fit_row_line(msg, "left",  self.lidar_yaw_offset)
            if fit is not None:
                self._nav_success_ticks += 1

    def _cb_odom(self, msg: Odometry):
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_yaw = math.atan2(siny, cosy)
        self._odom_ready = True
        # Coverage: track how far the robot has gone
        self._max_x_reached = max(self._max_x_reached, self._odom_x)
        # Trail
        if not self._trail or math.hypot(
            self._odom_x - self._trail[-1][0],
            self._odom_y - self._trail[-1][1],
        ) > 0.05:
            self._trail.append((self._odom_x, self._odom_y))
            if len(self._trail) > 3000:
                self._trail = self._trail[-3000:]

    def _cb_cmd(self, msg: TwistStamped):
        self._cmd_lin = msg.twist.linear.x
        self._cmd_ang = msg.twist.angular.z

    def _cb_status(self, msg: String):
        status = msg.data
        self._status = status
        prev_state = self._current_state
        self._current_state = _extract_state(status)

        # Mission timing
        if self._current_state in ("FOLLOW_OUT", "FOLLOW_BACK") \
                and self._mission_start_time is None:
            self._mission_start_time = time.time()

        if self._current_state == "DONE" and self._mission_end_time is None:
            self._mission_end_time = time.time()

        # Avoidance tracking
        if self._current_state == "AVOID_OBSTACLE" and not self._in_avoidance:
            self._avoid_entries += 1
            self._in_avoidance = True

        if self._in_avoidance and self._current_state in ("FOLLOW_OUT", "FOLLOW_BACK"):
            self._avoid_resolved += 1
            self._in_avoidance = False

        if "EMERGENCY STOP" in status or "Avoidance timeout" in status:
            self._estop_count += 1
            if self._in_avoidance:
                self._avoid_hard_stops += 1
                self._in_avoidance = False

    # ── metric helpers ──────────────────────────────────────────────────────

    def _detection_rate(self) -> float:
        if self._follow_ticks == 0:
            return 0.0
        return self._detect_ticks / self._follow_ticks

    def _rolling_detection_rate(self) -> float:
        if not self._detect_history:
            return 0.0
        return sum(self._detect_history) / len(self._detect_history)

    def _avg_approach_dist(self) -> Optional[float]:
        if not self._approach_events:
            return None
        return sum(self._approach_events) / len(self._approach_events)

    def _coverage_rate(self) -> float:
        if self._max_x_seen < 0.1:
            return 0.0
        return min(1.0, self._max_x_reached / self._max_x_seen)

    def _nav_success_rate(self) -> float:
        if self._nav_total_ticks == 0:
            return 0.0
        return self._nav_success_ticks / self._nav_total_ticks

    def _avoid_success_rate(self) -> float:
        if self._avoid_entries == 0:
            return 1.0   # no obstacles encountered = 100% (trivially)
        return self._avoid_resolved / self._avoid_entries

    def _elapsed_mission(self) -> Optional[float]:
        if self._mission_start_time is None:
            return None
        end = self._mission_end_time or time.time()
        return end - self._mission_start_time

    def _nut_counts(self) -> Tuple[int, int]:
        """Return (collected_nuts, missed_nuts)."""
        n_trees = len(self._trees)
        collected = len(self._visited_tree_ids)
        missed = n_trees - collected
        return collected * self.nut_yield, missed * self.nut_yield

    # ── rendering ──────────────────────────────────────────────────────────

    def _draw(self):
        lines = []
        W = 70   # terminal width

        def hline(char="─"):
            return _DIM + char * W + _RESET

        def section(title):
            pad = (W - len(title) - 2) // 2
            return (
                _DIM + "─" * pad + _RESET
                + _BOLD + _CYAN + f" {title} " + _RESET
                + _DIM + "─" * (W - pad - len(title) - 2) + _RESET
            )

        def metric(label, value_str, bar_val=None, bar_w=18, note=""):
            b = f" {_bar(bar_val, bar_w)}" if bar_val is not None else ""
            n = f"  {_DIM}{note}{_RESET}" if note else ""
            label_col = f"{_BOLD}{label:<36}{_RESET}"
            return f"  {label_col}{value_str}{b}{n}"

        # Header
        elapsed_wall = time.time() - self._run_start_wall
        state_col = {
            "FOLLOW_OUT": _GREEN, "FOLLOW_BACK": _BLUE,
            "AVOID_OBSTACLE": _YELLOW, "EMERGENCY_STOP": _RED,
            "DONE": _GREEN + _BOLD, "STOPPED": _RED,
        }.get(self._current_state, _WHITE)

        lines.append("")
        lines.append(
            _BOLD + _CYAN
            + "  MACADAMIA HARVEST DASHBOARD".center(W)
            + _RESET
        )
        lines.append(
            _DIM
            + f"  ros uptime {elapsed_wall:.0f}s   "
            + f"state: {state_col}{self._current_state}{_RESET}   "
            + f"pose ({self._odom_x:+.2f}, {self._odom_y:+.2f})  "
            + f"yaw {math.degrees(self._odom_yaw):+.0f}°"
            + _RESET
        )
        lines.append(hline())

        # ── metrics ─────────────────────────────────────────────────────

        lines.append(section("MISSION METRICS"))

        # 1. Tree detection success rate
        dr = self._detection_rate()
        rdr = self._rolling_detection_rate()
        lines.append(metric(
            "1  Tree detection success",
            f"{_pct(dr):>8}",
            bar_val=dr,
            note=f"rolling(200) {_pct(rdr)}"
        ))

        # 2. Average approach distance
        aad = self._avg_approach_dist()
        if aad is not None:
            aad_str = f"{aad:.3f} m"
            aad_bar = max(0.0, 1.0 - aad / 1.0)  # closer = higher bar
        else:
            aad_str = "   —   "
            aad_bar = 0.0
        lines.append(metric(
            "2  Avg approach to tree",
            f"{aad_str:>8}",
            bar_val=aad_bar,
            note=f"{len(self._approach_events)} events"
        ))

        # 3. Coverage completion rate
        cov = self._coverage_rate()
        lines.append(metric(
            "3  Coverage completion",
            f"{_pct(cov):>8}",
            bar_val=cov,
            note=f"robot {self._max_x_reached:.2f}m / field {self._max_x_seen:.2f}m"
        ))

        # 4. Navigation success rate
        nav = self._nav_success_rate()
        lines.append(metric(
            "4  Navigation success",
            f"{_pct(nav):>8}",
            bar_val=nav,
            note=f"{self._nav_success_ticks}/{self._nav_total_ticks} ticks"
        ))

        # 5. Obstacle avoidance success
        avs = self._avoid_success_rate()
        avoid_note = (
            f"entries {self._avoid_entries}  "
            f"resolved {self._avoid_resolved}  "
            f"hard-stops {self._avoid_hard_stops + self._estop_count}"
        )
        if self._avoid_entries == 0:
            avs_str = "  none"
        else:
            avs_str = f"{_pct(avs):>8}"
        lines.append(metric(
            "5  Obstacle avoidance success",
            avs_str,
            bar_val=avs if self._avoid_entries > 0 else None,
            note=avoid_note
        ))

        # 6. Traversal completion time
        et = self._elapsed_mission()
        if et is None:
            et_str = "   not started"
        elif self._mission_end_time:
            m, s = divmod(et, 60)
            et_str = f"  {int(m):02d}:{s:05.2f}  {_GREEN}COMPLETE{_RESET}"
        else:
            m, s = divmod(et, 60)
            et_str = f"  {int(m):02d}:{s:05.2f}  {_YELLOW}running…{_RESET}"
        lines.append(f"  {'6  Traversal time':<36}{et_str}")

        lines.append(hline())

        # ── nut harvest ─────────────────────────────────────────────────

        lines.append(section("NUT HARVEST ESTIMATE"))

        n_trees = len(self._trees)
        collected_nuts, missed_nuts = self._nut_counts()
        visited = len(self._visited_tree_ids)

        nut_header = (
            f"  Trees detected: {_BOLD}{n_trees}{_RESET}   "
            f"Visited: {_GREEN}{_BOLD}{visited}{_RESET}   "
            f"Missed: {_RED}{_BOLD}{n_trees - visited}{_RESET}   "
            f"Yield/tree: {self.nut_yield}"
        )
        lines.append(nut_header)
        lines.append(
            f"  Nuts collected: {_GREEN}{_BOLD}{collected_nuts}{_RESET}   "
            f"Nuts missed:   {_RED}{_BOLD}{missed_nuts}{_RESET}"
        )

        if n_trees > 0:
            harvest_pct = visited / n_trees
            lines.append(f"  Harvest rate  {_pct(harvest_pct):>8}  {_bar(harvest_pct, 28)}")

        # Per-tree closest approach table (compact)
        if self._per_tree_closest:
            lines.append(f"  {_DIM}Tree  closest(m)  status{_RESET}")
            for i, (tx, ty) in enumerate(self._trees[:20]):   # cap at 20
                cd = self._per_tree_closest.get(i, 9999.0)
                tag = f"{_GREEN}collected{_RESET}" if i in self._visited_tree_ids \
                      else f"{_RED}missed   {_RESET}"
                lines.append(
                    f"  {_DIM}#{i:<3}{_RESET}"
                    f"  ({tx:+.2f},{ty:+.2f})  "
                    f"  {cd:.3f} m  {tag}"
                )
            if len(self._trees) > 20:
                lines.append(f"  {_DIM}… {len(self._trees) - 20} more trees{_RESET}")

        lines.append(hline())

        # ── nut map ─────────────────────────────────────────────────────

        lines.append(section("NUT MAP  (top-down)"))
        map_lines = _render_nut_map(
            self._trees,
            self._visited_tree_ids,
            self._odom_x,
            self._odom_y,
            self._odom_yaw,
            self._trail,
            width=62,
            height=14,
        )
        lines.extend(map_lines)
        lines.append(hline())

        # ── live telemetry ──────────────────────────────────────────────

        lines.append(section("LIVE TELEMETRY"))
        lines.append(
            f"  cmd  lin={self._cmd_lin:+.3f} m/s   ang={self._cmd_ang:+.3f} rad/s"
        )
        perp = _extract_perp(self._status)
        perp_str = f"{perp:+.3f} m" if perp is not None else "—"
        lines.append(f"  perp dist to row: {perp_str}")
        # Truncate status to fit terminal
        short_status = self._status[:W - 4] if self._status else "(waiting)"
        lines.append(f"  {_DIM}{short_status}{_RESET}")
        lines.append(hline())

        # Print — clear screen then dump all lines
        out = _CLEAR_SCREEN + "\n".join(lines) + "\n"
        print(out, end="", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = HarvestDashboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal
        print(_RESET, end="", flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────
# To register with ROS2, add this line to setup.py console_scripts:
#
#   'harvest_dashboard = macadamia_sweep.harvest_dashboard:main',
#
# ──────────────────────────────────────────────────────────────────────────────
