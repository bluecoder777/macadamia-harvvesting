#!/usr/bin/env python3
"""Perception: the situated-recognition half of the reactive/skill tier.

Pure functions of the cached LiDAR scan + odom + parameters in the world
model. They turn raw sensor data into the percepts the skills use to steer and
the events the sequencer uses to decide transitions (row line fit, front
distance, "last tree abeam", next-row visibility). No actuation, no state
changes -- recognition only.
"""

import math
from typing import List, Optional, Tuple

from .util import normalize_angle


class Perception:
    def __init__(self, wm):
        self.wm = wm

    def valid_range(self, r: float, scan) -> bool:
        return (
            not math.isnan(r)
            and not math.isinf(r)
            and scan.range_min <= r <= scan.range_max
            and r >= self.wm.lidar_min_range
        )

    def collect_row_side_points(self, side: str) -> List[Tuple[float, float, float]]:
        scan = self.wm.latest_scan
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
            angle = scan.angle_min + i * scan.angle_increment + self.wm.lidar_yaw_offset
            angle = normalize_angle(angle)
            if not (min_a <= angle <= max_a):
                continue
            if not self.valid_range(r, scan):
                continue
            if r > self.wm.tree_max_range:
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
            if d <= self.wm.cluster_gap:
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
            if extent > self.wm.tree_max_extent:
                continue
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            trees.append((mx, my, len(c)))
        return trees

    def row_dir_robot_frame(self) -> Optional[float]:
        """Known row direction expressed in the robot frame."""
        if self.wm.outbound_yaw is None or self.wm.odom_yaw is None:
            return None
        return normalize_angle(self.wm.outbound_yaw - self.wm.odom_yaw)

    def select_nearest_row(
        self, trees: List[Tuple[float, float, int]], side: str
    ) -> List[Tuple[float, float, int]]:
        """Keep only the NEAREST row of trees on the requested side."""
        if not trees:
            return []
        a = self.row_dir_robot_frame()
        if a is None:
            a = 0.0  # robot parallel to row (true at start, before anchor)
        # A row is a LINE - its direction is only defined mod pi. Use the
        # representative closest to the robot's forward axis so that
        # perp > 0 always means the ROBOT's left ('side' is a robot-frame
        # concept).
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
            if side == "right" and perp > -self.wm.min_side_offset:
                continue
            if side == "left" and perp < self.wm.min_side_offset:
                continue
            tagged.append((perp, tx, ty, n))
        if not tagged:
            return []

        tagged.sort(key=lambda t: t[0])
        groups: List[List[Tuple[float, float, float, int]]] = [[tagged[0]]]
        for prev, cur in zip(tagged, tagged[1:]):
            if abs(cur[0] - prev[0]) <= self.wm.row_group_gap:
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
        if len(trees) < self.wm.min_trees_for_fit:
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
        # Same nearest-row gating as the fit, so "last tree abeam" is judged
        # against the CURRENT row only, not a neighbouring one.
        trees = self.select_nearest_row(trees, side)
        return [(t[0], t[1]) for t in trees]

    def get_front_distance(self) -> float:
        scan = self.wm.latest_scan
        if scan is None:
            return 10.0
        half_w = self.wm.front_width / 2.0
        vals = []
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment + self.wm.lidar_yaw_offset
            if abs(normalize_angle(angle - 0.0)) > half_w:
                continue
            if not self.valid_range(r, scan):
                continue
            if r > 3.0:
                continue
            vals.append(r)
        return min(vals) if vals else 10.0

    def get_sector_distance(self, min_deg: float, max_deg: float) -> float:
        """Return nearest valid LiDAR distance in an angle sector."""
        scan = self.wm.latest_scan
        if scan is None:
            return 10.0

        vals = []
        min_a = math.radians(min_deg)
        max_a = math.radians(max_deg)

        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment + self.wm.lidar_yaw_offset
            angle = normalize_angle(angle)

            if min_a <= angle <= max_a and self.valid_range(r, scan):
                vals.append(r)

        return min(vals) if vals else 10.0

    def row_visible(self, side: str) -> Tuple[bool, Optional[Tuple[float, float, int]]]:
        fit = self.fit_row_line(side)
        return (fit is not None), fit

    def last_tree_abeam_or_behind(self, side: str) -> bool:
        if not self.wm.seen_row_this_pass:
            return False
        pts = self.side_cone_points(side)
        if not pts:
            return False
        max_x = max(p[0] for p in pts)
        # Arm the cut only once a tree has been clearly AHEAD this pass.
        if max_x > self.wm.forward_tree_arm:
            self.wm.passed_forward_tree = True
        if not self.wm.passed_forward_tree:
            return False
        ahead_threshold = 0.05
        return max_x <= ahead_threshold
