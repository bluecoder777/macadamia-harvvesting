#!/usr/bin/env python3
"""World model: the shared blackboard for the 3T architecture.

Holds every piece of mutable state, sensor cache and tunable parameter the
three tiers read and write. The values below are the *defaults* and the fixed
tunings, identical to the original ``SimpleRowFollower.__init__``. The agent
(ROS node) re-reads the declared ROS parameters at start-up and overrides the
parameter-backed fields, so launch-time overrides (``-p start_side:=left``
etc.) behave exactly as before. Constructing a ``WorldModel`` on its own (no
ROS) therefore reproduces the default-parameter robot, which is what the
offline tests rely on.
"""

import math
from typing import List, Optional, Tuple

from .util import Skill


class WorldModel:
    def __init__(self):
        # ---- Parameters (defaults mirror the agent's declare_parameter) -----
        self.start_side = "right"
        self.lidar_yaw_offset = math.radians(180.0)

        self.max_rows = 0
        self.next_row_min_hits = 8
        self.next_row_max_dist = 0.60
        self.next_row_hits = 0
        self.rows_completed = 0

        # Tree-aware collection of uncollected nuts (before return-home).
        self.collect_before_home = True
        self.nuts_topic = "/nuts/uncollected"
        self.map_frame = "map"
        self.odom_frame = "odom"
        self.collect_sweep_offset = 0.40
        self.collect_sweep_through = 0.60
        self.collect_arrive_tol = 0.15
        self.collect_visit_timeout = 45.0
        self.collect_max_avoid = 3
        self.collect_max_consec_skips = 3
        self.collect_side_clearance = 0.25
        self.collect_turn_away = 0.30
        # Inter-aisle moves follow the recorded swept polyline.
        self.path_sample_spacing = 0.15
        self.path_max_points = 4000
        self.path_wp_tol = 0.20
        self.path_lookahead = 1

        # Pause/resume + bag. bag=0 disables the auto-pause; bag>0 parks at home
        # after that many nuts are collected, then again after each resume.
        self.bag = 0
        self.resume_headland_margin = 0.40
        self.resume_wp_tol = 0.15
        self.resume_max_duration = 90.0      # per-leg stuck cap
        # Battery safety (opt-in): pause + return home when voltage stays low.
        self.battery_safety = False
        self.battery_min_voltage = 10.0      # volts; tune per pack
        self.battery_low_duration = 5.0      # s held low before pausing

        self.uncollected_map: List[Tuple[float, float]] = []
        self._collect_skip = set()
        self._collect_target: Optional[Tuple[float, float]] = None
        self._collect_phase = "TO_HEADLAND"
        self._collect_deadline: Optional[float] = None
        self._collect_avoid_count = 0
        self._collect_consec_skips = 0
        self._collect_best = float("inf")
        self._path_idx: Optional[int] = None
        self._path_goal: Optional[int] = None

        # Pause/resume state. On pause the robot parks at home and remembers the
        # row it was on; on resume it skips back to that row's start (RESUME_NAV)
        # and re-sweeps, detection held off until it reaches the pause spot.
        self._paused = False
        self._resuming = False
        self._return_end_state = "DONE"      # RETURN_HOME hand-off (DONE / PAUSED)
        self._pause_row = 0                   # rows_completed at pause
        self._pause_side = "right"            # current_side at pause
        self._pause_along = 0.0               # along-row progress (rel. home) at pause
        self._pause_anchor: Optional[Tuple[float, float]] = None   # paused row start
        self._row_start_anchor: Optional[Tuple[float, float]] = None  # this row's start
        self._resume_phase = "HEADLAND"       # RESUME_NAV sub-phase
        self._collected_total = 0             # latest /nuts/collected_count
        self._bag_start_count = 0             # collected total when this bag began
        # Missed-nut collection by RE-SWEEPING (same path as resume): the start
        # anchor of every forward-swept row, and the queue of rows to re-sweep.
        self._row_anchors: List[Tuple[float, float, str]] = []   # (x,y,side) per row
        self._recollecting = False
        self._recollect_rows: List[int] = []  # row indices still to re-sweep
        # Battery safety: latest voltage + when it first dropped below threshold.
        self._battery_voltage: Optional[float] = None
        self._battery_low_since = None

        # Bounding box of the swept region, in ROW-FRAME coords.
        self._swept_long_min: Optional[float] = None
        self._swept_long_max: Optional[float] = None
        self._swept_lat_min: Optional[float] = None
        self._swept_lat_max: Optional[float] = None
        self._swept_path: List[Tuple[float, float]] = []
        # States in which the robot is actively sweeping (pose marks free ground).
        self._SWEEP_STATES = (
            "FOLLOW_OUT", "CLEAR_END", "ARC_TURN", "ALIGN", "LATERAL_ALIGN",
            "FOLLOW_BACK", "CLEAR_NEXT", "TURN_NEXT",
        )

        # ---- Sensor caches --------------------------------------------------
        self.latest_scan = None
        self.odom_x: Optional[float] = None
        self.odom_y: Optional[float] = None
        self.odom_yaw: Optional[float] = None

        # ---- Mission / return-home bookkeeping ------------------------------
        self.home_x: Optional[float] = None
        self.home_y: Optional[float] = None
        self.home_yaw: Optional[float] = None

        self.started = False
        self.state = Skill.WAITING
        # Clock-stamped instants. Set by the agent at start-up and on every
        # transition (the agent passes its rclpy clock time straight through).
        self.state_start_time = None
        self.last_row_seen_time = None
        self.seen_row_this_pass = False
        self.passed_forward_tree = False

        # Side the row is on for the CURRENT pass.
        self.current_side = self.start_side

        # CLEAR_END / arc bookkeeping.
        self.clear_start_x: Optional[float] = None
        self.clear_start_y: Optional[float] = None
        self.arc_start_yaw: Optional[float] = None
        self.arc_last_yaw: Optional[float] = None
        self.arc_accumulated_yaw = 0.0
        self.arc_center_x: Optional[float] = None
        self.arc_center_y: Optional[float] = None
        self._last_row_perp: Optional[float] = None

        # Heading anchor (row direction snapshot at /sweep_start).
        self.outbound_yaw: Optional[float] = None
        self.return_target_yaw: Optional[float] = None

        # ALIGN gains.
        self.k_align = 1.2
        self.min_align_angular = 0.08

        # Row-following tuning.
        self.desired_side_distance = 0.40
        self.too_close_side = 0.25
        self.tree_max_range = 1.20
        self.lidar_min_range = 0.25

        self.emergency_stop_distance = 0.18

        self.forward_speed = 0.06
        self.search_speed = 0.03

        self.k_xtrack = 1.2
        self.k_heading = 1.4
        self.max_follow_angular = 0.30

        self.start_straight_duration = 1.5

        # LATERAL_ALIGN (crab strafe).
        self.lateral_align_tol = 0.05
        self.lateral_align_max_duration = 20.0
        self.strafe_turn_speed = 0.6
        self.strafe_drive_speed = 0.08
        self.strafe_max_dist = 0.30
        self._lat_phase = "MEASURE"
        self._strafe_dist = 0.0
        self._strafe_heading = 0.0
        self._strafe_ref_yaw = 0.0
        self._strafe_toward = False
        self._strafe_start: Optional[Tuple[float, float]] = None
        self._strafe_next_state = "FOLLOW_OUT"

        # Row detection (cluster-then-fit).
        self.cluster_gap = 0.15
        self.tree_max_extent = 0.15
        self.row_group_gap = 0.35
        self.min_side_offset = 0.05
        self.min_trees_for_fit = 2
        self.front_width = math.radians(35)
        self.max_tree_points = 80

        self.min_pass_duration = 10.0
        self.max_pass_duration = 90.0
        self.row_lost_timeout = 2.0
        self.forward_tree_arm = 0.25
        self.return_end_margin = 0.0

        # CLEAR_END.
        self.clear_end_distance = 0.15
        self.clear_end_max_time = 8.0

        # Arc U-turn (tight).
        self.arc_radius = 0.40
        self.arc_linear_speed = 0.06
        self.arc_max_yaw = math.radians(220.0)
        nominal_omega = self.arc_linear_speed / self.arc_radius
        self.arc_max_duration = self.arc_max_yaw / nominal_omega + 5.0

        # ALIGN.
        self.align_max_angular = 0.30
        self.align_parallel_tol = math.radians(12.0)
        self.align_max_duration = 8.0

        # TURN_NEXT (in-place 180 deg toward the next row).
        self.turn_next_max_duration = 18.0

        self.recovery_angular = 0.20

        # RETURN_HOME.
        self.return_home_enabled = True
        self.return_goal_tolerance = 0.12
        self.return_yaw_tolerance = math.radians(12.0)
        self.return_max_duration = 120.0
        self.return_exit_margin = 0.60
        self.return_linear_speed = 0.07
        self.return_max_angular = 0.35
        self.return_heading_slowdown = math.radians(35.0)
        self._home_phase = "GOAL"

        # FRONT OBSTACLE AVOIDANCE.
        self.avoid_front_distance = 0.35
        self.avoid_backup_speed = -0.05
        self.avoid_turn_speed = 0.28
        self.avoid_forward_speed = 0.05
        self.avoid_backup_duration = 2.0
        self.avoid_turn_duration = 2.2
        self.avoid_forward_duration = 2.0
        self.avoid_previous_state = "FOLLOW_OUT"
        self.avoid_phase = "BACKUP"
        self.avoid_phase_start_time = None
