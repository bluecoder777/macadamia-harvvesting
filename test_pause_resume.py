#!/usr/bin/env python3
"""Offline test for PAUSE / RESUME + the bag auto-pause.

Stubs ROS, imports the REAL simple_row_follower, bypasses __init__, and drives
the real pause/resume methods with light stubs for the I/O (set_perception,
_drive_to, clock/logger). No ROS, no robot.

Covers:
  * _begin_pause snapshots row/side/along/anchor, freezes perception, and routes
    to RETURN_HOME with the hand-off set to PAUSED;
  * _finish_return routes a completed return to PAUSED, and resume_callback then
    leaves PAUSED into RESUME_NAV;
  * resume_nav waypoint geometry: the two buffer waypoints sit south of the row
    starts (along < 0) and the last waypoint is the row's start anchor; on
    arrival it restores the paused row and re-enters via the crab;
  * the detection gate flips perception back on exactly when along >= pause_along;
  * bag: collecting `bag` nuts triggers a pause, and resume rebases so the next
    `bag` NEW nuts trigger another.

Exit code 0 = all passed.
"""

import sys
import math
import types

# --- Stub every ROS import simple_row_follower pulls in, before importing it ---

def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

_mod("rclpy").init = lambda *a, **k: None
node_mod = _mod("rclpy.node")
class _Node:
    def __init__(self, *a, **k):
        pass
node_mod.Node = _Node

_mod("rclpy.executors").ExternalShutdownException = type(
    "ExternalShutdownException", (Exception,), {})

qos_mod = _mod("rclpy.qos")
qos_mod.QoSProfile = type("QoSProfile", (), {"__init__": lambda self, *a, **k: None})
qos_mod.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1, VOLATILE=0)
qos_mod.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)

time_mod = _mod("rclpy.time")
time_mod.Time = type("Time", (), {"__init__": lambda self, *a, **k: None})

geo = _mod("geometry_msgs.msg")
for _n in ("TwistStamped", "PoseArray"):
    setattr(geo, _n, type(_n, (), {}))
nav = _mod("nav_msgs.msg")
nav.Odometry = type("Odometry", (), {})
sens = _mod("sensor_msgs.msg")
sens.LaserScan = type("LaserScan", (), {})
sens.BatteryState = type("BatteryState", (), {})
std = _mod("std_msgs.msg")
for _n in ("Empty", "String", "Bool", "Int32"):
    setattr(std, _n, type(_n, (), {}))
tf2 = _mod("tf2_ros")
tf2.Buffer = type("Buffer", (), {})
tf2.TransformListener = type("TransformListener", (), {})

sys.path.insert(0, "macadamia_sweep")
from macadamia_sweep import simple_row_follower as SRF  # noqa: E402

PASS = FAIL = 0
def check(name, cond, extra=None):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  -- {extra}" if extra is not None else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  -- {extra}" if extra is not None else ""))


class _T:
    def __init__(self, ns): self.nanoseconds = ns
    def __sub__(self, other): return _T(self.nanoseconds - other.nanoseconds)
class _Clk:
    def now(self): return _T(0)
class _Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass


def make_follower():
    """A real SimpleRowFollower with __init__ bypassed and the fields/stubs the
    pause/resume methods touch."""
    f = SRF.SimpleRowFollower.__new__(SRF.SimpleRowFollower)
    f.get_clock = lambda: _Clk()
    f.get_logger = lambda: _Log()
    # I/O capture stubs
    f.perception = []                       # records each set_perception(value)
    f.set_perception = lambda v: f.perception.append(bool(v))
    f.stop_robot = lambda: None
    f.publish_cmd = lambda *a, **k: None
    f.publish_status = lambda *a, **k: None
    f._drive_targets = []                   # records each _drive_to target
    f._drive_returns = True                 # what the stubbed _drive_to returns
    def fake_drive_to(tx, ty, tol, avoid_state="COLLECT_NUTS"):
        f._drive_targets.append((tx, ty, avoid_state))
        return f._drive_returns
    f._drive_to = fake_drive_to
    # State machine fields
    f._SWEEP_STATES = ("FOLLOW_OUT", "CLEAR_END", "ARC_TURN", "ALIGN",
                       "LATERAL_ALIGN", "FOLLOW_BACK", "CLEAR_NEXT", "TURN_NEXT")
    f.state = "FOLLOW_OUT"
    f.state_start_time = _T(0)
    f.started = True
    f.return_home_enabled = True
    f.collect_before_home = True
    f.start_side = "right"
    f.desired_side_distance = 0.40
    f.uncollected_map = []
    f.rows_completed = 2
    f.current_side = "right"
    f.next_row_hits = 0
    f.seen_row_this_pass = True
    f.passed_forward_tree = True
    f.clear_start_x = f.clear_start_y = 0.0
    f.arc_start_yaw = f.arc_last_yaw = 0.0
    f.arc_accumulated_yaw = 1.0
    f._home_phase = "GOAL"
    # Pose / heading
    f.outbound_yaw = 0.0
    f.home_x = f.home_y = 0.0
    f.home_yaw = 0.0
    f.odom_x = 1.0
    f.odom_y = 0.0
    f.odom_yaw = 0.0
    # Pause/resume state
    f._paused = False
    f._resuming = False
    f._return_end_state = "DONE"
    f._pause_row = 0
    f._pause_side = "right"
    f._pause_along = 0.0
    f._pause_anchor = None
    f._row_start_anchor = (0.0, 1.40)       # row 3 start: along 0, lateral 1.40
    f._resume_phase = "HEADLAND"
    f._collected_total = 0
    f._bag_start_count = 0
    f.bag = 0
    f._row_anchors = []
    f._recollecting = False
    f._recollect_rows = []
    f.battery_safety = False
    f.battery_min_voltage = 10.0
    f.battery_low_duration = 5.0
    f._battery_voltage = None
    f._battery_low_since = None
    # Params / gains
    f.resume_headland_margin = 0.40
    f.resume_wp_tol = 0.15
    f.resume_max_duration = 90.0
    f.align_parallel_tol = math.radians(12)
    f.align_max_angular = 0.30
    f.min_align_angular = 0.08
    f.k_align = 1.2
    # crab fields touched by _enter_strafe
    f._strafe_next_state = "FOLLOW_OUT"
    f._lat_phase = "MEASURE"
    f._strafe_start = None
    return f


def along(f, x, y):
    return ((x - f.home_x) * math.cos(f.outbound_yaw)
            + (y - f.home_y) * math.sin(f.outbound_yaw))


print("=== pause: snapshot + route to RETURN_HOME (-> PAUSED) ===")
f = make_follower()
f.odom_x, f.odom_y = 1.0, 0.0          # 1.0 m up the row
f._begin_pause("manual pause")
check("paused flag set", f._paused is True)
check("perception frozen off", f.perception and f.perception[-1] is False)
check("pause row recorded (rows_completed)", f._pause_row == 2, f._pause_row)
check("pause side recorded", f._pause_side == "right")
check("pause along recorded (~1.0)", abs(f._pause_along - 1.0) < 1e-9, f._pause_along)
check("pause anchor = this row's start", f._pause_anchor == (0.0, 1.40))
check("return hand-off armed to PAUSED", f._return_end_state == "PAUSED")
check("went to RETURN_HOME", f.state == "RETURN_HOME", f.state)

# pause is ignored when not sweeping
f2 = make_follower()
f2.state = "COLLECT_NUTS"
f2._begin_pause("manual pause")
check("pause ignored when not in a sweep state", f2._paused is False and f2.state == "COLLECT_NUTS")

print("=== _finish_return routes to PAUSED, resume leaves it ===")
f._finish_return("home")
check("RETURN_HOME completion parks in PAUSED", f.state == "PAUSED", f.state)
check("still flagged paused while parked", f._paused is True)
f.resume_callback(None)
check("resume clears paused + arms resuming", f._paused is False and f._resuming is True)
check("resume enters RESUME_NAV", f.state == "RESUME_NAV", f.state)
check("resume rebases the bag", f._bag_start_count == f._collected_total)

print("=== RESUME_NAV waypoint geometry (south buffer, then the anchor) ===")
f = make_follower()
f._paused = True
f._pause_row = 0                        # paused on row 1; anchor at lateral 1.40
f._pause_side = "right"
f._pause_anchor = (0.0, 1.40)          # set by _begin_pause in the real flow
f.resume_callback(None)                 # -> RESUME_NAV, phase HEADLAND
f._drive_returns = True                 # each leg 'arrives' so phases advance
f.resume_nav(); w1 = f._drive_targets[-1]
f.resume_nav(); w2 = f._drive_targets[-1]
f.resume_nav(); w3 = f._drive_targets[-1]
check("W1 is in the south buffer (along < 0)", along(f, w1[0], w1[1]) < 0,
      f"along={along(f, w1[0], w1[1]):.2f}")
check("W1 at home's lateral, margin south", abs(w1[0] + 0.40) < 1e-9 and abs(w1[1]) < 1e-9,
      f"W1={w1[:2]}")
check("W2 still in buffer, at the row's lateral",
      abs(w2[0] + 0.40) < 1e-9 and abs(w2[1] - 1.40) < 1e-9, f"W2={w2[:2]}")
check("W3 == the row start anchor", abs(w3[0]) < 1e-9 and abs(w3[1] - 1.40) < 1e-9, f"W3={w3[:2]}")
check("legs route with avoid_state=RESUME_NAV", all(t[2] == "RESUME_NAV" for t in f._drive_targets))
# ALIGN_OUT: heading already at outbound -> restore row and enter the crab
f.resume_nav()
check("resume restores the paused row", f.rows_completed == 0 and f.current_side == "right",
      f"rows={f.rows_completed} side={f.current_side}")
check("resume hands to the crab (LATERAL_ALIGN)", f.state == "LATERAL_ALIGN", f.state)
check("crab will hand on to FOLLOW_OUT", f._strafe_next_state == "FOLLOW_OUT")
check("detection still off through re-align", f._resuming is True)

print("=== detection re-enable gate (no duplicates) ===")
f = make_follower()
f._resuming = True
f._pause_along = 0.5
f.state = "FOLLOW_OUT"
f.bag = 0
f.perception = []
f.odom_x, f.odom_y = 0.30, 0.0          # before the pause spot
f._bag_and_resume_tick()
check("detection stays OFF before the pause spot", f._resuming is True and f.perception == [],
      f"perc={f.perception}")
f.odom_x = 0.60                          # past the pause spot
f._bag_and_resume_tick()
check("detection re-enabled at the pause spot", f.perception and f.perception[-1] is True)
check("gate disarms after re-enable", f._resuming is False)

print("=== bag: auto-pause every `bag` new nuts ===")
f = make_follower()
f.bag = 3
f._bag_start_count = 0
f._collected_total = 2
f.state = "FOLLOW_OUT"
f._bag_and_resume_tick()
check("no pause below the bag (2 < 3)", f._paused is False)
f._collected_total = 3
f._bag_and_resume_tick()
check("auto-pause when the bag fills (3 >= 3)", f._paused is True and f.state == "RETURN_HOME")
# resume rebases, next bag counts only NEW nuts
f.resume_callback(None)
check("resume rebased bag_start to current total", f._bag_start_count == 3)
f.state = "FOLLOW_OUT"; f._paused = False
f._collected_total = 5                   # 2 new since resume
f._bag_and_resume_tick()
check("no pause after 2 new (5-3 < 3)", f._paused is False)
f._collected_total = 6                   # 3 new since resume
f._bag_and_resume_tick()
check("auto-pause again after 3 new (6-3 >= 3)", f._paused is True)

print("=== missed-nut collection: map nuts to rows, re-sweep each ===")
f = make_follower()
f.outbound_yaw = 0.0                     # d=(1,0); right-side hug -> sweeper_unit=(0,-1)
f.start_side = "right"
f.desired_side_distance = 0.40
# Three swept rows, 0.70 m apart laterally (more negative y = further across).
f._row_anchors = [(0.0, 0.0, "right"), (0.0, -0.70, "right"), (0.0, -1.40, "right")]
f._map_to_odom = lambda: (0.0, 0.0, 1.0, 0.0)    # identity map<->odom
# Nuts sit on each row's tree line: lateral = anchor_lateral + 0.40 (so y is
# 0.40 more negative than the anchor). Place one on row 0 and one on row 2.
f.uncollected_map = [(1.0, -0.40), (1.2, -1.80)]
rows = f._rows_with_missed_nuts()
check("missed nuts map to their rows (0 and 2, not 1)", rows == [0, 2], rows)

# Entry: queue those rows and start re-sweeping the first.
started = f._begin_recollect("Sweep done.")
check("recollect starts when rows have missed nuts", started is True)
check("recollecting flag set", f._recollecting is True)
check("first re-sweep targets row 1 (index 0) via RESUME_NAV",
      f.state == "RESUME_NAV" and f._pause_row == 0, f"state={f.state} row={f._pause_row}")
check("navigates to that row's anchor", f._pause_anchor == (0.0, 0.0), f._pause_anchor)
check("re-sweep keeps detection off (not the resume gate)", f._resuming is False)
check("queue holds the remaining row", f._recollect_rows == [2], f._recollect_rows)

# FOLLOW_BACK end on the first re-sweep -> next queued row.
f._recollect_next("Re-swept a row.")
check("second re-sweep targets row 3 (index 2)",
      f.state == "RESUME_NAV" and f._pause_row == 2, f"state={f.state} row={f._pause_row}")
check("anchor for row 3", f._pause_anchor == (0.0, -1.40), f._pause_anchor)

# Queue now empty -> go home.
f._recollect_next("Re-swept a row.")
check("recollect finishes -> RETURN_HOME", f.state == "RETURN_HOME", f.state)
check("recollecting flag cleared", f._recollecting is False)

# No missed nuts -> no re-sweep, straight home.
f2 = make_follower()
f2.outbound_yaw = 0.0
f2._row_anchors = [(0.0, 0.0, "right")]
f2._map_to_odom = lambda: (0.0, 0.0, 1.0, 0.0)
f2.uncollected_map = []
check("no re-sweep when nothing is missed", f2._begin_recollect("done") is False)

print("=== battery safety: pause + home only when enabled and held low ===")
# Disabled: low voltage does nothing.
f = make_follower()
f.state = "FOLLOW_OUT"
f.battery_safety = False
f._battery_voltage = 9.0
f._battery_safety_check()
check("disabled: low voltage does NOT pause", f._paused is False and f.state == "FOLLOW_OUT")

# Enabled, voltage fine: no pause, no low timer.
f = make_follower()
f.state = "FOLLOW_OUT"
f.battery_safety = True
f.battery_min_voltage = 10.0
f._battery_voltage = 11.5
f._battery_safety_check()
check("healthy voltage does not arm the low timer",
      f._paused is False and f._battery_low_since is None)

# Enabled, just dropped low: debounced, not yet paused.
f._battery_voltage = 9.5
f._battery_safety_check()
check("low voltage arms the timer but waits (debounce)",
      f._paused is False and f._battery_low_since is not None)

# Held low past the duration -> pause + return home.
f._battery_low_since = _T(-6 * 10**9)      # 6 s ago (> 5 s duration)
f._battery_safety_check()
check("held low past duration pauses + returns home",
      f._paused is True and f.state == "RETURN_HOME")

# Recovery clears the timer (only relevant before a trip).
f2 = make_follower()
f2.state = "FOLLOW_OUT"
f2.battery_safety = True
f2.battery_min_voltage = 10.0
f2._battery_voltage = 9.0
f2._battery_safety_check()                  # arm
f2._battery_voltage = 11.0
f2._battery_safety_check()                  # recover
check("voltage recovery clears the low timer", f2._battery_low_since is None
      and f2._paused is False)

# Not sweeping: timer resets, no pause (e.g. already returning home).
f3 = make_follower()
f3.state = "RETURN_HOME"
f3.battery_safety = True
f3._battery_voltage = 9.0
f3._battery_low_since = _T(-9 * 10**9)
f3._battery_safety_check()
check("not in a sweep state: no battery pause", f3._paused is False
      and f3._battery_low_since is None)

print()
print("=" * 60)
print(f"PAUSE/RESUME TEST: {PASS}/{PASS + FAIL} checks passed")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
