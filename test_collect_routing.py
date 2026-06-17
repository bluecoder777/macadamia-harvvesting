#!/usr/bin/env python3
"""Offline test for COLLECT_NUTS inter-aisle routing (option 2).

Stubs ROS, imports the REAL simple_row_follower, bypasses __init__, and drives
the real _drive_to_via_path over a synthetic swept polyline shaped like two
aisles joined by a U-turn AROUND the end of the tree row. A straight line
between the aisles would cut through the tree row; the corridor router must
instead follow the recorded path around the end and never enter the gap.

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
std = _mod("std_msgs.msg")
for _n in ("Empty", "String", "Bool"):
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


def make_follower():
    f = SRF.SimpleRowFollower.__new__(SRF.SimpleRowFollower)
    f.path_wp_tol = 0.20
    f.path_lookahead = 1
    f.collect_arrive_tol = 0.15
    f._path_idx = None
    f._path_goal = None
    f.odom_x = 0.0
    f.odom_y = 0.0
    # Stub the low-level driver: step the robot toward the commanded point by a
    # fixed amount, and report arrival within tol. No ROS, no avoidance.
    f._drive_calls = []
    def fake_drive_to(tx, ty, tol):
        f._drive_calls.append((tx, ty))
        dx, dy = tx - f.odom_x, ty - f.odom_y
        d = math.hypot(dx, dy)
        if d <= tol:
            f.odom_x, f.odom_y = tx, ty
            return True
        stepd = min(0.08, d)
        f.odom_x += stepd * dx / d
        f.odom_y += stepd * dy / d
        return d - stepd <= tol
    f._drive_to = fake_drive_to
    return f


# Swept polyline: aisle A at y=0 (x 0->3), U-turn arc AROUND the row end at
# x~=3 (clear, past the trees), aisle B at y=-0.7 (x 3->0). Trees sit at
# y=-0.35 for x in [0, 2.5]; the gap between aisles is the forbidden region.
print("=== collect routing: follow the corridor, never cross the tree row ===")
path = []
x = 0.0
while x <= 3.0001:
    path.append((round(x, 3), 0.0))
    x += 0.15
# half-circle (right half) connecting (3.0,0) to (3.0,-0.7) around the row end,
# center (3.0,-0.35), radius 0.35, sampled ~0.14 m like a real recording.
cx, cy, r = 3.0, -0.35, 0.35
steps = 8
for i in range(1, steps):
    ang = math.pi / 2 - math.pi * i / steps     # +90deg down to -90deg
    path.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
x = 3.0
while x >= -0.0001:
    path.append((round(x, 3), -0.7))
    x -= 0.15

f = make_follower()
f._swept_path = path

# Robot starts at the end of aisle B; entry is a point partway up aisle A. The
# straight line between them would cross y=-0.35 at x~0.5 (inside the row).
f.odom_x, f.odom_y = 0.0, -0.7
ENTRY = (1.0, 0.0)

def in_tree_row(px, py):
    return (abs(py - (-0.35)) < 0.15) and (px < 2.6)

check("straight line WOULD cross the tree row (control)",
      in_tree_row(0.5, -0.35), "x=0.5,y=-0.35")

trail = [(f.odom_x, f.odom_y)]
arrived = False
for _ in range(4000):
    if f._drive_to_via_path(ENTRY[0], ENTRY[1]):
        arrived = True
        break
    trail.append((f.odom_x, f.odom_y))

violations = [(px, py) for (px, py) in trail if in_tree_row(px, py)]
check("router reached the entry on the far aisle", arrived,
      f"end=({f.odom_x:.2f},{f.odom_y:.2f})")
check("no trajectory point entered the tree-row gap",
      len(violations) == 0, f"{len(violations)} violations")
check("router actually went around the end (reached x>=2.6)",
      max(px for (px, py) in trail) >= 2.6,
      f"max_x={max(px for (px, py) in trail):.2f}")
# Every visited point should lie near SOME recorded path point (on corridor).
max_off = 0.0
for (px, py) in trail:
    nearest = min(math.hypot(px - qx, py - qy) for (qx, qy) in path)
    max_off = max(max_off, nearest)
check("trajectory stayed on the recorded corridor (<0.2 m off)",
      max_off < 0.20, f"max_off={max_off:.3f} m")

# Same-aisle move must NOT detour around the end: entry ahead in aisle B.
f2 = make_follower()
f2._swept_path = path
f2.odom_x, f2.odom_y = 0.0, -0.7
ENTRY2 = (1.5, -0.7)
trail2 = [(f2.odom_x, f2.odom_y)]
arrived2 = False
for _ in range(4000):
    if f2._drive_to_via_path(ENTRY2[0], ENTRY2[1]):
        arrived2 = True
        break
    trail2.append((f2.odom_x, f2.odom_y))
check("same-aisle move reached the entry", arrived2,
      f"end=({f2.odom_x:.2f},{f2.odom_y:.2f})")
check("same-aisle move did NOT detour around the end (max_x<2.0)",
      max(px for (px, py) in trail2) < 2.0,
      f"max_x={max(px for (px, py) in trail2):.2f}")

# No-path fallback: with <2 samples it should still drive straight to the goal.
f3 = make_follower()
f3._swept_path = [(0.0, 0.0)]
f3.odom_x, f3.odom_y = 0.0, 0.0
arrived3 = False
for _ in range(4000):
    if f3._drive_to_via_path(0.5, 0.0):
        arrived3 = True
        break
check("fallback drives straight when no path recorded", arrived3,
      f"end=({f3.odom_x:.2f},{f3.odom_y:.2f})")

# ===========================================================================
print("=== safety-point geometry (staging + dip) ===")
# Rows along +x (outbound_yaw=0); hug side = right = -y. A nut on the hug side
# at y=-0.40 corresponds to a robot aisle at y=0. Swept long [0,5], lat [-1,0.5].
def make_safety_follower():
    f = SRF.SimpleRowFollower.__new__(SRF.SimpleRowFollower)
    f.outbound_yaw = 0.0
    f.start_side = "right"
    f.collect_sweep_offset = 0.40
    f.collect_sweep_through = 0.60
    f._swept_long_min, f._swept_long_max = 0.0, 5.0
    f._swept_lat_min, f._swept_lat_max = -1.0, 0.5
    return f

fs = make_safety_follower()
# Nut near the FAR end (x=3 of 0..5) -> approach from the far headland.
s_far = fs._safety_for_nut(3.0, -0.40)
sx, sy, dipx, dipy, end_long, l_lat = s_far
check("aisle lateral recovered (nut 0.40 off the aisle)", abs(l_lat - 0.0) < 1e-6,
      f"l_lat={l_lat:.3f}")
check("far-end nut staged at the FAR headland (x=5)", abs(sx - 5.0) < 1e-6,
      f"safety=({sx:.2f},{sy:.2f})")
check("safety point sits IN the aisle (y=0)", abs(sy - 0.0) < 1e-6, f"sy={sy:.3f}")
check("dip overshoots the nut toward the entry end (x=2.4)", abs(dipx - 2.4) < 1e-6,
      f"dipx={dipx:.3f}")
check("dip stays in the aisle (y=0), never toward the trees", abs(dipy - 0.0) < 1e-6,
      f"dipy={dipy:.3f}")

# Nut near the NEAR end (x=1) -> approach from the near headland (x=0).
s_near = fs._safety_for_nut(1.0, -0.40)
check("near-end nut staged at the NEAR headland (x=0)", abs(s_near[0] - 0.0) < 1e-6,
      f"safety_x={s_near[0]:.2f}")
check("near-end dip overshoots away from entry (x=1.6)", abs(s_near[2] - 1.6) < 1e-6,
      f"dipx={s_near[2]:.3f}")

# A nut whose aisle falls outside the swept columns is clamped back in. The
# lateral is in the hug-unit basis (hug = -y here), so a nut deep in -y maps to
# a LARGE positive lat (1.6) and clamps to the column max (0.5), not the min.
s_clip = fs._safety_for_nut(3.0, -2.0)
check("aisle lateral clamped into the swept columns", abs(s_clip[5] - 0.5) < 1e-6,
      f"l_lat={s_clip[5]:.3f}")

print()
print("=" * 60)
print(f"COLLECT ROUTING TEST: {PASS}/{PASS + FAIL} checks passed")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
