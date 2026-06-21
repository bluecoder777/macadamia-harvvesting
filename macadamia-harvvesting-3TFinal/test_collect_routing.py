#!/usr/bin/env python3
"""Offline test for COLLECT_NUTS inter-aisle routing (option 2).

Drives the REAL collection skill (``three_tier.skills.Skills``) over a synthetic
swept polyline shaped like two aisles joined by a U-turn AROUND the end of the
tree row. A straight line between the aisles would cut through the tree row; the
corridor router must instead follow the recorded path around the end and never
enter the gap. Also checks the safety-point staging/dip geometry.

The skill-layer modules import no ROS, so no stubbing is needed.

Exit code 0 = all passed.
"""

import sys
import math

sys.path.insert(0, "macadamia_sweep")
from macadamia_sweep.three_tier.world_model import WorldModel   # noqa: E402
from macadamia_sweep.three_tier.skills import Skills            # noqa: E402

PASS = FAIL = 0
def check(name, cond, extra=None):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  -- {extra}" if extra is not None else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  -- {extra}" if extra is not None else ""))


class _IO:
    """No-op io: the routing/geometry paths under test never actuate (the
    low-level driver is monkeypatched and the safety geometry is pure)."""
    def stop_robot(self):
        pass
    def publish_cmd(self, *a, **k):
        pass
    def publish_status(self, *a, **k):
        pass


def make_skills():
    wm = WorldModel()
    s = Skills(wm, _IO(), None)
    wm.path_wp_tol = 0.20
    wm.path_lookahead = 1
    wm.collect_arrive_tol = 0.15
    wm._path_idx = None
    wm._path_goal = None
    wm.odom_x = 0.0
    wm.odom_y = 0.0
    # Stub the low-level driver: step the robot toward the commanded point by a
    # fixed amount, and report arrival within tol. No ROS, no avoidance.
    s._drive_calls = []
    def fake_drive_to(tx, ty, tol):
        s._drive_calls.append((tx, ty))
        dx, dy = tx - wm.odom_x, ty - wm.odom_y
        d = math.hypot(dx, dy)
        if d <= tol:
            wm.odom_x, wm.odom_y = tx, ty
            return True
        stepd = min(0.08, d)
        wm.odom_x += stepd * dx / d
        wm.odom_y += stepd * dy / d
        return d - stepd <= tol
    s._drive_to = fake_drive_to
    return s, wm


# Swept polyline: aisle A at y=0 (x 0->3), U-turn arc AROUND the row end at
# x~=3 (clear, past the trees), aisle B at y=-0.7 (x 3->0). Trees sit at
# y=-0.35 for x in [0, 2.5]; the gap between aisles is the forbidden region.
print("=== collect routing: follow the corridor, never cross the tree row ===")
path = []
x = 0.0
while x <= 3.0001:
    path.append((round(x, 3), 0.0))
    x += 0.15
cx, cy, r = 3.0, -0.35, 0.35
steps = 8
for i in range(1, steps):
    ang = math.pi / 2 - math.pi * i / steps     # +90deg down to -90deg
    path.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
x = 3.0
while x >= -0.0001:
    path.append((round(x, 3), -0.7))
    x -= 0.15

s, wm = make_skills()
wm._swept_path = path

wm.odom_x, wm.odom_y = 0.0, -0.7
ENTRY = (1.0, 0.0)

def in_tree_row(px, py):
    return (abs(py - (-0.35)) < 0.15) and (px < 2.6)

check("straight line WOULD cross the tree row (control)",
      in_tree_row(0.5, -0.35), "x=0.5,y=-0.35")

trail = [(wm.odom_x, wm.odom_y)]
arrived = False
for _ in range(4000):
    if s._drive_to_via_path(ENTRY[0], ENTRY[1]):
        arrived = True
        break
    trail.append((wm.odom_x, wm.odom_y))

violations = [(px, py) for (px, py) in trail if in_tree_row(px, py)]
check("router reached the entry on the far aisle", arrived,
      f"end=({wm.odom_x:.2f},{wm.odom_y:.2f})")
check("no trajectory point entered the tree-row gap",
      len(violations) == 0, f"{len(violations)} violations")
check("router actually went around the end (reached x>=2.6)",
      max(px for (px, py) in trail) >= 2.6,
      f"max_x={max(px for (px, py) in trail):.2f}")
max_off = 0.0
for (px, py) in trail:
    nearest = min(math.hypot(px - qx, py - qy) for (qx, qy) in path)
    max_off = max(max_off, nearest)
check("trajectory stayed on the recorded corridor (<0.2 m off)",
      max_off < 0.20, f"max_off={max_off:.3f} m")

# Same-aisle move must NOT detour around the end: entry ahead in aisle B.
s2, wm2 = make_skills()
wm2._swept_path = path
wm2.odom_x, wm2.odom_y = 0.0, -0.7
ENTRY2 = (1.5, -0.7)
trail2 = [(wm2.odom_x, wm2.odom_y)]
arrived2 = False
for _ in range(4000):
    if s2._drive_to_via_path(ENTRY2[0], ENTRY2[1]):
        arrived2 = True
        break
    trail2.append((wm2.odom_x, wm2.odom_y))
check("same-aisle move reached the entry", arrived2,
      f"end=({wm2.odom_x:.2f},{wm2.odom_y:.2f})")
check("same-aisle move did NOT detour around the end (max_x<2.0)",
      max(px for (px, py) in trail2) < 2.0,
      f"max_x={max(px for (px, py) in trail2):.2f}")

# No-path fallback: with <2 samples it should still drive straight to the goal.
s3, wm3 = make_skills()
wm3._swept_path = [(0.0, 0.0)]
wm3.odom_x, wm3.odom_y = 0.0, 0.0
arrived3 = False
for _ in range(4000):
    if s3._drive_to_via_path(0.5, 0.0):
        arrived3 = True
        break
check("fallback drives straight when no path recorded", arrived3,
      f"end=({wm3.odom_x:.2f},{wm3.odom_y:.2f})")

# ===========================================================================
print("=== safety-point geometry (staging + dip) ===")
def make_safety_skills():
    wm = WorldModel()
    s = Skills(wm, _IO(), None)
    wm.outbound_yaw = 0.0
    wm.start_side = "right"
    wm.collect_sweep_offset = 0.40
    wm.collect_sweep_through = 0.60
    wm._swept_long_min, wm._swept_long_max = 0.0, 5.0
    wm._swept_lat_min, wm._swept_lat_max = -1.0, 0.5
    return s, wm

fs, _ = make_safety_skills()
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

s_near = fs._safety_for_nut(1.0, -0.40)
check("near-end nut staged at the NEAR headland (x=0)", abs(s_near[0] - 0.0) < 1e-6,
      f"safety_x={s_near[0]:.2f}")
check("near-end dip overshoots away from entry (x=1.6)", abs(s_near[2] - 1.6) < 1e-6,
      f"dipx={s_near[2]:.3f}")

s_clip = fs._safety_for_nut(3.0, -2.0)
check("aisle lateral clamped into the swept columns", abs(s_clip[5] - 0.5) < 1e-6,
      f"l_lat={s_clip[5]:.3f}")

print()
print("=" * 60)
print(f"COLLECT ROUTING TEST: {PASS}/{PASS + FAIL} checks passed")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
