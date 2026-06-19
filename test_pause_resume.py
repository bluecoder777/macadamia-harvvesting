#!/usr/bin/env python3
"""Offline test for PAUSE / RESUME + the bag auto-pause + re-sweep collection
+ battery safety, against the 3T architecture.

Builds the real three-tier stack (WorldModel + Skills + Sequencer + Deliberator)
with a stub io, and drives the real pause/resume/battery/recollect logic. The
skill-layer modules import no ROS, so no stubbing is needed.

Exit code 0 = all passed.
"""

import sys
import math

sys.path.insert(0, "macadamia_sweep")
from macadamia_sweep.three_tier.world_model import WorldModel    # noqa: E402
from macadamia_sweep.three_tier.skills import Skills             # noqa: E402
from macadamia_sweep.three_tier.sequencer import Sequencer       # noqa: E402
from macadamia_sweep.three_tier.deliberator import Deliberator   # noqa: E402

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
    def __sub__(self, o): return _T(self.nanoseconds - o.nanoseconds)


class _IO:
    """Stub io: records perception toggles + drive targets; no ROS."""
    def __init__(self):
        self.perception = []
        self.m2o = None
    def now(self): return _T(0)
    def elapsed_since(self, t): return (0 - t.nanoseconds) / 1e9
    def log_info(self, *a, **k): pass
    def log_warn(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def publish_cmd(self, *a, **k): pass
    def publish_status(self, *a, **k): pass
    def stop_robot(self): pass
    def set_perception(self, v): self.perception.append(bool(v))
    def map_to_odom(self): return self.m2o


def make_stack():
    wm = WorldModel()
    io = _IO()
    skills = Skills(wm, io, None)
    seq = Sequencer(wm, io, None, skills)
    delib = Deliberator(wm, io, seq)
    seq.deliberator = delib
    skills.enter_avoid = seq.enter_avoid_front
    delib.skills = skills

    # The state-machine fields the methods under test touch.
    wm.state = "FOLLOW_OUT"
    wm.state_start_time = _T(0)
    wm.started = True
    wm.rows_completed = 2
    wm.current_side = "right"
    wm.start_side = "right"
    wm.outbound_yaw = 0.0
    wm.home_x = wm.home_y = 0.0
    wm.home_yaw = 0.0
    wm.odom_x = 1.0
    wm.odom_y = 0.0
    wm.odom_yaw = 0.0
    wm._row_start_anchor = (0.0, 1.40)       # row 3 start: along 0, lateral 1.40

    # Stub the low-level driver: record each target + 'arrive' so phases advance.
    drv = {"returns": True}
    def fake_drive_to(tx, ty, tol, avoid_state="COLLECT_NUTS"):
        skills._drive_targets.append((tx, ty, avoid_state))
        return drv["returns"]
    skills._drive_targets = []
    skills._drive_to = fake_drive_to
    return wm, io, skills, seq, delib, drv


def along(wm, x, y):
    return ((x - wm.home_x) * math.cos(wm.outbound_yaw)
            + (y - wm.home_y) * math.sin(wm.outbound_yaw))


print("=== pause: snapshot + route to RETURN_HOME (-> PAUSED) ===")
wm, io, skills, seq, delib, drv = make_stack()
wm.odom_x, wm.odom_y = 1.0, 0.0          # 1.0 m up the row
delib._begin_pause("manual pause")
check("paused flag set", wm._paused is True)
check("perception frozen off", io.perception and io.perception[-1] is False)
check("pause row recorded (rows_completed)", wm._pause_row == 2, wm._pause_row)
check("pause side recorded", wm._pause_side == "right")
check("pause along recorded (~1.0)", abs(wm._pause_along - 1.0) < 1e-9, wm._pause_along)
check("pause anchor = this row's start", wm._pause_anchor == (0.0, 1.40))
check("return hand-off armed to PAUSED", wm._return_end_state == "PAUSED")
check("went to RETURN_HOME", wm.state == "RETURN_HOME", wm.state)

# pause is ignored when not sweeping
wm2, io2, skills2, seq2, delib2, _ = make_stack()
wm2.state = "COLLECT_NUTS"
delib2._begin_pause("manual pause")
check("pause ignored when not in a sweep state",
      wm2._paused is False and wm2.state == "COLLECT_NUTS")

print("=== _finish_return routes to PAUSED, resume leaves it ===")
seq._finish_return("home")
check("RETURN_HOME completion parks in PAUSED", wm.state == "PAUSED", wm.state)
check("still flagged paused while parked", wm._paused is True)
delib.on_resume()
check("resume clears paused + arms resuming", wm._paused is False and wm._resuming is True)
check("resume enters RESUME_NAV", wm.state == "RESUME_NAV", wm.state)
check("resume rebases the bag", wm._bag_start_count == wm._collected_total)

print("=== RESUME_NAV waypoint geometry (south buffer, then the anchor) ===")
wm, io, skills, seq, delib, drv = make_stack()
wm._paused = True
wm._pause_row = 0                        # paused on row 1; anchor at lateral 1.40
wm._pause_side = "right"
wm._pause_anchor = (0.0, 1.40)
delib.on_resume()                        # -> RESUME_NAV, phase HEADLAND
drv["returns"] = True                    # each leg 'arrives' so phases advance
seq._tick_resume_nav(); w1 = skills._drive_targets[-1]
seq._tick_resume_nav(); w2 = skills._drive_targets[-1]
seq._tick_resume_nav(); w3 = skills._drive_targets[-1]
check("W1 is in the south buffer (along < 0)", along(wm, w1[0], w1[1]) < 0,
      f"along={along(wm, w1[0], w1[1]):.2f}")
check("W1 at home's lateral, margin south", abs(w1[0] + 0.40) < 1e-9 and abs(w1[1]) < 1e-9,
      f"W1={w1[:2]}")
check("W2 still in buffer, at the row's lateral",
      abs(w2[0] + 0.40) < 1e-9 and abs(w2[1] - 1.40) < 1e-9, f"W2={w2[:2]}")
check("W3 == the row start anchor", abs(w3[0]) < 1e-9 and abs(w3[1] - 1.40) < 1e-9, f"W3={w3[:2]}")
check("legs route with avoid_state=RESUME_NAV", all(t[2] == "RESUME_NAV" for t in skills._drive_targets))
# ALIGN_OUT: heading already at outbound -> restore row and enter the crab
seq._tick_resume_nav()
check("resume restores the paused row", wm.rows_completed == 0 and wm.current_side == "right",
      f"rows={wm.rows_completed} side={wm.current_side}")
check("resume hands to the crab (LATERAL_ALIGN)", wm.state == "LATERAL_ALIGN", wm.state)
check("crab will hand on to FOLLOW_OUT", wm._strafe_next_state == "FOLLOW_OUT")
check("detection still off through re-align", wm._resuming is True)

print("=== detection re-enable gate (no duplicates) ===")
wm, io, skills, seq, delib, drv = make_stack()
wm._resuming = True
wm._pause_along = 0.5
wm.state = "FOLLOW_OUT"
wm.bag = 0
io.perception = []
wm.odom_x, wm.odom_y = 0.30, 0.0          # before the pause spot
delib.bag_and_resume_tick()
check("detection stays OFF before the pause spot", wm._resuming is True and io.perception == [],
      f"perc={io.perception}")
wm.odom_x = 0.60                          # past the pause spot
delib.bag_and_resume_tick()
check("detection re-enabled at the pause spot", io.perception and io.perception[-1] is True)
check("gate disarms after re-enable", wm._resuming is False)

print("=== bag: auto-pause every `bag` new nuts ===")
wm, io, skills, seq, delib, drv = make_stack()
wm.bag = 3
wm._bag_start_count = 0
wm._collected_total = 2
wm.state = "FOLLOW_OUT"
delib.bag_and_resume_tick()
check("no pause below the bag (2 < 3)", wm._paused is False)
wm._collected_total = 3
delib.bag_and_resume_tick()
check("auto-pause when the bag fills (3 >= 3)", wm._paused is True and wm.state == "RETURN_HOME")
delib.on_resume()
check("resume rebased bag_start to current total", wm._bag_start_count == 3)
wm.state = "FOLLOW_OUT"; wm._paused = False
wm._collected_total = 5                   # 2 new since resume
delib.bag_and_resume_tick()
check("no pause after 2 new (5-3 < 3)", wm._paused is False)
wm._collected_total = 6                   # 3 new since resume
delib.bag_and_resume_tick()
check("auto-pause again after 3 new (6-3 >= 3)", wm._paused is True)

print("=== missed-nut collection: map nuts to rows, re-sweep each ===")
wm, io, skills, seq, delib, drv = make_stack()
wm.outbound_yaw = 0.0                     # d=(1,0); right-side hug -> sweeper_unit=(0,-1)
wm.start_side = "right"
wm.desired_side_distance = 0.40
wm._row_anchors = [(0.0, 0.0, "right"), (0.0, -0.70, "right"), (0.0, -1.40, "right")]
io.m2o = (0.0, 0.0, 1.0, 0.0)            # identity map<->odom
wm.uncollected_map = [(1.0, -0.40), (1.2, -1.80)]
rows = delib._rows_with_missed_nuts()
check("missed nuts map to their rows (0 and 2, not 1)", rows == [0, 2], rows)

started = delib._begin_recollect("Sweep done.")
check("recollect starts when rows have missed nuts", started is True)
check("recollecting flag set", wm._recollecting is True)
check("first re-sweep targets row 1 (index 0) via RESUME_NAV",
      wm.state == "RESUME_NAV" and wm._pause_row == 0, f"state={wm.state} row={wm._pause_row}")
check("navigates to that row's anchor", wm._pause_anchor == (0.0, 0.0), wm._pause_anchor)
check("re-sweep keeps detection off (not the resume gate)", wm._resuming is False)
check("queue holds the remaining row", wm._recollect_rows == [2], wm._recollect_rows)

delib._recollect_next("Re-swept a row.")
check("second re-sweep targets row 3 (index 2)",
      wm.state == "RESUME_NAV" and wm._pause_row == 2, f"state={wm.state} row={wm._pause_row}")
check("anchor for row 3", wm._pause_anchor == (0.0, -1.40), wm._pause_anchor)

delib._recollect_next("Re-swept a row.")
check("recollect finishes -> RETURN_HOME", wm.state == "RETURN_HOME", wm.state)
check("recollecting flag cleared", wm._recollecting is False)

wm2, io2, skills2, seq2, delib2, _ = make_stack()
wm2.outbound_yaw = 0.0
wm2._row_anchors = [(0.0, 0.0, "right")]
io2.m2o = (0.0, 0.0, 1.0, 0.0)
wm2.uncollected_map = []
check("no re-sweep when nothing is missed", delib2._begin_recollect("done") is False)

print("=== battery safety: pause + home only when enabled and held low ===")
wm, io, skills, seq, delib, drv = make_stack()
wm.state = "FOLLOW_OUT"
wm.battery_safety = False
wm._battery_voltage = 9.0
delib.battery_safety_check()
check("disabled: low voltage does NOT pause", wm._paused is False and wm.state == "FOLLOW_OUT")

wm, io, skills, seq, delib, drv = make_stack()
wm.state = "FOLLOW_OUT"
wm.battery_safety = True
wm.battery_min_voltage = 10.0
wm._battery_voltage = 11.5
delib.battery_safety_check()
check("healthy voltage does not arm the low timer",
      wm._paused is False and wm._battery_low_since is None)

wm._battery_voltage = 9.5
delib.battery_safety_check()
check("low voltage arms the timer but waits (debounce)",
      wm._paused is False and wm._battery_low_since is not None)

wm._battery_low_since = _T(-6 * 10**9)      # 6 s ago (> 5 s duration)
delib.battery_safety_check()
check("held low past duration pauses + returns home",
      wm._paused is True and wm.state == "RETURN_HOME")

wm2, io2, skills2, seq2, delib2, _ = make_stack()
wm2.state = "FOLLOW_OUT"
wm2.battery_safety = True
wm2.battery_min_voltage = 10.0
wm2._battery_voltage = 9.0
delib2.battery_safety_check()                  # arm
wm2._battery_voltage = 11.0
delib2.battery_safety_check()                  # recover
check("voltage recovery clears the low timer", wm2._battery_low_since is None
      and wm2._paused is False)

wm3, io3, skills3, seq3, delib3, _ = make_stack()
wm3.state = "RETURN_HOME"
wm3.battery_safety = True
wm3._battery_voltage = 9.0
wm3._battery_low_since = _T(-9 * 10**9)
delib3.battery_safety_check()
check("not in a sweep state: no battery pause", wm3._paused is False
      and wm3._battery_low_since is None)

print()
print("=" * 60)
print(f"PAUSE/RESUME TEST: {PASS}/{PASS + FAIL} checks passed")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
