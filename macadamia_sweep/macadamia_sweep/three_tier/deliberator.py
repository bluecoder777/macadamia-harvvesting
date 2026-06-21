#!/usr/bin/env python3
"""Deliberator: the planning / mission tier.

Owns the long-horizon plan and every decision that is about the *mission*
rather than the immediate control loop:

  * reacting to the operator triggers (/sweep_start, /sweep_stop, /return_home),
  * deciding, after each row's return pass, whether to sweep another row
    (fixed-count vs. auto-from-perception), collect missed nuts, or go home,
  * choosing between nut collection and return-home, and between returning home
    and finishing.

It writes its decisions into the world model and hands the sequencer the next
goal-state.
"""

import math

from .util import normalize_angle, Skill


class Deliberator:
    def __init__(self, wm, io, seq):
        self.wm = wm
        self.io = io
        self.seq = seq
        self.skills = None   # wired by the agent (used for row geometry in re-sweep)

    # -----------------------------
    # Operator triggers
    # -----------------------------
    def on_sweep_start(self):
        wm = self.wm
        # GUARD: repeated /sweep_start publishes must not re-snapshot the heading
        # anchor mid-run or reset the machine.
        if wm.started:
            self.io.log_warn(
                "sweep_start ignored - sweep already running. "
                "Publish /sweep_stop first to restart."
            )
            return
        wm.started = True
        wm.rows_completed = 0
        wm.next_row_hits = 0
        wm.state = Skill.FOLLOW_OUT
        wm.state_start_time = self.io.now()
        wm.last_row_seen_time = self.io.now()
        wm.seen_row_this_pass = False
        wm.passed_forward_tree = False
        wm.current_side = wm.start_side
        wm.clear_start_x = None
        wm.clear_start_y = None
        wm.arc_start_yaw = None
        wm.arc_last_yaw = None
        wm.arc_accumulated_yaw = 0.0
        self.io.set_perception(True)
        # Fresh run: discard the previous sweep's free-space record.
        wm._swept_path = []
        wm._swept_long_min = wm._swept_long_max = None
        wm._swept_lat_min = wm._swept_lat_max = None
        # Fresh run: clear any pause/resume/recollect state and start a new bag.
        wm._paused = False
        wm._resuming = False
        wm._recollecting = False
        wm._recollect_rows = []
        wm._return_end_state = "DONE"
        wm._bag_start_count = wm._collected_total
        wm._row_start_anchor = (
            (wm.odom_x, wm.odom_y) if wm.odom_x is not None else None)
        # Row 0 starts here (home). Lazily filled by the sequencer if odom is late.
        wm._row_anchors = (
            [(wm.odom_x, wm.odom_y, wm.current_side)]
            if wm.odom_x is not None else [])

        # Save the starting pose for RETURN_HOME.
        if wm.odom_x is not None and wm.odom_y is not None:
            wm.home_x = wm.odom_x
            wm.home_y = wm.odom_y
            wm.home_yaw = wm.odom_yaw
            self.io.log_info(
                f"Home pose saved: x={wm.home_x:+.2f}, y={wm.home_y:+.2f}, "
                f"yaw={math.degrees(wm.home_yaw or 0.0):+.1f}deg"
            )
        else:
            wm.home_x = None
            wm.home_y = None
            wm.home_yaw = None
            self.io.log_warn(
                "Odom not ready at sweep_start - home pose will be saved on first odom tick."
            )

        # Snapshot the row direction NOW (robot is parallel to the row).
        if wm.odom_yaw is not None:
            wm.outbound_yaw = wm.odom_yaw
            wm.return_target_yaw = normalize_angle(wm.odom_yaw + math.pi)
            self.io.log_info(
                f"Anchored row heading: outbound={math.degrees(wm.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(wm.return_target_yaw):+.1f}deg"
            )
        else:
            wm.outbound_yaw = None
            wm.return_target_yaw = None
            self.io.log_warn(
                "Odom not ready at sweep_start - heading anchor will be set on first odom tick."
            )

        self.io.publish_status(f"Started: line-fit following on {wm.current_side.upper()}")
        self.io.log_info(f"Sweep start received (side={wm.current_side})")

    def on_sweep_stop(self):
        wm = self.wm
        wm.started = False
        wm.state = Skill.STOPPED
        self.io.stop_robot()
        self.io.publish_status("Manual stop received")
        self.io.log_warn("Sweep stop received")

    def on_return_home(self):
        """Manual trigger: abandon the current sweep and drive home."""
        wm = self.wm
        if wm.home_x is None or wm.home_y is None:
            self.io.log_warn(
                "return_home requested, but no home pose has been saved. "
                "Publish /sweep_start first so the robot knows where home is."
            )
            self.io.publish_status(
                "Cannot RETURN_HOME: no saved home pose. Publish /sweep_start first.")
            return

        wm.started = True
        wm.next_row_hits = 0
        wm.clear_start_x = None
        wm.clear_start_y = None
        wm.arc_start_yaw = None
        wm.arc_last_yaw = None
        wm.arc_accumulated_yaw = 0.0
        self.io.stop_robot()
        self.io.set_perception(False)
        wm._home_phase = "OUT"
        self.seq.set_state(
            Skill.RETURN_HOME,
            f"Manual return-home requested. Exiting the field, then returning to "
            f"x={wm.home_x:+.2f}, y={wm.home_y:+.2f}."
        )
        self.io.log_warn("Manual return-home requested")

    # -----------------------------
    # End-of-row-pass mission decision
    # -----------------------------
    def after_row_pass(self, opp: str):
        """Decide what happens after a finished return pass: sweep the next row,
        collect nuts, or go home."""
        wm = self.wm
        wm.rows_completed += 1
        next_row_seen = wm.next_row_hits >= wm.next_row_min_hits

        if wm.max_rows > 0:
            # FIXED COUNT: proceed until exactly max_rows are done.
            proceed = wm.rows_completed < wm.max_rows
            done_reason = f"reached requested {wm.max_rows} rows"
        else:
            # AUTO: only continue while a next row is actually seen.
            proceed = next_row_seen
            done_reason = (f"no next row ({wm.next_row_hits} hits "
                           f"< {wm.next_row_min_hits})")

        if proceed:
            wm.clear_start_x = None
            wm.clear_start_y = None
            seen_str = (f"next row on {opp.upper()} "
                        f"({wm.next_row_hits} hits)" if next_row_seen
                        else "next row not yet confirmed by lidar")
            self.seq.set_state(
                Skill.CLEAR_NEXT,
                f"Row {wm.rows_completed} done; {seen_str}. "
                f"Clearing row start."
            )
        else:
            self.finish_or_return_home(
                f"Row {wm.rows_completed} done, {done_reason}."
            )

    def finish_or_return_home(self, reason: str):
        """Re-sweep any row that still has missed nuts (same path as resume),
        then return home."""
        wm = self.wm
        self.io.stop_robot()
        # Sweep over: freeze the nut world model AND the tree map for the rest of
        # the mission (re-sweep + return home) by pausing perception. The
        # drive-over collection in nut_tracker still picks up the missed nuts.
        self.io.set_perception(False)
        if (wm.collect_before_home and wm.uncollected_map
                and not wm._recollecting):
            if self._begin_recollect(reason):
                return
        self._go_home(reason)

    def _go_home(self, reason: str):
        """Drive to the saved start pose if return-home is enabled, else finish."""
        wm = self.wm
        self.io.stop_robot()
        if wm.return_home_enabled and wm.home_x is not None and wm.home_y is not None:
            wm._home_phase = "OUT"
            self.seq.set_state(
                Skill.RETURN_HOME,
                reason + f" Exiting the field, then returning to start "
                f"x={wm.home_x:+.2f}, y={wm.home_y:+.2f}."
            )
        else:
            self.seq.set_state(Skill.DONE, reason + " Mission complete.")

    # -----------------------------
    # Pause / resume (operator + bag + battery)
    # -----------------------------
    def on_pause(self):
        """Manual pause: park at home and wait for /resume_collection."""
        self._begin_pause("manual pause")

    def on_resume(self):
        """Resume from a pause: skip back to the paused row and re-sweep it."""
        wm = self.wm
        if not wm._paused:
            self.io.log_warn("resume_collection ignored - not paused.")
            return
        wm._paused = False
        wm._resuming = True
        wm._return_end_state = "DONE"
        wm._bag_start_count = wm._collected_total   # next bag counts new nuts
        wm._resume_phase = "HEADLAND"
        self.seq.set_state(
            Skill.RESUME_NAV,
            f"Resuming: navigating back to row {wm._pause_row + 1}."
        )
        self.io.log_warn(f"RESUME: heading back to row {wm._pause_row + 1}")

    def battery_safety_check(self):
        """If enabled, pause (return home) when the pack voltage stays below the
        threshold for battery_low_duration while sweeping. Reset otherwise."""
        wm = self.wm
        if not wm.battery_safety or wm._battery_voltage is None:
            return
        if wm._paused or wm.state not in wm._SWEEP_STATES:
            wm._battery_low_since = None
            return
        if wm._battery_voltage >= wm.battery_min_voltage:
            wm._battery_low_since = None          # recovered (or just a sag)
            return
        if wm._battery_low_since is None:
            wm._battery_low_since = self.io.now()
        held = (self.io.now() - wm._battery_low_since).nanoseconds / 1e9
        if held >= wm.battery_low_duration:
            self.io.log_error(
                f"BATTERY LOW: {wm._battery_voltage:.2f}V < "
                f"{wm.battery_min_voltage:.2f}V for {held:.0f}s - "
                f"pausing and returning home."
            )
            self._begin_pause(f"battery low ({wm._battery_voltage:.2f}V)")

    def _begin_pause(self, reason: str):
        """Snapshot the row being swept, freeze perception, and head home to wait.
        Shared by the manual pause and the bag auto-pause."""
        wm = self.wm
        if not wm.started or wm._paused:
            return
        if wm.state not in wm._SWEEP_STATES:
            self.io.log_warn(
                f"Pause ignored ({reason}): not actively sweeping (state={wm.state})."
            )
            return
        # Remember which row + where on it, so resume can skip straight back.
        wm._pause_row = wm.rows_completed
        wm._pause_side = wm.current_side
        if (wm.outbound_yaw is not None and wm.home_x is not None
                and wm.odom_x is not None):
            wm._pause_along = (
                (wm.odom_x - wm.home_x) * math.cos(wm.outbound_yaw)
                + (wm.odom_y - wm.home_y) * math.sin(wm.outbound_yaw))
        else:
            wm._pause_along = 0.0
        wm._pause_anchor = wm._row_start_anchor or (
            (wm.home_x, wm.home_y) if wm.home_x is not None else None)
        wm._paused = True
        wm._resuming = False
        self.io.set_perception(False)         # no detection while paused / heading home
        self.io.stop_robot()
        if wm.return_home_enabled and wm.home_x is not None and wm.home_y is not None:
            wm._return_end_state = "PAUSED"
            wm._home_phase = "OUT"
            wm.next_row_hits = 0
            wm.clear_start_x = None
            wm.clear_start_y = None
            wm.arc_start_yaw = None
            wm.arc_last_yaw = None
            wm.arc_accumulated_yaw = 0.0
            self.seq.set_state(
                Skill.RETURN_HOME,
                f"Paused ({reason}) on row {wm._pause_row + 1}. Returning home; "
                f"waiting for /resume_collection."
            )
        else:
            self.seq.set_state(
                Skill.PAUSED, f"Paused ({reason}); no home pose - holding in place."
            )
        self.io.log_warn(f"PAUSE: {reason} on row {wm._pause_row + 1}")

    def bag_and_resume_tick(self):
        """Per-tick while actively sweeping: the bag auto-pause and, after a
        resume, the detection re-enable gate. Called from the sequencer."""
        wm = self.wm
        # Bag: park at home once `bag` NEW nuts have been collected this bag.
        # Suppressed during a re-sweep (end-of-mission cleanup).
        if (wm.bag > 0 and not wm._paused and not wm._recollecting
                and wm._collected_total - wm._bag_start_count >= wm.bag):
            self._begin_pause(f"bag full ({wm.bag})")
            return
        # Resume: hold detection off over the already-swept part of the row, then
        # re-enable the moment we drive back up to where we paused (no duplicates).
        if (wm._resuming and wm.outbound_yaw is not None
                and wm.home_x is not None and wm.odom_x is not None):
            along = (
                (wm.odom_x - wm.home_x) * math.cos(wm.outbound_yaw)
                + (wm.odom_y - wm.home_y) * math.sin(wm.outbound_yaw))
            if along >= wm._pause_along:
                self.io.set_perception(True)
                wm._resuming = False
                self.io.log_info(
                    f"Resumed detection at the pause area (along={along:+.2f}m)."
                )

    # -----------------------------
    # Missed-nut collection by RE-SWEEPING the row (same path as resume)
    # -----------------------------
    def _rows_with_missed_nuts(self):
        """Indices of forward-swept rows that still hold uncollected nuts."""
        wm = self.wm
        tf = self.io.map_to_odom()
        units = self.skills._row_units()
        if tf is None or units is None or not wm._row_anchors:
            return []
        (_dx, _dy), (sx, sy) = units
        # Expected nut lateral for each row = anchor lateral + hug offset.
        exp = [ax * sx + ay * sy + wm.desired_side_distance
               for (ax, ay, _s) in wm._row_anchors]
        rows = set()
        for (mx, my) in wm.uncollected_map:
            ox, oy = self.skills._apply_tf(mx, my, tf)
            nut_lat = ox * sx + oy * sy
            best_i = min(range(len(exp)), key=lambda i: abs(nut_lat - exp[i]))
            rows.add(best_i)
        return sorted(rows)

    def _begin_recollect(self, reason: str) -> bool:
        """If any swept row still has missed nuts, queue those rows for a re-sweep
        and start. Returns True if a re-sweep was started."""
        wm = self.wm
        rows = self._rows_with_missed_nuts()
        if not rows:
            return False
        wm._recollect_rows = rows
        wm._recollecting = True
        self._recollect_next(
            reason + f" {len(wm.uncollected_map)} nut(s) missed; re-sweeping "
            f"row(s) {[r + 1 for r in rows]}."
        )
        return True

    def _recollect_next(self, reason: str = ""):
        """Re-sweep the next queued row (navigate to its start via RESUME_NAV),
        or go home when the queue is empty."""
        wm = self.wm
        while wm._recollect_rows:
            row = wm._recollect_rows.pop(0)
            if row < len(wm._row_anchors) and wm._row_anchors[row] is not None:
                ax, ay, side = wm._row_anchors[row]
                wm._pause_row = row
                wm._pause_side = side
                wm._pause_anchor = (ax, ay)
                wm._resuming = False          # recollect: perception stays off
                wm._resume_phase = "HEADLAND"
                self.seq.set_state(
                    Skill.RESUME_NAV,
                    (reason + " " if reason else "")
                    + f"Re-sweeping row {row + 1} for missed nuts."
                )
                return
            self.io.log_warn(f"Recollect: no anchor for row {row + 1}, skipping.")
        wm._recollecting = False
        self._go_home(reason or "Re-swept all rows with missed nuts.")
