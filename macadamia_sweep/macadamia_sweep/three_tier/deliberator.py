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
goal-state. All numeric/logic content is a verbatim port of the original
callbacks and the ``finish_or_return_home`` / ``_go_home`` helpers.
"""

import math

from .util import normalize_angle, Skill


class Deliberator:
    def __init__(self, wm, io, seq):
        self.wm = wm
        self.io = io
        self.seq = seq

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
        collect nuts, or go home. Verbatim port of the FOLLOW_BACK-end block."""
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
        """Collect any uncollected nuts first (if enabled), then return home."""
        wm = self.wm
        self.io.stop_robot()
        # Sweep over: freeze the nut world model AND the tree map for the rest of
        # the mission by pausing perception.
        self.io.set_perception(False)
        if wm.collect_before_home and wm.uncollected_map:
            wm._collect_skip = set()
            wm._collect_target = None
            wm._collect_phase = "TO_HEADLAND"
            wm._collect_deadline = None
            wm._collect_avoid_count = 0
            wm._collect_consec_skips = 0
            self.seq.set_state(
                Skill.COLLECT_NUTS,
                reason + f" Collecting {len(wm.uncollected_map)} uncollected nut(s) "
                f"before home."
            )
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
