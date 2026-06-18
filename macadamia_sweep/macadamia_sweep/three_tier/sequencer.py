#!/usr/bin/env python3
"""Sequencer: the executive tier.

Runs once per 10 Hz control cycle. It keeps exactly one reactive skill active
(``world_model.state``), feeds it, reads the skill's completion event plus the
perception guards, and performs the state transitions that choreograph a row
sweep. At mission forks (end of a return pass, end of collection) it defers to
the Deliberator. It also exposes ``enter_avoid_front`` -- the one reactive
interrupt a skill may trigger -- and the small private transition helpers
(``_enter_strafe`` / ``_enter_follow_back`` / ``_finish_strafe`` /
``_begin_next_row``) ported verbatim from the original.

``tick()`` reproduces the original ``control_loop`` byte-for-byte: same preamble
(idle/scan guards, lazy heading anchor, lazy home, swept-bounds recording) and
the same per-state dispatch and transition conditions.
"""

import math

from .util import normalize_angle, opposite, Skill, Event


class Sequencer:
    def __init__(self, wm, io, perc, skills):
        self.wm = wm
        self.io = io
        self.perc = perc
        self.skills = skills
        self.deliberator = None   # wired by the agent (circular dependency)

    # ---- timing helpers (verbatim semantics) ----------------------------
    def elapsed_in_state(self) -> float:
        return self.io.elapsed_since(self.wm.state_start_time)

    def time_since_row_seen(self) -> float:
        return self.io.elapsed_since(self.wm.last_row_seen_time)

    # ---- core transition primitives -------------------------------------
    def set_state(self, new_state: str, status: str = ""):
        self.wm.state = new_state
        self.wm.state_start_time = self.io.now()
        if status:
            self.io.publish_status(status)
        self.io.log_info(f"State changed to {new_state}")

    def enter_avoid_front(self, reason: str = ""):
        """Reactive front-obstacle interrupt. A skill calls this exactly where
        the original called enter_avoid_front; the from-state is always the
        currently-active skill, so we resume it when avoidance completes."""
        wm = self.wm
        wm.avoid_previous_state = wm.state
        wm.avoid_phase = "BACKUP"
        wm.avoid_phase_start_time = self.io.now()
        self.io.stop_robot()
        self.set_state(
            Skill.AVOID_FRONT,
            f"Front obstacle detected. Avoiding before continuing "
            f"{wm.avoid_previous_state}. {reason}"
        )

    def _enter_follow_back(self, status: str):
        wm = self.wm
        wm.last_row_seen_time = self.io.now()
        wm.seen_row_this_pass = False
        wm.passed_forward_tree = False
        wm.next_row_hits = 0  # fresh count for THIS return pass
        self.set_state(Skill.FOLLOW_BACK, status)

    def _enter_strafe(self, next_state: str, status: str):
        wm = self.wm
        wm._strafe_next_state = next_state
        wm._lat_phase = "MEASURE"
        wm._strafe_start = None
        self.set_state(Skill.LATERAL_ALIGN, status)

    def _finish_strafe(self, why: str):
        wm = self.wm
        self.io.stop_robot()
        if wm._strafe_next_state == "FOLLOW_BACK":
            self._enter_follow_back(f"{why}. Sweeping back.")
        else:
            self.set_state(Skill.FOLLOW_OUT, f"{why}. Sweeping row.")

    def _begin_next_row(self, status: str):
        """Reset per-row state and start the outbound pass on the next row."""
        wm = self.wm
        wm.next_row_hits = 0
        wm.seen_row_this_pass = False
        wm.passed_forward_tree = False
        wm.last_row_seen_time = self.io.now()
        wm.clear_start_x = None
        wm.clear_start_y = None
        wm.arc_start_yaw = None
        wm.arc_last_yaw = None
        wm.arc_accumulated_yaw = 0.0
        # current_side is UNCHANGED. Strafe to exactly 0.40 m before sweeping.
        self._enter_strafe(Skill.FOLLOW_OUT, status + " Strafing to 0.40 m.")

    def _should_end_pass(self) -> bool:
        wm = self.wm
        elapsed = self.elapsed_in_state()
        if elapsed < wm.min_pass_duration:
            return False
        if wm.seen_row_this_pass and self.time_since_row_seen() >= wm.row_lost_timeout:
            return True
        if elapsed >= wm.max_pass_duration:
            return True
        return False

    # -----------------------------
    # Main 10 Hz control cycle
    # -----------------------------
    def tick(self):
        wm = self.wm
        if not wm.started:
            self.io.stop_robot()
            return
        if wm.latest_scan is None:
            self.io.stop_robot()
            self.io.publish_status("Waiting for /scan")
            return

        # Lazy heading anchor: set on first odom tick if not ready at sweep_start.
        if wm.outbound_yaw is None and wm.odom_yaw is not None:
            wm.outbound_yaw = wm.odom_yaw
            wm.return_target_yaw = normalize_angle(wm.odom_yaw + math.pi)
            self.io.log_info(
                f"Lazy anchor: outbound={math.degrees(wm.outbound_yaw):+.1f}deg "
                f"return_target={math.degrees(wm.return_target_yaw):+.1f}deg"
            )

        if (wm.home_x is None or wm.home_y is None) and wm.odom_x is not None and wm.odom_y is not None:
            wm.home_x = wm.odom_x
            wm.home_y = wm.odom_y
            wm.home_yaw = wm.odom_yaw
            self.io.log_info(
                f"Lazy home saved: x={wm.home_x:+.2f}, y={wm.home_y:+.2f}, "
                f"yaw={math.degrees(wm.home_yaw or 0.0):+.1f}deg"
            )

        # Record swept free-space while actively sweeping.
        if wm.state in wm._SWEEP_STATES:
            self.skills.update_swept_bounds()

        if wm.state == Skill.FOLLOW_OUT:
            self._tick_follow_out()
        elif wm.state == Skill.CLEAR_END:
            self._tick_clear(Skill.ARC_TURN, "CLEAR_END")
        elif wm.state == Skill.ARC_TURN:
            self._tick_arc()
        elif wm.state == Skill.ALIGN:
            self._tick_align()
        elif wm.state == Skill.LATERAL_ALIGN:
            self._tick_lateral_align()
        elif wm.state == Skill.FOLLOW_BACK:
            self._tick_follow_back()
        elif wm.state == Skill.CLEAR_NEXT:
            self._tick_clear(Skill.TURN_NEXT, "CLEAR_NEXT")
        elif wm.state == Skill.TURN_NEXT:
            self._tick_turn_next()
        elif wm.state == Skill.AVOID_FRONT:
            self._tick_avoid_front()
        elif wm.state == Skill.COLLECT_NUTS:
            self._tick_collect_nuts()
        elif wm.state == Skill.RETURN_HOME:
            self._tick_return_home()
        elif wm.state == Skill.DONE:
            self.io.stop_robot()
            wm.started = False
            self.io.publish_status("Demo complete. Robot stopped.")
            return
        elif wm.state == Skill.STOPPED:
            self.io.stop_robot()
            return
        else:
            self.io.stop_robot()
            self.io.publish_status(f"Unknown state: {wm.state}")

    # -----------------------------
    # Per-state handlers
    # -----------------------------
    def _tick_follow_out(self):
        wm = self.wm
        self.skills.follow_side(wm.current_side)
        if wm.state != Skill.FOLLOW_OUT:
            # follow_side handed off to AVOID_FRONT; let it run instead of
            # overriding it with an end-of-row transition this tick.
            return
        abeam = self.perc.last_tree_abeam_or_behind(wm.current_side)
        _pts = self.perc.side_cone_points(wm.current_side)
        _fwd = max((p[0] for p in _pts), default=float("nan"))
        tree_ahead = bool(_pts) and _fwd > wm.forward_tree_arm
        hard_timeout = self.elapsed_in_state() >= wm.max_pass_duration
        lost = hard_timeout or (self._should_end_pass()
                                and wm.passed_forward_tree and not tree_ahead)
        if abeam or lost:
            trigger = "last tree abeam" if abeam else "row lost"
            self.io.log_warn(
                f"FOLLOW_OUT end: {trigger} | trees_seen={len(_pts)} "
                f"fwd_tree_x={_fwd:+.2f}m elapsed={self.elapsed_in_state():.1f}s"
            )
            wm.clear_start_x = None
            wm.clear_start_y = None
            wm.arc_start_yaw = None
            wm.arc_last_yaw = None
            wm.arc_accumulated_yaw = 0.0
            wm.seen_row_this_pass = False
            wm.passed_forward_tree = False
            tgt = (math.degrees(wm.return_target_yaw)
                   if wm.return_target_yaw is not None else float("nan"))
            self.set_state(
                Skill.CLEAR_END,
                f"End of row ({trigger}). Return heading={tgt:.0f}deg. "
                f"Driving past last tree."
            )
            return

    def _tick_clear(self, next_state: str, label: str):
        ev, status = self.skills.clear_straight(next_state, label)
        if ev == Event.DONE:
            self.set_state(next_state, status)

    def _tick_arc(self):
        ev, status = self.skills.arc_turn()
        if ev == Event.ARC_DONE:
            self.set_state(Skill.ALIGN, status)

    def _tick_align(self):
        ev, status = self.skills.align_to_row()
        if ev == Event.ALIGNED:
            self._enter_strafe(Skill.FOLLOW_BACK, status)
        elif ev == Event.ALIGN_TIMEOUT:
            self._enter_strafe(Skill.FOLLOW_BACK, status)
        elif ev == Event.ALIGN_NO_REF:
            self._enter_follow_back(status)

    def _tick_lateral_align(self):
        wm = self.wm
        ev, status = self.skills.lateral_align()
        if wm.state != Skill.LATERAL_ALIGN:
            # lateral_align handed off to AVOID_FRONT.
            return
        if ev == Event.DONE:
            self._finish_strafe(status)

    def _tick_follow_back(self):
        wm = self.wm
        self.skills.follow_side(wm.current_side)
        if wm.state != Skill.FOLLOW_BACK:
            # follow_side handed off to AVOID_FRONT.
            return

        # While sweeping back, count confident opposite-side fits (the next row).
        opp = opposite(wm.current_side)
        opp_visible, opp_fit = self.perc.row_visible(opp)
        if opp_visible and abs(opp_fit[1]) <= wm.next_row_max_dist:
            wm.next_row_hits += 1

        # End the return by POSITION (driven back to the outbound start longitude).
        along = None
        if (wm.home_x is not None and wm.odom_x is not None
                and wm.outbound_yaw is not None):
            along = ((wm.odom_x - wm.home_x) * math.cos(wm.outbound_yaw)
                     + (wm.odom_y - wm.home_y) * math.sin(wm.outbound_yaw))
            end_pass = (along <= wm.return_end_margin
                        or self.elapsed_in_state() >= wm.max_pass_duration)
        else:
            end_pass = self._should_end_pass()
        if end_pass:
            _why = (f"reached home (along={along:+.2f}m)"
                    if along is not None and along <= wm.return_end_margin
                    else f"timeout/lost (along={along}, "
                         f"elapsed={self.elapsed_in_state():.1f}s)")
            self.io.log_warn(f"FOLLOW_BACK end: {_why}")
            self.deliberator.after_row_pass(opp)
            return

    def _tick_turn_next(self):
        ev, status = self.skills.turn_to_next_row()
        if ev in (Event.TURNED, Event.TURN_TIMEOUT, Event.TURN_NO_REF):
            self._begin_next_row(status)

    def _tick_avoid_front(self):
        wm = self.wm
        ev, status = self.skills.avoid_front_obstacle()
        if ev == Event.DONE:
            # Resume the interrupted state. RETURN_HOME recalculates its heading
            # on the next tick.
            wm.last_row_seen_time = self.io.now()
            wm.seen_row_this_pass = False
            wm.passed_forward_tree = False
            self.set_state(wm.avoid_previous_state, status)

    def _tick_collect_nuts(self):
        wm = self.wm
        ev, payload = self.skills.collect_nuts()
        if wm.state != Skill.COLLECT_NUTS:
            # collect drive handed off to AVOID_FRONT mid-tick.
            return
        if ev == Event.COLLECT_GO_HOME:
            self.deliberator._go_home(payload)

    def _tick_return_home(self):
        wm = self.wm
        ev, status = self.skills.return_home()
        if wm.state != Skill.RETURN_HOME:
            # return_home handed off to AVOID_FRONT.
            return
        if ev == Event.HOME_REACHED:
            self.set_state(Skill.DONE, status)
