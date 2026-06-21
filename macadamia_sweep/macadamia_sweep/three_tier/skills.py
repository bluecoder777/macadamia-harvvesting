#!/usr/bin/env python3
"""Skill layer: the acting half of the reactive/controller tier.

Each public skill is a tight sensor->actuator feedback loop. When the
sequencer has it active, the skill reads the world model + perception, publishes
exactly one velocity command (and its own running status), and returns an
``(event, status)`` outcome telling the sequencer whether it is still RUNNING or
has completed/aborted. Skills NEVER decide what runs next -- with a single
deliberate exception: the front-avoidance safety reflex, which a skill triggers
through the sequencer-provided ``enter_avoid`` callback.

Each skill reads state/params from ``self.wm`` (world model), actuates and reads
the clock through ``self.io``, and queries ``self.perc`` (perception); it never
sets the executive state itself, signalling completion via the outcome return.
"""

import math
from typing import List, Optional, Tuple

from .util import normalize_angle, Event


RUNNING = (Event.RUNNING, "")


class Skills:
    def __init__(self, wm, io, perc):
        self.wm = wm
        self.io = io
        self.perc = perc
        # Wired by the agent after the sequencer exists. Signature: (reason:str).
        self.enter_avoid = None

    # ---- small timing/percept helpers -----------------------------------
    def _elapsed_in_state(self) -> float:
        return self.io.elapsed_since(self.wm.state_start_time)

    def _time_since_row_seen(self) -> float:
        return self.io.elapsed_since(self.wm.last_row_seen_time)

    def _elapsed_in_avoid_phase(self) -> float:
        return self.io.elapsed_since(self.wm.avoid_phase_start_time)

    def mark_row_seen_if_visible(self, visible: bool):
        if visible:
            self.wm.last_row_seen_time = self.io.now()
            self.wm.seen_row_this_pass = True

    # -----------------------------
    # Line-fit follow  (FOLLOW_OUT / FOLLOW_BACK)
    # -----------------------------
    def follow_side(self, side: str):
        wm = self.wm
        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            self.io.publish_status(f"EMERGENCY STOP: front={front:.2f} m")
            return RUNNING

        if front < wm.avoid_front_distance:
            self.enter_avoid(f"front={front:.2f}m")
            return RUNNING

        visible, fit = self.perc.row_visible(side)
        self.mark_row_seen_if_visible(visible)

        if self._elapsed_in_state() < wm.start_straight_duration:
            self.io.publish_cmd(wm.forward_speed, 0.0)
            self.io.publish_status(
                f"START STRAIGHT | row_visible={visible} | front={front:.2f}m"
            )
            return RUNNING

        if not visible:
            self.io.publish_cmd(wm.search_speed, 0.0)
            self.io.publish_status(
                f"ROW NOT CONFIRMED {side.upper()} | creeping forward | "
                f"front={front:.2f}m"
            )
            return RUNNING

        line_angle, perp, n = fit  # type: ignore[assignment]
        wm._last_row_perp = abs(perp)

        desired_perp = (-wm.desired_side_distance
                        if side == "right" else wm.desired_side_distance)
        move_left = perp - desired_perp
        heading_error = line_angle

        angular = wm.k_xtrack * move_left + wm.k_heading * heading_error
        angular = max(-wm.max_follow_angular,
                      min(wm.max_follow_angular, angular))
        linear = wm.forward_speed

        actual_dist = abs(perp)
        if actual_dist < wm.too_close_side:
            linear = wm.search_speed
            status = f"TOO CLOSE {side.upper()} d={actual_dist:.2f}m"
        else:
            status = f"FOLLOWING {side.upper()} d={actual_dist:.2f}m"

        self.io.publish_cmd(linear, angular)
        self.io.publish_status(
            f"{status} | "
            f"perp={perp:+.2f} target={desired_perp:+.2f} | "
            f"line_ang={math.degrees(line_angle):+.0f}deg | "
            f"n={n} | front={front:.2f}m | ang={angular:+.2f}"
        )
        return RUNNING

    # -----------------------------
    # CLEAR_END / CLEAR_NEXT
    # -----------------------------
    def clear_straight(self, next_state: str, label: str):
        """Drive straight until clear_end_distance past the row end, then report
        DONE. next_state/label are supplied by the sequencer so the published
        status names the upcoming transition."""
        wm = self.wm
        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            self.io.publish_status(f"EMERGENCY STOP during {label}: front={front:.2f}m")
            return RUNNING

        if wm.clear_start_x is None and wm.odom_x is not None:
            wm.clear_start_x = wm.odom_x
            wm.clear_start_y = wm.odom_y

        if wm.clear_start_x is None or wm.odom_x is None:
            self.io.publish_cmd(wm.forward_speed, 0.0)
            self.io.publish_status(f"{label} (no odom yet)")
            return RUNNING

        dx = wm.odom_x - wm.clear_start_x
        dy = wm.odom_y - wm.clear_start_y
        traveled = math.hypot(dx, dy)

        if traveled >= wm.clear_end_distance:
            return (Event.DONE,
                    f"Cleared {traveled:.2f}m past row end. -> {next_state}.")

        if self._elapsed_in_state() > wm.clear_end_max_time:
            return (Event.DONE, f"{label} time cap, -> {next_state} anyway.")

        self.io.publish_cmd(wm.forward_speed, 0.0)
        self.io.publish_status(
            f"{label} {traveled:.2f}/{wm.clear_end_distance:.2f}m | front={front:.2f}m"
        )
        return RUNNING

    # -----------------------------
    # Arc U-turn  (ARC_TURN)
    # -----------------------------
    def arc_turn(self):
        wm = self.wm
        sign = -1.0 if wm.current_side == "right" else +1.0

        if wm.odom_yaw is not None:
            if wm.arc_last_yaw is None:
                wm.arc_start_yaw = wm.odom_yaw
                wm.arc_last_yaw = wm.odom_yaw
            else:
                d = normalize_angle(wm.odom_yaw - wm.arc_last_yaw)
                wm.arc_accumulated_yaw += sign * d
                wm.arc_last_yaw = wm.odom_yaw

        yaw_deg = math.degrees(wm.arc_accumulated_yaw)

        v = wm.arc_linear_speed
        base_omega = v / wm.arc_radius
        omega = base_omega

        # PRIMARY exit: rotated 180 deg from arc start.
        if wm.arc_accumulated_yaw >= math.pi:
            return (Event.ARC_DONE,
                    f"Arc 180 deg complete at +{yaw_deg:.0f}. Aligning to row.")

        # Safety cap.
        if (wm.arc_accumulated_yaw >= wm.arc_max_yaw
                or self._elapsed_in_state() >= wm.arc_max_duration):
            return (Event.ARC_DONE, f"Arc capped at {yaw_deg:.0f} deg. Aligning to row.")

        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.publish_cmd(0.0, sign * base_omega)
            self.io.publish_status(
                f"ARC (front block {front:.2f}m) | yaw +{yaw_deg:.0f}deg"
            )
            return RUNNING

        self.io.publish_cmd(v, sign * omega)
        self.io.publish_status(
            f"ARC | yaw +{yaw_deg:.0f}/180deg | r={wm.arc_radius:.2f}m | "
            f"front={front:.2f}m"
        )
        return RUNNING

    # -----------------------------
    # ALIGN - heading-anchored
    # -----------------------------
    def align_to_row(self):
        wm = self.wm
        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            self.io.publish_status(f"EMERGENCY STOP during ALIGN: front={front:.2f}m")
            return RUNNING

        if wm.return_target_yaw is None or wm.odom_yaw is None:
            self.io.publish_cmd(0.0, 0.0)
            self.io.publish_status(
                "ALIGN cannot proceed: missing heading anchor or odom."
            )
            if self._elapsed_in_state() > wm.align_max_duration:
                return (Event.ALIGN_NO_REF, "ALIGN aborted - no heading reference.")
            return RUNNING

        err = normalize_angle(wm.return_target_yaw - wm.odom_yaw)

        if abs(err) < wm.align_parallel_tol:
            return (Event.ALIGNED,
                    f"ALIGN done. odom_yaw={math.degrees(wm.odom_yaw):+.1f} "
                    f"target={math.degrees(wm.return_target_yaw):+.1f} "
                    f"err={math.degrees(err):+.1f}deg. Strafing to 0.40 m.")

        mag = min(wm.align_max_angular,
                  max(wm.min_align_angular, wm.k_align * abs(err)))
        angular = math.copysign(mag, err)
        self.io.publish_cmd(0.0, angular)
        self.io.publish_status(
            f"ALIGN | odom={math.degrees(wm.odom_yaw):+.0f}deg "
            f"target={math.degrees(wm.return_target_yaw):+.0f}deg "
            f"err={math.degrees(err):+.0f}deg (tol "
            f"{math.degrees(wm.align_parallel_tol):.0f}) | rot={angular:+.2f}"
        )

        if self._elapsed_in_state() > wm.align_max_duration:
            return (Event.ALIGN_TIMEOUT,
                    f"ALIGN timeout. err={math.degrees(err):+.0f}deg. "
                    f"Strafing to 0.40 m.")
        return RUNNING

    # -----------------------------
    # LATERAL_ALIGN (crab strafe)
    # -----------------------------
    def lateral_align(self):
        wm = self.wm
        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            self.io.publish_status(f"EMERGENCY STOP during STRAFE: front={front:.2f}m")
            return RUNNING
        if front < wm.avoid_front_distance:
            self.enter_avoid(f"front={front:.2f}m")
            return RUNNING
        if wm.odom_x is None or wm.odom_yaw is None:
            self.io.stop_robot()
            self.io.publish_status("STRAFE cannot proceed: no odom.")
            return RUNNING
        if self._elapsed_in_state() > wm.lateral_align_max_duration:
            return (Event.DONE, "STRAFE timeout")

        if wm._lat_phase == "MEASURE":
            ref = (wm.outbound_yaw if wm._strafe_next_state == "FOLLOW_OUT"
                   else wm.return_target_yaw)
            wm._strafe_ref_yaw = ref if ref is not None else wm.odom_yaw
            visible, fit = self.perc.row_visible(wm.current_side)
            if not visible:
                self.io.publish_cmd(wm.search_speed, 0.0)
                self.io.publish_status("STRAFE | searching for row")
                return RUNNING
            self.mark_row_seen_if_visible(True)
            _, perp, _ = fit
            d = abs(perp)
            err = d - wm.desired_side_distance      # >0 too far, <0 too close
            if abs(err) < wm.lateral_align_tol:
                return (Event.DONE, f"aligned d={d:.2f}m (no crab needed)")
            wm._strafe_dist = min(abs(err), wm.strafe_max_dist)
            wm._strafe_toward = err > 0     # too far -> we drive toward the row
            sign = 1.0 if wm.current_side == "right" else -1.0
            away = normalize_angle(wm._strafe_ref_yaw + sign * math.pi / 2.0)
            toward = normalize_angle(wm._strafe_ref_yaw - sign * math.pi / 2.0)
            wm._strafe_heading = away if err < 0 else toward
            wm._strafe_start = None
            wm._lat_phase = "TURN"
            self.io.publish_status(
                f"STRAFE measured d={d:.2f}m -> crab {wm._strafe_dist:.2f}m "
                f"{'out' if err < 0 else 'in'} -> {wm._strafe_next_state}")
            return RUNNING

        if wm._lat_phase == "TURN":
            e = normalize_angle(wm._strafe_heading - wm.odom_yaw)
            if abs(e) < wm.align_parallel_tol:
                self.io.stop_robot()
                wm._strafe_start = (wm.odom_x, wm.odom_y)
                wm._lat_phase = "STRAFE"
                return RUNNING
            mag = min(wm.strafe_turn_speed,
                      max(wm.min_align_angular, wm.k_align * abs(e)))
            self.io.publish_cmd(0.0, math.copysign(mag, e))
            self.io.publish_status(f"STRAFE turn-perp | err={math.degrees(e):+.0f}deg")
            return RUNNING

        if wm._lat_phase == "STRAFE":
            traveled = math.hypot(wm.odom_x - wm._strafe_start[0],
                                  wm.odom_y - wm._strafe_start[1])
            if wm._strafe_toward and front <= wm.desired_side_distance:
                self.io.stop_robot()
                wm._lat_phase = "BACK"
                self.io.publish_status(
                    f"STRAFE stop: {front:.2f}m ahead (hug {wm.desired_side_distance:.2f}m)")
                return RUNNING
            if traveled >= wm._strafe_dist:
                self.io.stop_robot()
                wm._lat_phase = "BACK"
                return RUNNING
            self.io.publish_cmd(wm.strafe_drive_speed, 0.0)
            self.io.publish_status(
                f"STRAFE sliding {traveled:.2f}/{wm._strafe_dist:.2f}m "
                f"| front={front:.2f}m")
            return RUNNING

        if wm._lat_phase == "BACK":
            e = normalize_angle(wm._strafe_ref_yaw - wm.odom_yaw)
            if abs(e) < wm.align_parallel_tol:
                return (Event.DONE, f"crabbed to {wm.desired_side_distance:.2f}m")
            mag = min(wm.strafe_turn_speed,
                      max(wm.min_align_angular, wm.k_align * abs(e)))
            self.io.publish_cmd(0.0, math.copysign(mag, e))
            self.io.publish_status(f"STRAFE turn-back | err={math.degrees(e):+.0f}deg")
            return RUNNING

        return RUNNING

    # -----------------------------
    # TURN_NEXT (in-place 180 deg toward the next row)
    # -----------------------------
    def turn_to_next_row(self):
        wm = self.wm
        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            self.io.publish_status(f"EMERGENCY STOP during TURN_NEXT: front={front:.2f}m")
            return RUNNING

        if wm.outbound_yaw is None or wm.odom_yaw is None:
            self.io.stop_robot()
            self.io.publish_status("TURN_NEXT cannot proceed: no heading reference.")
            if self._elapsed_in_state() > wm.turn_next_max_duration:
                return (Event.TURN_NO_REF, "TURN_NEXT aborted blind - trying next row.")
            return RUNNING

        err = normalize_angle(wm.outbound_yaw - wm.odom_yaw)

        if abs(err) < wm.align_parallel_tol:
            return (Event.TURNED,
                    f"TURN_NEXT done (err={math.degrees(err):+.1f}deg). "
                    f"Starting row {wm.rows_completed + 1}.")

        if self._elapsed_in_state() > wm.turn_next_max_duration:
            return (Event.TURN_TIMEOUT,
                    f"TURN_NEXT timeout (err={math.degrees(err):+.0f}deg). "
                    f"Starting row {wm.rows_completed + 1} anyway.")

        turn_dir = +1.0 if wm.current_side == "right" else -1.0
        mag = min(wm.align_max_angular,
                  max(wm.min_align_angular, wm.k_align * abs(err)))
        self.io.publish_cmd(0.0, turn_dir * mag)
        self.io.publish_status(
            f"TURN_NEXT | odom={math.degrees(wm.odom_yaw):+.0f}deg "
            f"target={math.degrees(wm.outbound_yaw):+.0f}deg "
            f"err={math.degrees(err):+.0f}deg | rot={turn_dir * mag:+.2f}"
        )
        return RUNNING

    # -----------------------------
    # FRONT OBSTACLE AVOIDANCE (AVOID_FRONT)
    # -----------------------------
    def set_avoid_phase(self, phase: str):
        self.wm.avoid_phase = phase
        self.wm.avoid_phase_start_time = self.io.now()
        self.io.log_info(f"AVOID_FRONT phase changed to {phase}")

    def avoid_front_obstacle(self):
        wm = self.wm
        front = self.perc.get_front_distance()

        if front < wm.emergency_stop_distance:
            self.io.publish_cmd(wm.avoid_backup_speed, 0.0)
            self.io.publish_status(
                f"AVOID_FRONT emergency backup | front={front:.2f}m"
            )
            return RUNNING

        if wm.avoid_previous_state in ("RETURN_HOME", "COLLECT_NUTS"):
            left_clear = self.perc.get_sector_distance(25, 90)
            right_clear = self.perc.get_sector_distance(-90, -25)
            turn_dir = +1.0 if left_clear > right_clear else -1.0
            side_note = f"left={left_clear:.2f}m right={right_clear:.2f}m"
        else:
            turn_dir = +1.0 if wm.current_side == "right" else -1.0
            side_note = f"row_on={wm.current_side}"

        if wm.avoid_phase == "BACKUP":
            if self._elapsed_in_avoid_phase() < wm.avoid_backup_duration:
                self.io.publish_cmd(wm.avoid_backup_speed, 0.0)
                self.io.publish_status(
                    f"AVOID_FRONT backup | front={front:.2f}m | {side_note}"
                )
                return RUNNING
            self.set_avoid_phase("TURN_AWAY")
            return RUNNING

        if wm.avoid_phase == "TURN_AWAY":
            if self._elapsed_in_avoid_phase() < wm.avoid_turn_duration:
                self.io.publish_cmd(0.0, turn_dir * wm.avoid_turn_speed)
                self.io.publish_status(
                    f"AVOID_FRONT turning away | front={front:.2f}m | {side_note}"
                )
                return RUNNING
            self.set_avoid_phase("FORWARD_CLEAR")
            return RUNNING

        if wm.avoid_phase == "FORWARD_CLEAR":
            if self._elapsed_in_avoid_phase() < wm.avoid_forward_duration:
                self.io.publish_cmd(
                    wm.avoid_forward_speed,
                    turn_dir * 0.10
                )
                self.io.publish_status(
                    f"AVOID_FRONT clearing obstacle | front={front:.2f}m | {side_note}"
                )
                return RUNNING
            return (Event.DONE,
                    f"Obstacle avoided. Resuming {wm.avoid_previous_state}.")

        return RUNNING

    # -----------------------------
    # RETURN_HOME
    # -----------------------------
    def return_home(self):
        wm = self.wm
        if wm.home_x is None or wm.home_y is None:
            self.io.stop_robot()
            return (Event.HOME_REACHED, "No home pose saved. Robot stopped.")

        if wm.odom_x is None or wm.odom_y is None or wm.odom_yaw is None:
            self.io.stop_robot()
            self.io.publish_status("RETURN_HOME waiting for odometry")
            return RUNNING

        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            self.io.publish_status(f"EMERGENCY STOP during RETURN_HOME: front={front:.2f}m")
            return RUNNING

        if front < wm.avoid_front_distance:
            self.enter_avoid(f"front={front:.2f}m while returning home")
            return RUNNING

        # Phase 1 (OUT): back out of the field along the row direction.
        if wm._home_phase == "OUT":
            if wm.outbound_yaw is None:
                wm._home_phase = "GOAL"
            else:
                ddx, ddy = math.cos(wm.outbound_yaw), math.sin(wm.outbound_yaw)
                home_long = wm.home_x * ddx + wm.home_y * ddy
                robot_long = wm.odom_x * ddx + wm.odom_y * ddy
                out_long = home_long - wm.return_exit_margin
                if robot_long <= out_long + wm.return_goal_tolerance:
                    wm._home_phase = "GOAL"
                else:
                    step = out_long - robot_long
                    tx = wm.odom_x + step * ddx
                    ty = wm.odom_y + step * ddy
                    ex = tx - wm.odom_x
                    ey = ty - wm.odom_y
                    edist = math.hypot(ex, ey)
                    target_yaw = math.atan2(ey, ex)
                    heading_error = normalize_angle(target_yaw - wm.odom_yaw)
                    if abs(heading_error) > wm.return_heading_slowdown:
                        self.io.publish_cmd(
                            0.0, math.copysign(wm.return_max_angular, heading_error))
                    else:
                        linear = min(wm.return_linear_speed, max(0.03, 0.45 * edist))
                        angular = max(-wm.return_max_angular,
                                      min(wm.return_max_angular, 1.4 * heading_error))
                        self.io.publish_cmd(linear, angular)
                    self.io.publish_status(
                        f"RETURN_HOME exiting field | out_dist={edist:.2f}m "
                        f"| heading_err={math.degrees(heading_error):+.0f}deg "
                        f"| front={front:.2f}m"
                    )
                    if self._elapsed_in_state() > wm.return_max_duration:
                        self.io.stop_robot()
                        return (Event.HOME_REACHED,
                                f"RETURN_HOME timeout after {wm.return_max_duration:.0f}s "
                                f"while exiting field. Robot stopped.")
                    return RUNNING

        dx = wm.home_x - wm.odom_x
        dy = wm.home_y - wm.odom_y
        dist = math.hypot(dx, dy)

        # Phase 2 (GOAL): reach the saved x/y position.
        if dist > wm.return_goal_tolerance:
            target_yaw = math.atan2(dy, dx)
            heading_error = normalize_angle(target_yaw - wm.odom_yaw)

            if abs(heading_error) > wm.return_heading_slowdown:
                linear = 0.0
                angular = math.copysign(wm.return_max_angular, heading_error)
                mode = "turning toward start"
            else:
                linear = min(wm.return_linear_speed, max(0.03, 0.45 * dist))
                angular = max(-wm.return_max_angular,
                              min(wm.return_max_angular, 1.4 * heading_error))
                mode = "driving to start"

            self.io.publish_cmd(linear, angular)
            self.io.publish_status(
                f"RETURN_HOME {mode} | dist={dist:.2f}m "
                f"| heading_err={math.degrees(heading_error):+.0f}deg "
                f"| front={front:.2f}m"
            )
            if self._elapsed_in_state() > wm.return_max_duration:
                self.io.stop_robot()
                return (Event.HOME_REACHED,
                        f"RETURN_HOME timeout after {wm.return_max_duration:.0f}s. "
                        f"Robot stopped.")
            return RUNNING

        # At x/y start. Rotate back to the saved start heading if known.
        if wm.home_yaw is not None:
            yaw_error = normalize_angle(wm.home_yaw - wm.odom_yaw)
            if abs(yaw_error) > wm.return_yaw_tolerance:
                angular = max(-wm.return_max_angular,
                              min(wm.return_max_angular, 1.2 * yaw_error))
                if abs(angular) < wm.min_align_angular:
                    angular = math.copysign(wm.min_align_angular, yaw_error)
                self.io.publish_cmd(0.0, angular)
                self.io.publish_status(
                    f"RETURN_HOME at start, aligning yaw | "
                    f"yaw_err={math.degrees(yaw_error):+.0f}deg"
                )
                return RUNNING

        self.io.stop_robot()
        # Reached start: hand off to _return_end_state (DONE, or PAUSED when a
        # pause is active). The sequencer applies the hand-off.
        return (Event.RETURN_FINISHED,
                f"Returned to start. Final distance={dist:.2f}m."
                + ("" if wm._paused else " Mission complete."))

    # -----------------------------
    # RESUME_NAV - skip back to a paused / missed row, then re-sweep
    # -----------------------------
    def resume_nav(self):
        """Skip back to the paused row's start through the open buffer, then
        re-align. HEADLAND -> TRAVERSE -> TO_ROW are go-to waypoints (kept in the
        buffer, clear of the row starts); ALIGN_OUT rotates to the outbound
        heading; on arrival the sequencer restores the row and re-enters via the
        crab. Returns RUNNING, or (RESUME_DONE, reason) to start the sweep."""
        wm = self.wm
        if self._elapsed_in_state() > wm.resume_max_duration:
            self.io.log_warn("RESUME_NAV leg timed out - resuming from here.")
            return (Event.RESUME_DONE, "RESUME_NAV timeout")

        if (wm.odom_x is None or wm.odom_yaw is None
                or wm.outbound_yaw is None or wm.home_x is None
                or wm._pause_anchor is None):
            self.io.stop_robot()
            self.io.publish_status("RESUME_NAV waiting for odom/home/anchor.")
            return RUNNING

        # Row-frame basis at home: d along outbound, across perpendicular.
        cth, sth = math.cos(wm.outbound_yaw), math.sin(wm.outbound_yaw)
        dxu, dyu = cth, sth
        axu, ayu = -sth, cth
        ax, ay = wm._pause_anchor
        a_a = (ax - wm.home_x) * dxu + (ay - wm.home_y) * dyu
        l_a = (ax - wm.home_x) * axu + (ay - wm.home_y) * ayu
        headland_a = min(0.0, a_a) - wm.resume_headland_margin
        w1 = (wm.home_x + headland_a * dxu, wm.home_y + headland_a * dyu)
        w2 = (w1[0] + l_a * axu, w1[1] + l_a * ayu)
        w3 = (ax, ay)
        tol = wm.resume_wp_tol
        row = wm._pause_row + 1

        if wm._resume_phase == "HEADLAND":
            if self._drive_to(w1[0], w1[1], tol, "RESUME_NAV"):
                wm._resume_phase = "TRAVERSE"
            else:
                self.io.publish_status(f"RESUME_NAV into buffer -> row {row}")
            return RUNNING
        if wm._resume_phase == "TRAVERSE":
            if self._drive_to(w2[0], w2[1], tol, "RESUME_NAV"):
                wm._resume_phase = "TO_ROW"
            else:
                self.io.publish_status(f"RESUME_NAV crossing buffer to row {row}")
            return RUNNING
        if wm._resume_phase == "TO_ROW":
            if self._drive_to(w3[0], w3[1], tol, "RESUME_NAV"):
                wm._resume_phase = "ALIGN_OUT"
            else:
                self.io.publish_status(f"RESUME_NAV entering row {row}")
            return RUNNING
        # ALIGN_OUT: rotate in place to the outbound heading, then re-sweep.
        err = normalize_angle(wm.outbound_yaw - wm.odom_yaw)
        if abs(err) < wm.align_parallel_tol:
            return (Event.RESUME_DONE, f"Back at row {row}")
        mag = min(wm.align_max_angular,
                  max(wm.min_align_angular, wm.k_align * abs(err)))
        self.io.publish_cmd(0.0, math.copysign(mag, err))
        self.io.publish_status(
            f"RESUME_NAV align-out row {row} | err={math.degrees(err):+.0f}deg")
        return RUNNING

    # -----------------------------
    # COLLECT_NUTS - tree-aware pickup helpers + skill
    # -----------------------------
    def _map_to_odom(self):
        return self.io.map_to_odom()

    @staticmethod
    def _apply_tf(mx, my, tf):
        tx, ty, c, s = tf
        return (tx + c * mx - s * my, ty + s * mx + c * my)

    def _uncollected_targets(self, tf):
        out = []
        for (mx, my) in self.wm.uncollected_map:
            if (round(mx, 2), round(my, 2)) in self.wm._collect_skip:
                continue
            out.append(((mx, my), self._apply_tf(mx, my, tf)))
        return out

    def _row_units(self):
        """(d, sweeper_unit) in odom; None if the heading anchor isn't set."""
        wm = self.wm
        if wm.outbound_yaw is None:
            return None
        th = wm.outbound_yaw
        d = (math.cos(th), math.sin(th))
        if wm.start_side == "left":
            sw = (-math.sin(th), math.cos(th))   # left of d
        else:
            sw = (math.sin(th), -math.cos(th))   # right of d
        return d, sw

    def update_swept_bounds(self):
        wm = self.wm
        units = self._row_units()
        if units is None or wm.odom_x is None:
            return
        (dx, dy), (sx, sy) = units
        lon = wm.odom_x * dx + wm.odom_y * dy
        lat = wm.odom_x * sx + wm.odom_y * sy
        if wm._swept_long_min is None:
            wm._swept_long_min = wm._swept_long_max = lon
            wm._swept_lat_min = wm._swept_lat_max = lat
        else:
            wm._swept_long_min = min(wm._swept_long_min, lon)
            wm._swept_long_max = max(wm._swept_long_max, lon)
            wm._swept_lat_min = min(wm._swept_lat_min, lat)
            wm._swept_lat_max = max(wm._swept_lat_max, lat)
        if not wm._swept_path:
            wm._swept_path.append((wm.odom_x, wm.odom_y))
        else:
            lx, ly = wm._swept_path[-1]
            if math.hypot(wm.odom_x - lx, wm.odom_y - ly) >= wm.path_sample_spacing:
                wm._swept_path.append((wm.odom_x, wm.odom_y))
                if len(wm._swept_path) > wm.path_max_points:
                    wm._swept_path.pop(0)

    def _nearest_path_index(self, x: float, y: float) -> int:
        best_i, best_d = 0, float("inf")
        for i, (px, py) in enumerate(self.wm._swept_path):
            d = (px - x) * (px - x) + (py - y) * (py - y)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _drive_to_via_path(self, ex: float, ey: float) -> bool:
        wm = self.wm
        path = wm._swept_path
        if len(path) < 2:
            return self._drive_to(ex, ey, wm.collect_arrive_tol)
        if wm._path_goal is None:
            wm._path_goal = self._nearest_path_index(ex, ey)
            wm._path_idx = self._nearest_path_index(wm.odom_x, wm.odom_y)
        i_goal = wm._path_goal
        if wm._path_idx == i_goal:
            return self._drive_to(ex, ey, wm.collect_arrive_tol)
        step = 1 if i_goal > wm._path_idx else -1
        ox, oy = wm.odom_x, wm.odom_y
        nearest_i = wm._path_idx
        nearest_d = math.hypot(path[nearest_i][0] - ox, path[nearest_i][1] - oy)
        probe = wm._path_idx
        while probe != i_goal:
            probe += step
            d = math.hypot(path[probe][0] - ox, path[probe][1] - oy)
            if d < nearest_d:
                nearest_d, nearest_i = d, probe
            elif d > nearest_d + wm.path_wp_tol:
                break
        wm._path_idx = nearest_i
        if wm._path_idx == i_goal:
            return self._drive_to(ex, ey, wm.collect_arrive_tol)
        carrot = wm._path_idx
        for _ in range(max(1, wm.path_lookahead)):
            if carrot == i_goal:
                break
            carrot += step
        self._drive_to(path[carrot][0], path[carrot][1], 0.0)
        return False

    def _side_clearance(self):
        return (self.perc.get_sector_distance(25.0, 90.0),
                self.perc.get_sector_distance(-90.0, -25.0))

    def _safety_for_nut(self, nox, noy):
        wm = self.wm
        units = self._row_units()
        if units is None or wm._swept_long_min is None:
            return None
        (dx, dy), (su_x, su_y) = units
        lx = nox - wm.collect_sweep_offset * su_x
        ly = noy - wm.collect_sweep_offset * su_y
        l_lat = lx * su_x + ly * su_y
        l_lat = max(wm._swept_lat_min, min(wm._swept_lat_max, l_lat))
        nut_long = nox * dx + noy * dy
        near, far = wm._swept_long_min, wm._swept_long_max
        if abs(nut_long - near) <= abs(far - nut_long):
            end_long = near
            dip_long = nut_long + wm.collect_sweep_through
        else:
            end_long = far
            dip_long = nut_long - wm.collect_sweep_through

        def pt(longi, lat):
            return (longi * dx + lat * su_x, longi * dy + lat * su_y)

        sx, sy = pt(end_long, l_lat)
        dipx, dipy = pt(dip_long, l_lat)
        return (sx, sy, dipx, dipy, end_long, l_lat)

    def _drive_to(self, tx: float, ty: float, tol: float,
                  avoid_state: str = "COLLECT_NUTS") -> bool:
        """Go-to-point with the RETURN_HOME control style + AVOID_FRONT safety.
        avoid_state names the move (for the avoidance status); AVOID_FRONT always
        resumes the currently-active state. Returns True within tol."""
        wm = self.wm
        if wm.odom_x is None or wm.odom_yaw is None:
            self.io.stop_robot()
            return False
        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            return False
        if front < wm.avoid_front_distance:
            wm._collect_avoid_count += 1
            self.enter_avoid(f"front={front:.2f}m during {avoid_state}")
            return False
        dx = tx - wm.odom_x
        dy = ty - wm.odom_y
        dist = math.hypot(dx, dy)
        if dist <= tol:
            self.io.stop_robot()
            return True
        he = normalize_angle(math.atan2(dy, dx) - wm.odom_yaw)
        if abs(he) > wm.return_heading_slowdown:
            self.io.publish_cmd(0.0, math.copysign(wm.return_max_angular, he))
        else:
            lin = min(wm.return_linear_speed, max(0.03, 0.45 * dist))
            ang = max(-wm.return_max_angular, min(wm.return_max_angular, 1.4 * he))
            self.io.publish_cmd(lin, ang)
        return False

    def _collect_drive(self, tx: float, ty: float, tol: float) -> bool:
        """Guarded go-to-point for ALL collection motion (front + side guard)."""
        wm = self.wm
        if wm.odom_x is None or wm.odom_yaw is None:
            self.io.stop_robot()
            return False
        front = self.perc.get_front_distance()
        if front < wm.emergency_stop_distance:
            self.io.stop_robot()
            return False
        if front < wm.avoid_front_distance:
            wm._collect_avoid_count += 1
            self.enter_avoid(f"front={front:.2f}m during collect")
            return False
        dx = tx - wm.odom_x
        dy = ty - wm.odom_y
        dist = math.hypot(dx, dy)
        if dist <= tol:
            self.io.stop_robot()
            return True
        he = normalize_angle(math.atan2(dy, dx) - wm.odom_yaw)
        if abs(he) > wm.return_heading_slowdown:
            lin = 0.0
            ang = math.copysign(wm.return_max_angular, he)
        else:
            lin = min(wm.return_linear_speed, max(0.03, 0.45 * dist))
            ang = max(-wm.return_max_angular, min(wm.return_max_angular, 1.4 * he))
        left, right = self._side_clearance()
        if left < wm.collect_side_clearance or right < wm.collect_side_clearance:
            ang = -wm.collect_turn_away if left <= right else wm.collect_turn_away
            lin = min(lin, wm.search_speed)
        self.io.publish_cmd(lin, ang)
        return False

    def collect_nuts(self):
        """Returns RUNNING, or (Event.COLLECT_GO_HOME, reason) when collection is
        finished/aborted (the sequencer then asks the deliberator to go home)."""
        wm = self.wm
        if wm.odom_x is None or wm.odom_yaw is None:
            self.io.stop_robot()
            self.io.publish_status("COLLECT_NUTS waiting for odometry")
            return RUNNING
        tf = self._map_to_odom()
        if tf is None:
            self.io.stop_robot()
            self.io.publish_status("COLLECT_NUTS waiting for map->odom TF")
            return RUNNING
        if self._row_units() is None:
            return (Event.COLLECT_GO_HOME, "COLLECT_NUTS: no row heading anchor.")

        now = self.io.now().nanoseconds * 1e-9

        valid = False
        if wm._collect_target is not None:
            key = (round(wm._collect_target[0], 2), round(wm._collect_target[1], 2))
            valid = key not in wm._collect_skip and any(
                (round(mx, 2), round(my, 2)) == key for (mx, my) in wm.uncollected_map)
        if not valid:
            if wm._collect_target is not None:
                pkey = (round(wm._collect_target[0], 2), round(wm._collect_target[1], 2))
                if pkey not in wm._collect_skip:
                    wm._collect_consec_skips = 0
            targets = self._uncollected_targets(tf)
            if not targets:
                return (Event.COLLECT_GO_HOME, "All reachable nuts collected.")
            targets.sort(key=lambda t: math.hypot(t[1][0] - wm.odom_x,
                                                  t[1][1] - wm.odom_y))
            wm._collect_target = targets[0][0]
            same_aisle = False
            safety0 = self._safety_for_nut(*targets[0][1])
            if safety0 is not None:
                _, (su0x, su0y) = self._row_units()
                cur_lat0 = wm.odom_x * su0x + wm.odom_y * su0y
                same_aisle = abs(cur_lat0 - safety0[5]) < 1.5 * wm.collect_arrive_tol
            wm._collect_phase = "DIP" if same_aisle else "TO_HEADLAND"
            wm._collect_deadline = now + wm.collect_visit_timeout
            wm._collect_avoid_count = 0
            wm._collect_best = float("inf")

        if wm._collect_avoid_count > wm.collect_max_avoid:
            gh = self._skip_current_nut("blocked (avoidance limit)")
            if gh is not None:
                return (Event.COLLECT_GO_HOME, gh)
            return RUNNING

        if wm._collect_deadline is not None and now > wm._collect_deadline:
            gh = self._skip_current_nut("visit timeout")
            if gh is not None:
                return (Event.COLLECT_GO_HOME, gh)
            return RUNNING

        nox, noy = self._apply_tf(wm._collect_target[0], wm._collect_target[1], tf)
        (dx, dy), (sxu, syu) = self._row_units()

        safety = self._safety_for_nut(nox, noy)
        if safety is None:
            return (Event.COLLECT_GO_HOME, "COLLECT_NUTS: swept region unknown.")
        sxp, syp, _dipx, _dipy, end_long, l_lat = safety
        cur_lat = wm.odom_x * sxu + wm.odom_y * syu

        if wm._collect_phase == "TO_HEADLAND":
            tx = end_long * dx + cur_lat * sxu
            ty = end_long * dy + cur_lat * syu
            nxt = "ALONG_HEADLAND"
        elif wm._collect_phase == "ALONG_HEADLAND":
            tx, ty = sxp, syp
            nxt = "DIP"
        else:  # DIP
            nut_long = nox * dx + noy * dy
            cur_long = wm.odom_x * dx + wm.odom_y * dy
            direction = 1.0 if nut_long >= cur_long else -1.0
            dip_long = nut_long + direction * wm.collect_sweep_through
            dip_long = max(wm._swept_long_min, min(wm._swept_long_max, dip_long))
            tx = dip_long * dx + l_lat * sxu
            ty = dip_long * dy + l_lat * syu
            nxt = None

        remaining = math.hypot(tx - wm.odom_x, ty - wm.odom_y)
        if remaining < wm._collect_best - 1e-3:
            wm._collect_best = remaining
            wm._collect_deadline = now + wm.collect_visit_timeout

        arrived = self._collect_drive(tx, ty, wm.collect_arrive_tol)
        if arrived and wm._collect_phase == "DIP":
            gh = self._skip_current_nut("dipped past without pickup")
            if gh is not None:
                return (Event.COLLECT_GO_HOME, gh)
            return RUNNING
        if arrived:
            wm._collect_phase = nxt
            wm._collect_deadline = now + wm.collect_visit_timeout
            wm._collect_best = float("inf")
            wm._collect_avoid_count = 0
            return RUNNING
        self.io.publish_status(
            f"COLLECT {wm._collect_phase} | nut=({nox:+.2f},{noy:+.2f}) | "
            f"-> ({tx:+.2f},{ty:+.2f}) rem={remaining:.2f}m"
        )
        return RUNNING

    def _skip_current_nut(self, reason: str) -> Optional[str]:
        """Abandon the current target. Returns a go-home reason string when too
        many nuts in a row are unreachable, else None."""
        wm = self.wm
        self.io.stop_robot()
        if wm._collect_target is not None:
            wm._collect_skip.add((round(wm._collect_target[0], 2),
                                  round(wm._collect_target[1], 2)))
            self.io.log_warn(
                f"COLLECT_NUTS: skipping nut "
                f"({wm._collect_target[0]:+.2f},{wm._collect_target[1]:+.2f}) "
                f"- {reason}."
            )
        wm._collect_target = None
        wm._collect_consec_skips += 1
        if wm._collect_consec_skips >= wm.collect_max_consec_skips:
            return (f"COLLECT_NUTS: {wm._collect_consec_skips} nuts unreachable "
                    f"in a row; giving up on collection.")
        return None
