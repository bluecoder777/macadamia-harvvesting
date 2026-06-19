#!/usr/bin/env python3
"""Shared, ROS-free helpers and event tags used across the three tiers."""

import math


def normalize_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]. Verbatim port of the original helper."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def opposite(side: str) -> str:
    return "left" if side == "right" else "right"


class Skill:
    """Names of the reactive skills (one is active at a time).

    These double as the sequencer ``state`` strings, kept identical to the
    original FSM state names so every published /snc_status message and the
    swept-state membership test are byte-for-byte unchanged.
    """

    WAITING = "WAITING"
    FOLLOW_OUT = "FOLLOW_OUT"
    CLEAR_END = "CLEAR_END"
    ARC_TURN = "ARC_TURN"
    ALIGN = "ALIGN"
    LATERAL_ALIGN = "LATERAL_ALIGN"
    FOLLOW_BACK = "FOLLOW_BACK"
    CLEAR_NEXT = "CLEAR_NEXT"
    TURN_NEXT = "TURN_NEXT"
    AVOID_FRONT = "AVOID_FRONT"
    COLLECT_NUTS = "COLLECT_NUTS"   # superseded by re-sweep collection; never entered
    RETURN_HOME = "RETURN_HOME"
    RESUME_NAV = "RESUME_NAV"       # skip back to a paused/missed row and re-sweep it
    PAUSED = "PAUSED"               # parked at home, waiting for /resume_collection
    DONE = "DONE"
    STOPPED = "STOPPED"


class Event:
    """Outcome tags a skill reports to the sequencer after a tick.

    RUNNING means "still executing, no transition"; everything else is a
    completion/abort signal the sequencer (and, at mission forks, the
    Deliberator) interprets to choose the next skill.
    """

    RUNNING = "RUNNING"          # keep me active
    OBSTACLE = "OBSTACLE"        # front blocked -> sequencer enters AVOID_FRONT
    DONE = "DONE"               # generic success (cleared / strafed / avoided)
    # Skill-specific successes / aborts:
    ARC_DONE = "ARC_DONE"
    ALIGNED = "ALIGNED"
    ALIGN_TIMEOUT = "ALIGN_TIMEOUT"
    ALIGN_NO_REF = "ALIGN_NO_REF"        # missing heading anchor, timed out
    TURNED = "TURNED"
    TURN_TIMEOUT = "TURN_TIMEOUT"
    TURN_NO_REF = "TURN_NO_REF"
    HOME_REACHED = "HOME_REACHED"        # return aborted (timeout / no home) -> DONE
    RETURN_FINISHED = "RETURN_FINISHED"  # reached start -> hand to _return_end_state
    RESUME_DONE = "RESUME_DONE"          # resume_nav arrived/timed out -> start sweep
    COLLECT_GO_HOME = "COLLECT_GO_HOME"  # collection finished/aborted -> go home
