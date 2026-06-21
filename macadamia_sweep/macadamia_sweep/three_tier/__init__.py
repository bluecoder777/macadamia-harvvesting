"""3T (three-tier) robot architecture for the macadamia single-row sweep.

The controller is organised as the classic three-tier ("3T") robot control
architecture (Bonasso, Firby, Kortenkamp, Miller, Slack):

    Deliberator  (planner / mission tier)  -- deliberator.py
        Owns the mission plan and all long-horizon decisions: when to start,
        how many rows to sweep (fixed count vs. auto), whether to collect
        missed nuts, and when to return home.

    Sequencer    (executive tier)          -- sequencer.py
        Owns the task decomposition: it activates exactly one reactive skill
        at a time, watches the skill-completion events and perception guards,
        and drives the state transitions that choreograph a row sweep
        (follow -> clear -> U-turn arc -> align -> strafe -> follow back ...).
        At mission forks it asks the Deliberator what to do next.

    Skill layer  (reactive / controller)   -- skills.py + perception.py
        Tight sensor->actuator feedback loops. ``perception.py`` is the
        situated-recognition half (line fits, front distance, "last tree
        abeam", next-row detection); ``skills.py`` is the acting half (row
        following, clear-straight, arc turn, align, crab strafe, front
        avoidance, return-home drive, nut collection). Skills publish velocity
        commands and report a status/event; they never decide what runs next.

    World model  (shared blackboard)       -- world_model.py
        All sensor caches, parameters and mutable state the three tiers read
        and write. The tiers are stateless coordinators over this blackboard.

    Agent        (ROS node + wiring)       -- agent.py
        The thin ROS2 node: owns the publishers/subscribers/parameters/timer
        and the TF buffer, fills the world model from sensor callbacks, and
        ticks Deliberator-input -> Sequencer -> Skill once per 10 Hz cycle.

The clean tier split keeps each concern in one place: mission planning in the
Deliberator, choreography in the Sequencer, and tight reactive control in the
Skill layer, all coordinating through the shared world model.
"""
