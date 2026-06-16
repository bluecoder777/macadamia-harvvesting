# macadamia-harvvesting

simple_row_follower New

What changed
New: get_obstacle_info() method
The original get_front_distance() only returned a single number — how far away. The robot had no idea which side the obstacle was on, so it couldn't steer around it. The new method scans the same 35° forward cone but now returns a tuple: (min_dist, lateral_offset). lateral_offset > 0 means the obstacle is left of centre, < 0 means right. get_front_distance() now just calls this and discards the lateral value, so the rest of the code is unchanged.
New: three ROS2 parameters
obstacle_warn_distance  0.30 m   # avoidance kicks in here
avoid_resume_distance   0.45 m   # obstacle must be this far before resuming
avoid_side_gain         1.5      # how aggressively to steer sideways
avoid_creep_speed       0.03 m/s # forward speed while sidestepping

You can tune all of these at launch: --ros-args -p obstacle_warn_distance:=0.25
New: AVOID_OBSTACLE state + _enter_avoidance() + avoid_obstacle()

The three-zone system now looks like this:

← 0.18 m →←—— 0.30 m ——→←——— 0.45 m ———→
  HARD STOP   SIDESTEP       CLEAR / RESUME

When anything enters the 0.30 m warn zone, _enter_avoidance() is called, which saves the current state name into _pre_avoid_state (e.g. "FOLLOW_OUT") and transitions to AVOID_OBSTACLE. avoid_obstacle() then:

Steers proportionally away from the lateral offset — obstacle on the left means steer right, and vice versa.
Creeps forward slowly, slowing further as the obstacle gets closer.
The moment the obstacle retreats past 0.45 m, it transitions directly back to the saved state and reseeds last_row_seen_time so the row-lost timer doesn't fire.
If the obstacle stays there for 12 seconds, it stops and flags for manual intervention rather than looping forever.

follow_side() now returns a bool
It returns False if it triggered avoidance, True otherwise. The FOLLOW_OUT and FOLLOW_BACK branches in control_loop check this and skip the end-of-pass logic on that tick — otherwise the robot could falsely decide the row is lost the moment it starts sidestepping.
clear_straight() also calls avoidance
The original froze in CLEAR_END and CLEAR_NEXT too. Now those states also enter AVOID_OBSTACLE and resume correctly once clear.
The arc is deliberately left alone
During ARC_TURN the robot is mid-pivot and avoidance would interrupt the yaw accumulation, leaving it with a corrupted heading. The original spin-only behaviour (keep rotating, drop linear) is kept there — the tight 0.40 m radius geometry means an obstacle during the arc is almost certainly the tree it's already orbiting around, which is expected.


No changes needed to package.xml, setup.py, setup.cfg, or the launch file — the new ROS2 parameters (obstacle_warn_distance, avoid_resume_distance, etc.) all have defaults so the node starts identically to before without any extra config. If you want to override them at runtime:
bashros2 run macadamia_sweep simple_row_follower \
  --ros-args \
  -p obstacle_warn_distance:=0.25 \
  -p avoid_resume_distance:=0.50 \
  -p avoid_side_gain:=2.0
Or add them to the launch file if you want them baked in permanently:
python# in simple_demo.launch.py
Node(
    package="macadamia_sweep",
    executable="simple_row_follower",
    name="simple_row_follower",
    output="screen",
    parameters=[{
        "obstacle_warn_distance": 0.30,
        "avoid_resume_distance": 0.45,
        "avoid_side_gain": 1.5,
    }],
)