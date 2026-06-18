"""One launch for the whole mission: sweep + nut detection + nut tracking.

Brings up all three nodes together:
  * simple_row_follower  -- drives the row sweep (waits for /sweep_start)
  * nut_detector         -- perception (runs continuously)
  * nut_tracker          -- world model + collected/uncollected markers

Detection and tracking start immediately; the sweep itself waits for the
trigger, so the flow is just:

    ros2 launch macadamia_sweep sweep_and_detect.launch.py
    ros2 topic pub --once /sweep_start std_msgs/msg/Empty "{}"

Stop the sweep any time with:
    ros2 topic pub --once /sweep_stop std_msgs/msg/Empty "{}"

Common overrides:
    ros2 launch macadamia_sweep sweep_and_detect.launch.py start_side:=left max_rows:=4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    start_side = LaunchConfiguration("start_side")
    max_rows = LaunchConfiguration("max_rows")
    rgb_topic = LaunchConfiguration("rgb_topic")
    target_frame = LaunchConfiguration("target_frame")
    detect_mode = LaunchConfiguration("detect_mode")
    diag_csv = LaunchConfiguration("diag_csv")

    return LaunchDescription([
        DeclareLaunchArgument("start_side", default_value="right",
                              description="row side for the first pass: right|left"),
        DeclareLaunchArgument("max_rows", default_value="0",
                              description="0 = auto (continue while a next row is seen)"),
        DeclareLaunchArgument("rgb_topic", default_value="/oak/rgb/image_rect"),
        DeclareLaunchArgument("target_frame", default_value="map"),
        DeclareLaunchArgument(
            "detect_mode", default_value="color",
            description="color = match the orange nut hue (robust in clutter); "
                        "background = subtract the floor colour"),
        DeclareLaunchArgument(
            "diag_csv", default_value="",
            description="path to write a per-blob diagnostic CSV (empty = off)"),

        # --- Sweep controller (3T architecture; waits for /sweep_start) ---
        Node(
            package="macadamia_sweep",
            executable="row_follower_3t",
            name="row_follower_3t",
            output="screen",
            parameters=[{
                "start_side": start_side,
                # max_rows is an int param; convert the launch-arg string.
                "max_rows": ParameterValue(max_rows, value_type=int),
            }],
        ),

        # --- Perception (runs continuously) ---
        Node(
            package="macadamia_sweep",
            executable="nut_detector",
            name="nut_detector",
            output="screen",
            parameters=[{
                "rgb_topic": rgb_topic,
                "target_frame": target_frame,
                "detect_mode": detect_mode,
                "diag_csv": diag_csv,
                # Orange nut colour band (used when detect_mode == color).
                # Single band H 0-18 (orange doesn't wrap, so both bands are the
                # same); low S/V to capture the WHOLE card, not just its
                # saturated core -- avoids the small/ragged fragmentation.
                "h_lo1": 0, "h_hi1": 18,
                "h_lo2": 0, "h_hi2": 18,
                "s_min": 50, "v_min": 40,
            }],
        ),

        # --- World model + markers (runs continuously) ---
        Node(
            package="macadamia_sweep",
            executable="nut_tracker",
            name="nut_tracker",
            output="screen",
            parameters=[{
                "map_frame": target_frame,
                # The sweeper is on the hug side, which is the side the follower
                # starts on. Keep them in lock-step so the collection point and
                # the pickup geometry never disagree.
                "sweep_side": start_side,
            }],
        ),

        # --- Tree mapper: lidar trees -> /trees + zones; the tracker keeps
        #     only nuts within tree_gate_radius of a tree ---
        Node(
            package="macadamia_sweep",
            executable="tree_mapper",
            name="tree_mapper",
            output="screen",
            parameters=[{
                "target_frame": target_frame,
            }],
        ),
    ])
