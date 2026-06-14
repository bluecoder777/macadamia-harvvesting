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

        # --- Sweep controller (waits for /sweep_start) ---
        Node(
            package="macadamia_sweep",
            executable="simple_row_follower",
            name="simple_row_follower",
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
                # Orange nut colour band (used when detect_mode == color).
                "h_lo1": 5, "h_hi1": 28,
                "h_lo2": 5, "h_hi2": 28,
                "s_min": 90, "v_min": 60,
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
            }],
        ),
    ])
