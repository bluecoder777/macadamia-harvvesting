"""Launch the nut perception + world-model pair.

    ros2 launch macadamia_sweep nut_detection.launch.py

Optionally override the colour or camera:
    ros2 launch macadamia_sweep nut_detection.launch.py s_min:=140 h_lo2:=150
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rgb_topic = LaunchConfiguration("rgb_topic")
    target_frame = LaunchConfiguration("target_frame")
    detect_mode = LaunchConfiguration("detect_mode")

    return LaunchDescription([
        DeclareLaunchArgument("rgb_topic", default_value="/oak/rgb/image_rect"),
        DeclareLaunchArgument("target_frame", default_value="map"),
        DeclareLaunchArgument(
            "detect_mode", default_value="color",
            description="color = match the orange nut hue (robust in clutter); "
                        "background = subtract the floor colour"),

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
