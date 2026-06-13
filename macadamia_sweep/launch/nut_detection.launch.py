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

    return LaunchDescription([
        DeclareLaunchArgument("rgb_topic", default_value="/oak/rgb/image_rect"),
        DeclareLaunchArgument("target_frame", default_value="map"),

        Node(
            package="macadamia_sweep",
            executable="nut_detector",
            name="nut_detector",
            output="screen",
            parameters=[{
                "rgb_topic": rgb_topic,
                "target_frame": target_frame,
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
