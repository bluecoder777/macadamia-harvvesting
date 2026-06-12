from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="macadamia_sweep",
            executable="simple_row_follower",
            name="simple_row_follower",
            output="screen",
        ),
    ])
