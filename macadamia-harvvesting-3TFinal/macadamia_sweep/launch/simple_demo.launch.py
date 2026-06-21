from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="macadamia_sweep",
            executable="row_follower_3t",
            name="row_follower_3t",
            output="screen",
        ),
    ])
