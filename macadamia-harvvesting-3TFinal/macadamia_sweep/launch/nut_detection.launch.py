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
    diag_csv = LaunchConfiguration("diag_csv")

    return LaunchDescription([
        DeclareLaunchArgument("rgb_topic", default_value="/oak/rgb/image_rect"),
        DeclareLaunchArgument("target_frame", default_value="map"),
        DeclareLaunchArgument(
            "detect_mode", default_value="color",
            description="color = match the orange nut hue (robust in clutter); "
                        "background = subtract the floor colour"),
        DeclareLaunchArgument(
            "diag_csv", default_value="",
            description="path to write a per-blob diagnostic CSV (empty = off)"),

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
        Node(
            package="macadamia_sweep",
            executable="nut_tracker",
            name="nut_tracker",
            output="screen",
            parameters=[{
                "map_frame": target_frame,
            }],
        ),
        # Detects the trees from the lidar -> /trees + zone markers, and the
        # tracker keeps only nuts within tree_gate_radius of a tree.
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
