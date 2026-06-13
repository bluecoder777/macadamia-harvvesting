#!/usr/bin/env python3
"""Inject FAKE nut detections so you can verify the tracker + collection logic
end-to-end WITHOUT the camera or any HSV tuning.

It publishes a fixed nut position (in the map frame) onto /nuts/detections at a
steady rate -- exactly what nut_detector would emit. nut_tracker then confirms
it (after min_hits), shows a RED sphere, and flips it GREEN when you drive
base_link within collection_radius of it.

Because the point is FIXED in the map frame (computed once at startup), it stays
put as the robot moves -- so driving over it is a genuine test of the drive-over
collection check.

Usage (after sourcing your workspace, with nut_tracker running):
    # one nut 0.6 m ahead of the robot's current pose:
    python3 tools/inject_fake_nuts.py
    # a nut 0.8 m ahead:
    python3 tools/inject_fake_nuts.py --ahead 0.8
    # explicit map coordinates (repeatable), e.g. two nuts:
    python3 tools/inject_fake_nuts.py --at 1.5 0.2 --at 1.8 -0.3

Then teleop the robot over the red sphere and watch it turn green and the
/snc_status count tick up.
"""

import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseArray, Pose
import tf2_ros


def get_robot_pose(node, buffer, timeout=5.0):
    """Spin briefly until map->base_link is available; return (x, y, yaw)."""
    deadline = node.get_clock().now().nanoseconds + int(timeout * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            tf = buffer.lookup_transform("map", "base_link", Time())
        except Exception:
            continue
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw
    return None


class Injector(Node):
    def __init__(self, points_map):
        super().__init__("inject_fake_nuts")
        self.points = points_map
        self.pub = self.create_publisher(PoseArray, "/nuts/detections", 10)
        self.create_timer(0.2, self.tick)  # 5 Hz, like a throttled detector
        self.get_logger().info(
            f"Injecting {len(points_map)} fake nut(s) on /nuts/detections (map): "
            + ", ".join(f"({x:+.2f},{y:+.2f})" for x, y in points_map)
        )

    def tick(self):
        pa = PoseArray()
        pa.header.frame_id = "map"
        pa.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in self.points:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.pub.publish(pa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", nargs=2, type=float, action="append", metavar=("X", "Y"),
                    help="absolute map coordinates (repeatable)")
    ap.add_argument("--ahead", type=float, default=None,
                    help="place one nut this many metres ahead of the robot")
    args = ap.parse_args()

    rclpy.init()
    node = Node("inject_fake_nuts_bootstrap")
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer, node)

    points = []
    if args.at:
        points.extend((x, y) for x, y in args.at)
    if args.ahead is not None or not points:
        d = args.ahead if args.ahead is not None else 0.6
        pose = get_robot_pose(node, buffer)
        if pose is None:
            node.get_logger().error(
                "Could not read map->base_link TF; give explicit --at X Y instead.")
            node.destroy_node(); rclpy.shutdown(); return
        rx, ry, ryaw = pose
        points.append((rx + d * math.cos(ryaw), ry + d * math.sin(ryaw)))
    node.destroy_node()

    inj = Injector(points)
    try:
        rclpy.spin(inj)
    except KeyboardInterrupt:
        pass
    finally:
        inj.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
