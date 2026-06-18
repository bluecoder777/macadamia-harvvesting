#!/usr/bin/env python3

import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray


class MissionSummaryLogger(Node):
    """
    Lightweight mission summary logger for the macadamia row-following robot.

    This node does NOT modify the main robot controller. It only subscribes to
    existing ROS 2 topics and writes one summary row per mission to a CSV file.
    """

    def __init__(self):
        super().__init__("mission_summary_logger")

        self.log_dir = os.path.expanduser("~/macadamia_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.summary_path = os.path.join(self.log_dir, "mission_summary.csv")

        self.mission_active = False
        self.start_time = None
        self.end_time = None

        self.final_status = "WAITING"
        self.last_status = "WAITING"

        self.state_changes = 0
        self.avoid_events = 0
        self.emergency_stop_events = 0
        self.return_home_triggered = False
        self.return_home_completed = False

        self.final_x = None
        self.final_y = None
        self.final_yaw_deg = None
        self.missed_nuts_remaining = 0

        self.create_subscription(String, "/snc_status", self.status_callback, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 20)
        self.create_subscription(PoseArray, "/nuts/uncollected", self.nuts_callback, 10)

        self.create_subscription(Empty, "/sweep_start", self.sweep_start_callback, 10)
        self.create_subscription(Empty, "/sweep_stop", self.sweep_stop_callback, 10)
        self.create_subscription(Empty, "/return_home", self.return_home_callback, 10)

        self.create_summary_file_if_needed()

        self.get_logger().info("Mission summary logger started.")
        self.get_logger().info(f"Summary file: {self.summary_path}")

    def create_summary_file_if_needed(self):
        if not os.path.exists(self.summary_path):
            with open(self.summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "mission_start",
                    "mission_end",
                    "duration_seconds",
                    "final_status",
                    "state_changes",
                    "avoid_events",
                    "emergency_stop_events",
                    "missed_nuts_remaining",
                    "return_home_triggered",
                    "return_home_completed",
                    "final_x",
                    "final_y",
                    "final_yaw_deg",
                ])

    def reset_mission_values(self):
        self.start_time = datetime.now()
        self.end_time = None

        self.final_status = "MISSION_STARTED"
        self.last_status = "MISSION_STARTED"

        self.state_changes = 0
        self.avoid_events = 0
        self.emergency_stop_events = 0
        self.return_home_triggered = False
        self.return_home_completed = False

        self.final_x = None
        self.final_y = None
        self.final_yaw_deg = None
        self.missed_nuts_remaining = 0

    def status_callback(self, msg):
        status = msg.data
        self.final_status = status

        if status != self.last_status:
            self.state_changes += 1
            self.last_status = status

        upper_status = status.upper()

        if "AVOID" in upper_status or "TOO CLOSE" in upper_status:
            self.avoid_events += 1

        if "EMERGENCY STOP" in upper_status:
            self.emergency_stop_events += 1

        if "RETURN_HOME" in upper_status or "RETURN HOME" in upper_status:
            self.return_home_triggered = True

        if "HOME" in upper_status and (
            "COMPLETE" in upper_status
            or "REACHED" in upper_status
            or "DONE" in upper_status
        ):
            self.return_home_completed = True

    def odom_callback(self, msg):
        self.final_x = msg.pose.pose.position.x
        self.final_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.final_yaw_deg = math.degrees(yaw)

    def nuts_callback(self, msg):
        self.missed_nuts_remaining = len(msg.poses)

    def sweep_start_callback(self, _msg):
        self.mission_active = True
        self.reset_mission_values()
        self.get_logger().info("Mission started. Summary logging active.")

    def sweep_stop_callback(self, _msg):
        if self.mission_active:
            self.final_status = "SWEEP_STOPPED"
            self.finish_and_save_summary()

    def return_home_callback(self, _msg):
        self.return_home_triggered = True

    def finish_and_save_summary(self):
        self.end_time = datetime.now()

        duration_seconds = None
        if self.start_time is not None:
            duration_seconds = round((self.end_time - self.start_time).total_seconds(), 2)

        with open(self.summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.start_time.isoformat(timespec="seconds") if self.start_time else None,
                self.end_time.isoformat(timespec="seconds"),
                duration_seconds,
                self.final_status,
                self.state_changes,
                self.avoid_events,
                self.emergency_stop_events,
                self.missed_nuts_remaining,
                self.return_home_triggered,
                self.return_home_completed,
                self.final_x,
                self.final_y,
                self.final_yaw_deg,
            ])

        self.get_logger().info("Mission summary saved.")
        self.get_logger().info(f"Final status: {self.final_status}")
        self.get_logger().info(f"Duration: {duration_seconds} seconds")
        self.get_logger().info(f"Missed nuts remaining: {self.missed_nuts_remaining}")

        self.mission_active = False

    def destroy_node(self):
        # Save a summary if the node is stopped while a mission is still active.
        if self.mission_active:
            self.finish_and_save_summary()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionSummaryLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
