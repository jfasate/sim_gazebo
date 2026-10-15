#!/usr/bin/env python3
"""
Ackermann Drive to Twist bridge node.

Subscribes to /drive (AckermannDriveStamped) from the MPC controller
and publishes /cmd_vel (Twist) for the Gazebo Ackermann steering plugin.

This allows the existing MPC nodes to work unchanged — they publish
AckermannDriveStamped as they do with the F1Tenth gym, and this bridge
converts to what Gazebo expects.

Usage:
    ros2 run f1tenth_gym_gazebo ackermann_to_twist
"""

import math
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist


class AckermannToTwist(Node):

    def __init__(self):
        super().__init__('ackermann_to_twist')

        # F1Tenth wheelbase
        self.declare_parameter('wheelbase', 0.3302)
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.wheelbase = self.get_parameter('wheelbase').value
        drive_topic = self.get_parameter('drive_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.sub = self.create_subscription(
            AckermannDriveStamped,
            drive_topic,
            self.drive_callback,
            10,
        )

        self.pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.get_logger().info(
            f'Ackermann-to-Twist bridge: {drive_topic} -> {cmd_vel_topic} '
            f'(wheelbase={self.wheelbase:.4f}m)'
        )

    def drive_callback(self, msg):
        """Convert AckermannDriveStamped to Twist for Gazebo."""
        twist = Twist()

        speed = msg.drive.speed
        steering_angle = msg.drive.steering_angle

        # Linear velocity (forward speed)
        twist.linear.x = float(speed)

        # Angular velocity from bicycle model: omega = v * tan(delta) / L
        if abs(speed) > 0.01:
            twist.angular.z = float(
                speed * math.tan(steering_angle) / self.wheelbase
            )
        else:
            twist.angular.z = 0.0

        self.pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannToTwist()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
