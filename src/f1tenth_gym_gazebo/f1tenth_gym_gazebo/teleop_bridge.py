#!/usr/bin/env python3
"""
Bridge node to convert teleop Float32 commands to AckermannDriveStamped.

Subscribes to:
  - /accelerator_cmd (Float32): forward/backward throttle [-1, 1]
  - /steering_cmd (Float32): steering angle [-1, 1]

Publishes to:
  - /drive (AckermannDriveStamped): standard F1Tenth drive command

This allows the teleop_keyboard node to work with the ackermann_to_twist bridge.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class TeleopBridge(Node):
    def __init__(self):
        super().__init__('teleop_bridge')

        # Parameters
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_steering', 0.34) 
        self.declare_parameter('accelerator_cmd_topic', 'accelerator_cmd')
        self.declare_parameter('steering_cmd_topic', 'steering_cmd')
        self.declare_parameter('drive_topic', '/drive')

        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering = self.get_parameter('max_steering').value
        accel_topic = self.get_parameter('accelerator_cmd_topic').value
        steer_topic = self.get_parameter('steering_cmd_topic').value
        drive_topic = self.get_parameter('drive_topic').value

        # Subscriptions
        self.accel_sub = self.create_subscription(
            Float32, accel_topic, self.accel_callback, 10
        )
        self.steer_sub = self.create_subscription(
            Float32, steer_topic, self.steer_callback, 10
        )

        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )

        # Current command state
        self.accel_cmd = 0.0
        self.steer_cmd = 0.0

        self.get_logger().info(
            f'Teleop Bridge initialized: {accel_topic}, {steer_topic} -> {drive_topic}'
        )

    def accel_callback(self, msg):
        """Handle accelerator command."""
        self.accel_cmd = msg.data
        self.publish_drive()

    def steer_callback(self, msg):
        """Handle steering command."""
        self.steer_cmd = msg.data
        self.publish_drive()

    def publish_drive(self):
        """Publish AckermannDriveStamped command."""
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = 'base_link'

        # Scale commands to physical ranges
        ackermann_cmd.drive.speed = self.accel_cmd * self.max_speed
        ackermann_cmd.drive.steering_angle = self.steer_cmd * self.max_steering

        self.drive_pub.publish(ackermann_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
