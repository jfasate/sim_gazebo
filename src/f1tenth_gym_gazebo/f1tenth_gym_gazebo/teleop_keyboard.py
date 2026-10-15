#!/usr/bin/env python3
#
# F1Tenth keyboard teleop with independent, persistent throttle and steering.
#
# - Press 'i' once: car drives forward until you press 'k' or ','
# - Press 'j' once: car steers left until you press 'l' or 'k'
# - Throttle and steering are independent, just like a real car.
#
# Usage: ros2 run f1tenth_gym_gazebo teleop_keyboard

import sys
import select
import threading

from ackermann_msgs.msg import AckermannDriveStamped
import rclpy
from rclpy.node import Node

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

MSG = """
F1Tenth Teleop Keyboard
---------------------------
Throttle (persists until changed):
   i : drive forward
   , : drive reverse
   k : stop and center steering

Steering (persists until changed):
   j : steer left
   l : steer right

Speed adjustment:
   q/z : increase/decrease max speed by 10%
   e/c : increase/decrease max steering by 10%

Commands are persistent: press once and the car keeps going.
CTRL-C to quit
"""


class TeleopKeyboard(Node):

    def __init__(self):
        super().__init__('teleop_keyboard')
        self.settings = termios.tcgetattr(sys.stdin)

        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_steering', 0.34)
        self.declare_parameter('update_frequency', 10.0)

        drive_topic = self.get_parameter('drive_topic').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering = self.get_parameter('max_steering').value
        self.keyboard_timeout = 1.0 / self.get_parameter('update_frequency').value

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.get_logger().info(
            f'Teleop keyboard -> {drive_topic} '
            f'(max_speed={self.max_speed}, max_steering={self.max_steering:.2f})'
        )

    def get_key(self):
        if sys.platform == 'win32':
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            res, _, _ = select.select([sys.stdin], [], [], self.keyboard_timeout)
            key = '' if not res else sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = speed
        msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(msg)

    def run(self):
        speed = 0.0
        steering = 0.0
        try:
            print(MSG)
            print(f'max speed: {self.max_speed:.2f} m/s   max steering: {self.max_steering:.2f} rad')
            while rclpy.ok():
                key = self.get_key()

                if key == 'i':
                    speed = self.max_speed
                    print(f'max speed: {self.max_speed:.2f} m/s   max steering: {self.max_steering:.2f} rad')
                elif key == ',':
                    speed = -self.max_speed
                elif key == 'j':
                    steering = self.max_steering
                elif key == 'l':
                    steering = -self.max_steering
                elif key == 'k':
                    speed = 0.0
                    steering = 0.0
                elif key == 'q':
                    self.max_speed *= 1.1
                    print(f'max speed: {self.max_speed:.2f} m/s   max steering: {self.max_steering:.2f} rad')
                elif key == 'z':
                    self.max_speed *= 0.9
                    print(f'max speed: {self.max_speed:.2f} m/s   max steering: {self.max_steering:.2f} rad')
                elif key == 'e':
                    self.max_steering *= 1.1
                    print(f'max speed: {self.max_speed:.2f} m/s   max steering: {self.max_steering:.2f} rad')
                elif key == 'c':
                    self.max_steering *= 0.9
                    print(f'max speed: {self.max_speed:.2f} m/s   max steering: {self.max_steering:.2f} rad')
                elif key == '\x03':
                    break

                # Publish continuously at update_frequency, regardless of key press
                self.publish_drive(speed, steering)

        except Exception as e:
            print(e)

    def shutdown(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        self.publish_drive(0.0, 0.0)


def main():
    rclpy.init()
    node = TeleopKeyboard()
    spinner = threading.Thread(target=rclpy.spin, args=(node,))
    spinner.start()
    node.run()
    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    spinner.join()


if __name__ == '__main__':
    main()
