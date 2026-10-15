#!/usr/bin/env python3
"""Configurable transport-delay relay for sensor topics.

Subscribes a sensor topic and republishes each message after a fixed delay,
modelling sensor + transport + driver latency that the clean Gazebo pipeline
lacks. Insert one instance per sensor stream you want to delay.

Parameters:
    msg_type      : one of {LaserScan, Imu, Image, CameraInfo, Odometry}
    input_topic   : topic to subscribe (the raw, zero-latency stream)
    output_topic  : topic to republish on (what the stack should consume)
    delay         : delay [s]

Example (launch a delayed /scan):
    ros2 run f1tenth_gym_gazebo sensor_delay --ros-args \
        -p msg_type:=LaserScan -p input_topic:=/scan_raw \
        -p output_topic:=/scan -p delay:=0.03
"""

from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan, Imu, Image, CameraInfo
from nav_msgs.msg import Odometry

MSG_TYPES = {
    'LaserScan': LaserScan,
    'Imu': Imu,
    'Image': Image,
    'CameraInfo': CameraInfo,
    'Odometry': Odometry,
}


class SensorDelay(Node):

    def __init__(self):
        super().__init__('sensor_delay')

        self.declare_parameter('msg_type', 'LaserScan')
        self.declare_parameter('input_topic', '/scan_raw')
        self.declare_parameter('output_topic', '/scan')
        self.declare_parameter('delay', 0.03)

        type_name = self.get_parameter('msg_type').value
        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value
        self.delay = float(self.get_parameter('delay').value)

        if type_name not in MSG_TYPES:
            raise ValueError(
                f'msg_type "{type_name}" not in {list(MSG_TYPES)}')
        msg_cls = MSG_TYPES[type_name]

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._buf = deque()
        self.pub = self.create_publisher(msg_cls, out_topic, qos)
        self.sub = self.create_subscription(
            msg_cls, in_topic, self._cb, qos)

        # Poll fast enough to preserve the source rate.
        self.timer = self.create_timer(0.002, self._flush)

        self.get_logger().info(
            f'sensor_delay: {in_topic} -> {out_topic} '
            f'({type_name}, delay={self.delay*1000:.0f} ms)')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb(self, msg):
        self._buf.append((self._now(), msg))

    def _flush(self):
        t_release = self._now() - self.delay
        while self._buf and self._buf[0][0] <= t_release:
            _, msg = self._buf.popleft()
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorDelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
