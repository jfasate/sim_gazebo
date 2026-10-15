#!/usr/bin/env python3
"""System-ID excitation + logger.

Drives a scripted sequence of maneuvers that make the vehicle parameters
observable, while logging the command and the measured response to CSV. Feed
the CSV to `sysid_fit` to recover VehicleParams.

Runs against the real car OR the sim: publish /drive, record /ego_racecar/odom
(vx, vy, yaw-rate) and /imu (ax, ay). Same pipeline both places.

Maneuvers
---------
  1. Longitudinal steps: accelerate to 2, 4, 6 m/s and brake back down.
     -> observability for k_speed, tau_motor, c_rr, c_drag.
  2. Constant-speed steering steps (increasing amplitude, both signs) at a
     fixed cruise speed.  -> observability for cornering stiffness (B), grip
     limit (mu), understeer.
  3. Steering sine sweep (chirp) at cruise speed.  -> transient / yaw inertia.

CSV columns: t, cmd_v, cmd_delta, vx, vy, r, ax, ay

Parameters:
    csv_path      : output file
    rate          : command + log rate [Hz] (default 50)
    cruise_speed  : speed held during the steering tests [m/s] (default 4.0)
"""

import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from f1tenth_gym_gazebo.dynamics.sysid_common import build_timeline


class SysIdExcite(Node):

    def __init__(self):
        super().__init__('sysid_excite')

        self.declare_parameter('csv_path', '/tmp/sysid_log.csv')
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('cruise_speed', 4.0)

        self.csv_path = self.get_parameter('csv_path').value
        self.rate = float(self.get_parameter('rate').value)
        cruise = float(self.get_parameter('cruise_speed').value)

        self.timeline = build_timeline(self.rate, cruise)
        self.idx = 0

        # latest measurements
        self.vx = self.vy = self.r = 0.0
        self.ax = self.ay = 0.0

        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self._odom, odom_qos)
        self.create_subscription(Imu, '/imu', self._imu, odom_qos)

        self._rows = []
        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self._tick)

        dur = len(self.timeline) * self.dt
        self.get_logger().info(
            f'sysid_excite: {len(self.timeline)} setpoints, ~{dur:.0f} s, '
            f'logging -> {self.csv_path}')

    def _odom(self, msg):
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.r = msg.twist.twist.angular.z

    def _imu(self, msg):
        self.ax = msg.linear_acceleration.x
        self.ay = msg.linear_acceleration.y

    def _tick(self):
        if self.idx >= len(self.timeline):
            self._finish()
            return
        v, d = self.timeline[self.idx]
        self.idx += 1

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(v)
        msg.drive.steering_angle = float(d)
        self.drive_pub.publish(msg)

        t = self.idx * self.dt
        self._rows.append([t, v, d, self.vx, self.vy, self.r, self.ax, self.ay])

    def _finish(self):
        self.timer.cancel()
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'cmd_v', 'cmd_delta', 'vx', 'vy', 'r', 'ax', 'ay'])
            w.writerows(self._rows)
        # stop the car
        stop = AckermannDriveStamped()
        self.drive_pub.publish(stop)
        self.get_logger().info(
            f'Done. Wrote {len(self._rows)} rows -> {self.csv_path}. '
            f'Run: sysid_fit {self.csv_path}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = SysIdExcite()
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    node.destroy_node()


if __name__ == '__main__':
    main()
