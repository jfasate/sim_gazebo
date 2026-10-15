#!/usr/bin/env python3
"""Publish a rolling local horizon sliced from a reference line.

Non-ML stand-in for the neural trajectory generator. Each tick it finds the
car's nearest point on a (stored) line and republishes the next ~window_m of
points as an OPEN local horizon on /planning/ref_path. This is the horizon
baseline for the LPV-MPC (ref_source='topic'): it proves the local-horizon
follower matches the full-line baseline, and neural_generator later replaces it
behind the same (N,6) [s,x,y,psi,kappa,vx] message contract.
"""

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray


def resolve_csv(ref):
    """Resolve the reference-line path like the MPC node: a bare name ->
    src/csv_data/<name>.csv; a value containing '/' (or an absolute/~ path) is
    used directly."""
    ref = str(ref)
    if '/' in ref:
        return os.path.abspath(os.path.expanduser(ref))
    if not ref.endswith('.csv'):
        ref += '.csv'
    return os.path.join(os.path.abspath(os.path.join('src', 'csv_data')), ref)


class WindowRepublisher(Node):
    def __init__(self):
        super().__init__('window_republisher')
        self.declare_parameter('reference_csv', 'test_worldv5')
        self.declare_parameter('ref_path_topic', '/planning/ref_path')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        # Arc-length published ahead of the car. Must exceed the MPC horizon
        # reach (hz*Ts*v_max ~= 9 m) so _build_reference never clamps early.
        self.declare_parameter('window_m', 12.0)
        self.declare_parameter('publish_rate_hz', 30.0)

        self.ref_path_topic = str(self.get_parameter('ref_path_topic').value)
        data = np.loadtxt(resolve_csv(self.get_parameter('reference_csv').value),
                          delimiter=';', skiprows=0)
        if data.ndim != 2 or data.shape[1] < 6:
            raise ValueError(f'reference needs >= 6 columns, got {data.shape}')
        self.data = data[:, :6].astype(float)      # [s,x,y,psi,kappa,vx]
        self.xy = self.data[:, 1:3]
        self.n = self.xy.shape[0]
        # Closed-loop segment lengths (point i -> i+1, wrapping) for the walk.
        self.seg = np.linalg.norm(np.roll(self.xy, -1, axis=0) - self.xy, axis=1)
        self.window_m = float(self.get_parameter('window_m').value)

        self._car = None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        odom_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._pub = self.create_publisher(Float64MultiArray, self.ref_path_topic, qos)
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._odom_cb, odom_qos)
        rate = float(self.get_parameter('publish_rate_hz').value)
        self.create_timer(1.0 / max(rate, 1e-3), self._tick)
        self.get_logger().info(
            f'window_republisher: {self.n} pts, window={self.window_m:.1f} m '
            f'-> {self.ref_path_topic} @ {rate:.0f} Hz (waiting for odom)')

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self._car = (p.x, p.y)

    def _window_indices(self, start):
        """Walk forward (wrapping the closed loop) from `start` until window_m of
        arc length is covered. The points stay geometrically continuous across the
        wrap; the MPC just treats the result as an open horizon."""
        idxs = [start]
        acc = 0.0
        i = start
        while acc < self.window_m and len(idxs) < self.n:
            acc += self.seg[i]
            i = (i + 1) % self.n
            idxs.append(i)
        return idxs

    def _tick(self):
        if self._car is None:
            return
        d2 = np.sum((self.xy - np.asarray(self._car)) ** 2, axis=1)
        idxs = self._window_indices(int(np.argmin(d2)))
        msg = Float64MultiArray()
        msg.data = self.data[idxs, :].ravel().tolist()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WindowRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
