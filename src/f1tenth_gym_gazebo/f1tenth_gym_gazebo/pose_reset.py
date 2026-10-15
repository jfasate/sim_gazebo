#!/usr/bin/env python3
"""
Pose Reset node for Gazebo — handles TF and 2D Pose Estimate from RViz.

TF architecture (matches f1tenth_gym_ros):
  - Publishes map -> ego_racecar/base_link directly on every odom message
  - Single source of truth — no two-publisher timing race
  - diff_drive plugin publishes odom topic only (publish_odom_tf: false)

When the user clicks "2D Pose Estimate" in RViz:
  1. Teleports the robot in Gazebo via /set_entity_state
  2. Updates the map -> odom correction so the TF stays consistent
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import tf2_ros


def quaternion_inverse(q):
    """Return inverse of quaternion [x, y, z, w]."""
    norm_sq = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    if norm_sq < 1e-10:
        return [0.0, 0.0, 0.0, 1.0]
    return [-q[0]/norm_sq, -q[1]/norm_sq, -q[2]/norm_sq, q[3]/norm_sq]


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]


def rotate_point_by_quaternion(point, q):
    """Rotate a 3D point by quaternion q [x,y,z,w]."""
    p = [point[0], point[1], point[2], 0.0]
    q_inv = quaternion_inverse(q)
    result = quaternion_multiply(quaternion_multiply(q, p), q_inv)
    return result[:3]


class PoseReset(Node):

    def __init__(self):
        super().__init__('pose_reset')

        self.declare_parameter('entity_name', 'f1tenth')
        self.entity_name = self.get_parameter('entity_name').value

        # Latest odom pose (for pose reset math)
        self._last_odom = None

        # map -> odom correction (initially identity).
        # This only changes when the user does a 2D Pose Estimate.
        self.map_to_odom_translation = [0.0, 0.0, 0.0]
        self.map_to_odom_rotation = [0.0, 0.0, 0.0, 1.0]  # xyzw

        # Single TF broadcaster — publishes map -> base_link directly,
        # just like f1tenth_gym_ros's gym_bridge.
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribe to odom from diff_drive (topic only, no TF)
        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            odom_qos,
        )

        # Subscribe to 2D Pose Estimate from RViz
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initialpose_callback,
            10,
        )

        # Gazebo set_entity_state client
        self.client = self.create_client(
            SetEntityState,
            '/set_entity_state',
        )

        self.get_logger().info(
            f'Pose reset ready: /initialpose -> teleport "{self.entity_name}" '
            f'| publishing map -> base_link from odom'
        )

    def odom_callback(self, msg):
        """Compute map -> base_link from odom data + correction, publish TF."""
        self._last_odom = msg.pose.pose  # store for pose reset math

        odom_pos = msg.pose.pose.position
        odom_ori = msg.pose.pose.orientation
        stamp = msg.header.stamp

        odom_q = [odom_ori.x, odom_ori.y, odom_ori.z, odom_ori.w]

        # map_rotation = map_to_odom_rotation * odom_rotation
        map_q = quaternion_multiply(self.map_to_odom_rotation, odom_q)

        # map_position = map_to_odom_rotation * odom_position + map_to_odom_translation
        rotated_pos = rotate_point_by_quaternion(
            [odom_pos.x, odom_pos.y, odom_pos.z],
            self.map_to_odom_rotation,
        )
        map_x = self.map_to_odom_translation[0] + rotated_pos[0]
        map_y = self.map_to_odom_translation[1] + rotated_pos[1]

        # Publish map -> ego_racecar/base_link
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'ego_racecar/base_link'
        t.transform.translation.x = map_x
        t.transform.translation.y = map_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = map_q[0]
        t.transform.rotation.y = map_q[1]
        t.transform.rotation.z = map_q[2]
        t.transform.rotation.w = map_q[3]
        self.tf_broadcaster.sendTransform(t)

    def initialpose_callback(self, msg):
        """Teleport robot in Gazebo + update map->odom correction."""
        desired = msg.pose.pose

        # 1. Teleport entity in Gazebo
        if self.client.service_is_ready():
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = self.entity_name
            req.state.pose = desired
            req.state.pose.position.z = 0.0
            req.state.reference_frame = 'world'
            future = self.client.call_async(req)
            future.add_done_callback(self._teleport_done)
        else:
            self.get_logger().warn('Waiting for /set_entity_state service...')
            return

        # 2. Recompute map -> odom correction
        #    desired = map -> base_link (what we want)
        #    odom_callback will apply: map_pos = correction * odom_pos
        #    After teleport, diff_drive odom doesn't reset, so we need
        #    to find the correction that maps current odom -> desired pose.
        #
        #    We read the latest odom from the last message. The diff_drive
        #    odom won't change instantly (teleport doesn't reset it), so
        #    the correction computed here will be correct going forward.
        desired_q = [
            desired.orientation.x,
            desired.orientation.y,
            desired.orientation.z,
            desired.orientation.w,
        ]

        if self._last_odom is not None:
            odom_q = [
                self._last_odom.orientation.x,
                self._last_odom.orientation.y,
                self._last_odom.orientation.z,
                self._last_odom.orientation.w,
            ]

            odom_q_inv = quaternion_inverse(odom_q)
            self.map_to_odom_rotation = quaternion_multiply(desired_q, odom_q_inv)

            odom_pos = [
                self._last_odom.position.x,
                self._last_odom.position.y,
                self._last_odom.position.z,
            ]
            rotated = rotate_point_by_quaternion(odom_pos, self.map_to_odom_rotation)
            self.map_to_odom_translation = [
                desired.position.x - rotated[0],
                desired.position.y - rotated[1],
                0.0,
            ]
        else:
            # No odom yet — assume odom is at origin
            self.map_to_odom_translation = [
                desired.position.x,
                desired.position.y,
                0.0,
            ]
            self.map_to_odom_rotation = desired_q

        self.get_logger().info(
            f'Teleporting to x={desired.position.x:.2f}, '
            f'y={desired.position.y:.2f} | map->odom corrected'
        )

    def _teleport_done(self, future):
        try:
            result = future.result()
            if not result.success:
                self.get_logger().warn('Gazebo teleport failed')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PoseReset()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
