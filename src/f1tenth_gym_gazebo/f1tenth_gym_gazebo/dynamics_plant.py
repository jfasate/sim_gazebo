#!/usr/bin/env python3
"""F1Tenth dynamics plant node — co-simulation with Gazebo Classic.

Replaces the kinematic `planar_move` plugin and the `ackermann_to_twist` bridge
with a real double-track vehicle model that uses combined-slip Pacejka tyres
(see f1tenth_gym_gazebo.dynamics). Gazebo is used purely as the sensor renderer
(LiDAR / camera / IMU raycast against the world at the car's pose); the vehicle
motion is integrated here and written back to Gazebo each tick.

Data flow
---------
    /drive (AckermannDriveStamped)  -->  [ double-track + Pacejka, RK4 ]
        --> /set_entity_state (move car in Gazebo)
        --> /ego_racecar/odom (nav_msgs/Odometry)
        --> /joint_states (steer + wheel spin, for RViz)
        --> /set_joint_trajectory (same angles, for the Gazebo model)
        --> TF: map -> odom (identity) , odom -> base_link (integrated)

    /initialpose (RViz 2D Pose Estimate)  -->  reset integrator state.

The car's links must have gravity disabled in the URDF so Gazebo physics does
not fight the prescribed pose.
"""

import math
import random
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import tf2_ros

from f1tenth_gym_gazebo.dynamics.vehicle import (
    DoubleTrackModel, VehicleParams, ackermann_angles,
)


def yaw_to_quat(yaw):
    """Return (x, y, z, w) for a yaw-only rotation."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def rpy_to_quat(roll, pitch, yaw):
    """Return (x, y, z, w) for roll/pitch/yaw (ZYX)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def quat_to_yaw(q):
    """Extract yaw from quaternion (x, y, z, w)."""
    return math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                      1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


class DynamicsPlant(Node):

    def __init__(self):
        super().__init__('dynamics_plant')

        # -- parameters --
        self.declare_parameter('entity_name', 'f1tenth')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('base_frame', 'ego_racecar/base_link')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('rate', 100.0)          # integration/publish Hz
        self.declare_parameter('substeps', 4)          # RK4 substeps per tick
        self.declare_parameter('spawn_z', 0.0)
        # command (comms + compute) latency applied to /drive [s]
        self.declare_parameter('command_latency', 0.0)
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('vibration_pitch_deg', 0.0)
        self.declare_parameter('vibration_roll_deg', 0.0)
        self.declare_parameter('vibration_freq_hz', 2.0)
        self.declare_parameter('vibration_heave_mm', 0.0)
        self.declare_parameter('vibration_speed_ref', 2.0)
        # The plant prescribes pose via /set_entity_state, so Gazebo contact
        # physics cannot block walls. This map guard makes occupied cells act
        # like hard barriers for the integrated vehicle state.
        self.declare_parameter('collision_guard_enabled', True)
        self.declare_parameter('collision_map_topic', '/map')
        self.declare_parameter('collision_unknown_is_occupied', True)
        self.declare_parameter('collision_occupancy_threshold', 50)
        self.declare_parameter('collision_check_step_m', 0.025)
        self.declare_parameter('collision_footprint_length', 0.37)
        self.declare_parameter('collision_footprint_width', 0.22)
        self.declare_parameter('collision_rear_overhang', 0.02)

        self.entity_name = self.get_parameter('entity_name').value
        self.base_frame = self.get_parameter('base_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.rate = float(self.get_parameter('rate').value)
        self.substeps = int(self.get_parameter('substeps').value)
        self.spawn_z = float(self.get_parameter('spawn_z').value)
        # Chassis vibration (visual pose only: camera + lidar wobble, the planar
        # dynamics are untouched). Amplitude [deg] at >= vibration_speed_ref m/s,
        # scaled down linearly to 0 at standstill.
        self.vib_pitch = math.radians(float(self.get_parameter('vibration_pitch_deg').value))
        self.vib_roll = math.radians(float(self.get_parameter('vibration_roll_deg').value))
        self.vib_freq = float(self.get_parameter('vibration_freq_hz').value)
        self.vib_heave = float(self.get_parameter('vibration_heave_mm').value) / 1000.0
        self.vib_vref = max(1e-3, float(self.get_parameter('vibration_speed_ref').value))
        self.vib_phase = [random.uniform(0.0, 2.0 * math.pi) for _ in range(6)]

        # -- vehicle model --
        self.model = DoubleTrackModel(VehicleParams())

        # -- state: [X, Y, psi, vx, vy, r, delta, Fx_drive] --
        self.state = [
            float(self.get_parameter('x').value),
            float(self.get_parameter('y').value),
            float(self.get_parameter('yaw').value),
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        # wheel spin accumulator (visual only)
        self.wheel_angle = 0.0
        self.wheel_radius = 0.0508

        # -- command handling (with latency buffer) --
        self.command_latency = float(self.get_parameter('command_latency').value)
        self.v_des = 0.0
        self.delta_cmd = 0.0
        # buffer of (t_recv_sec, v_des, delta_cmd) for latency modelling
        self._cmd_buf = deque(maxlen=2048)

        # -- occupancy-map collision guard --
        self.collision_guard_enabled = bool(
            self.get_parameter('collision_guard_enabled').value)
        self.collision_unknown_is_occupied = bool(
            self.get_parameter('collision_unknown_is_occupied').value)
        self.collision_occupancy_threshold = int(
            self.get_parameter('collision_occupancy_threshold').value)
        self.collision_check_step_m = max(
            0.005, float(self.get_parameter('collision_check_step_m').value))
        self._map_data = None
        self._map_w = 0
        self._map_h = 0
        self._map_res = 0.0
        self._map_origin = (0.0, 0.0, 0.0)
        self._collision_count = 0
        self._last_collision_log_t = -1e9
        self._footprint_points = self._build_collision_footprint()

        # -- interfaces --
        self.drive_sub = self.create_subscription(
            AckermannDriveStamped,
            self.get_parameter('drive_topic').value,
            self.drive_cb, 10)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.get_parameter('collision_map_topic').value,
            self.map_cb,
            map_qos)

        odom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_pub = self.create_publisher(
            Odometry, self.get_parameter('odom_topic').value, odom_qos)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory, '/set_joint_trajectory', 1)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_tf = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_static_map_odom()

        self.set_state_cli = self.create_client(
            SetEntityState, '/set_entity_state')
        # diagnostics for the Gazebo write path (so silent failures surface)
        self._no_service_warned_t = -1e9
        self._push_fail_warned = False
        self._push_ok_logged = False

        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info(
            f'Dynamics plant up: double-track + combined-slip Pacejka @ '
            f'{self.rate:.0f} Hz, entity="{self.entity_name}"')

    # -- callbacks ----------------------------------------------------------
    def drive_cb(self, msg):
        v = float(msg.drive.speed)
        d = float(msg.drive.steering_angle)
        if self.command_latency <= 0.0:
            self.v_des, self.delta_cmd = v, d
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        self._cmd_buf.append((t, v, d))

    def map_cb(self, msg):
        self._map_w = int(msg.info.width)
        self._map_h = int(msg.info.height)
        self._map_res = float(msg.info.resolution)
        self._map_origin = (
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
            quat_to_yaw((
                msg.info.origin.orientation.x,
                msg.info.origin.orientation.y,
                msg.info.origin.orientation.z,
                msg.info.origin.orientation.w,
            )),
        )
        self._map_data = tuple(int(v) for v in msg.data)

    def _resolve_delayed_command(self):
        """Pick the most recent command older than command_latency."""
        if self.command_latency <= 0.0:
            return
        t_now = self.get_clock().now().nanoseconds * 1e-9
        t_apply = t_now - self.command_latency
        while self._cmd_buf and self._cmd_buf[0][0] <= t_apply:
            _, v, d = self._cmd_buf.popleft()
            self.v_des, self.delta_cmd = v, d

    def initialpose_cb(self, msg):
        """RViz '2D Pose Estimate' -> teleport the car in BOTH RViz and Gazebo.

        Resets the integrator state to the clicked pose (this drives odom + TF,
        i.e. the RViz view) and immediately writes it to Gazebo via
        /set_entity_state, so the physical model + its sensors jump there too.
        """
        p = msg.pose.pose
        q = (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        yaw = quat_to_yaw(q)
        self.state = [p.position.x, p.position.y, yaw,
                      0.0, 0.0, 0.0, 0.0, 0.0]
        self.v_des = 0.0
        self.delta_cmd = 0.0
        # Teleport Gazebo now rather than waiting for the next tick.
        self._push_to_gazebo()
        self.get_logger().info(
            f'2D Pose Estimate -> teleport "{self.entity_name}" to '
            f'x={p.position.x:.2f} y={p.position.y:.2f} yaw={yaw:.2f} '
            f'(RViz + Gazebo)')

    # -- main loop ----------------------------------------------------------
    def update(self):
        self._resolve_delayed_command()
        prev_state = list(self.state)
        next_state = self.model.step(
            self.state, self.v_des, self.delta_cmd, self.dt, self.substeps)
        self.state = self._apply_collision_guard(prev_state, next_state)

        # accumulate wheel spin for visualization
        vx = self.state[3]
        self.wheel_angle += (vx / self.wheel_radius) * self.dt

        now = self.get_clock().now().to_msg()
        self._push_to_gazebo()
        self._publish_odom(now)
        self._publish_joints(now)
        self._publish_tf(now)

    # -- collision guard ----------------------------------------------------
    def _build_collision_footprint(self):
        length = max(0.05, float(
            self.get_parameter('collision_footprint_length').value))
        width = max(0.05, float(
            self.get_parameter('collision_footprint_width').value))
        rear = max(0.0, min(length, float(
            self.get_parameter('collision_rear_overhang').value)))
        front = length - rear
        step = 0.05
        points = []
        x = -rear
        while x <= front + 1.0e-9:
            y = -0.5 * width
            while y <= 0.5 * width + 1.0e-9:
                points.append((x, y))
                y += step
            x += step
        for x, y in [(-rear, -0.5 * width), (-rear, 0.5 * width),
                     (front, -0.5 * width), (front, 0.5 * width)]:
            if (x, y) not in points:
                points.append((x, y))
        return points

    def _map_cell_occupied(self, wx, wy):
        if self._map_data is None or self._map_res <= 0.0:
            return False
        ox, oy, oyaw = self._map_origin
        dx = wx - ox
        dy = wy - oy
        c = math.cos(-oyaw)
        s = math.sin(-oyaw)
        mx_f = (c * dx - s * dy) / self._map_res
        my_f = (s * dx + c * dy) / self._map_res
        mx = int(math.floor(mx_f))
        my = int(math.floor(my_f))
        if mx < 0 or my < 0 or mx >= self._map_w or my >= self._map_h:
            return True
        occ = self._map_data[my * self._map_w + mx]
        if occ < 0:
            return self.collision_unknown_is_occupied
        return occ >= self.collision_occupancy_threshold

    def _pose_collides(self, x, y, yaw):
        c = math.cos(yaw)
        s = math.sin(yaw)
        for px, py in self._footprint_points:
            wx = x + c * px - s * py
            wy = y + s * px + c * py
            if self._map_cell_occupied(wx, wy):
                return True
        return False

    @staticmethod
    def _interp_angle(a, b, t):
        d = math.atan2(math.sin(b - a), math.cos(b - a))
        return a + t * d

    def _apply_collision_guard(self, prev_state, next_state):
        if (not self.collision_guard_enabled
                or self._map_data is None
                or self._map_res <= 0.0):
            return next_state

        x0, y0, yaw0 = prev_state[:3]
        x1, y1, yaw1 = next_state[:3]
        dist = math.hypot(x1 - x0, y1 - y0)
        checks = max(1, int(math.ceil(dist / self.collision_check_step_m)))
        last_free = list(prev_state)

        for i in range(1, checks + 1):
            a = i / checks
            trial = list(next_state)
            trial[0] = x0 + (x1 - x0) * a
            trial[1] = y0 + (y1 - y0) * a
            trial[2] = self._interp_angle(yaw0, yaw1, a)
            if self._pose_collides(trial[0], trial[1], trial[2]):
                stopped = list(last_free)
                stopped[3] = 0.0
                stopped[4] = 0.0
                stopped[5] = 0.0
                stopped[7] = 0.0
                self.v_des = 0.0
                self._cmd_buf.clear()
                self._collision_count += 1
                self._log_collision(stopped)
                return stopped
            last_free = trial

        return next_state

    def _log_collision(self, stopped_state):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t - self._last_collision_log_t < 1.0:
            return
        self._last_collision_log_t = t
        self.get_logger().warn(
            f'collision guard blocked wall crossing #{self._collision_count}: '
            f'x={stopped_state[0]:.2f} y={stopped_state[1]:.2f}; '
            'velocity zeroed')

    # -- outputs ------------------------------------------------------------
    def _push_to_gazebo(self):
        """Write the integrated pose to the Gazebo model via /set_entity_state.

        This is what makes the car (and its sensors) actually move in Gazebo —
        including teleporting on a 2D Pose Estimate. Failures are surfaced
        (throttled) because a silent failure looks like "moves in RViz but not
        in Gazebo": the usual cause is the entity not being spawned.
        """
        if not self.set_state_cli.service_is_ready():
            t = self.get_clock().now().nanoseconds * 1e-9
            if t - self._no_service_warned_t > 5.0:
                self.get_logger().warn(
                    '/set_entity_state not available yet — car will not move '
                    'in Gazebo until it is (is gzserver up with '
                    'libgazebo_ros_state.so?)')
                self._no_service_warned_t = t
            return
        X, Y, psi, vx, vy, r, delta, Fx_drive = self.state
        pitch, roll, heave = self._vibration(vx)
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, psi)
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.entity_name
        st.reference_frame = 'world'
        st.pose.position.x = X
        st.pose.position.y = Y
        st.pose.position.z = self.spawn_z + heave
        st.pose.orientation.x = qx
        st.pose.orientation.y = qy
        st.pose.orientation.z = qz
        st.pose.orientation.w = qw
        # world-frame linear velocity (for smoother sensor motion)
        st.twist.linear.x = vx * math.cos(psi) - vy * math.sin(psi)
        st.twist.linear.y = vx * math.sin(psi) + vy * math.cos(psi)
        st.twist.angular.z = r
        req.state = st
        fut = self.set_state_cli.call_async(req)
        fut.add_done_callback(self._push_done)

    def _vibration(self, vx):
        """Chassis pitch/roll [rad] and heave [m] (vertical bounce: camera + lidar move up
        and down): two sines per axis (f and 1.9 f, random phases), amplitude scaled by
        speed. Pose is written at `rate` (100 Hz), so f up to ~30 Hz is represented; the
        camera samples it at its own frame rate (aliasing, as on a real camera)."""
        if self.vib_pitch == 0.0 and self.vib_roll == 0.0 and self.vib_heave == 0.0:
            return 0.0, 0.0, 0.0
        t = self.get_clock().now().nanoseconds * 1e-9
        w = 2.0 * math.pi * self.vib_freq
        k = min(1.0, abs(vx) / self.vib_vref)
        ph = self.vib_phase
        pitch = k * self.vib_pitch * (0.7 * math.sin(w * t + ph[0]) + 0.3 * math.sin(1.9 * w * t + ph[1]))
        roll = k * self.vib_roll * (0.7 * math.sin(w * t + ph[2]) + 0.3 * math.sin(1.9 * w * t + ph[3]))
        heave = k * self.vib_heave * (0.7 * math.sin(w * t + ph[4]) + 0.3 * math.sin(1.9 * w * t + ph[5]))
        return pitch, roll, heave

    def _push_done(self, fut):
        try:
            res = fut.result()
        except Exception as e:                       # noqa: BLE001
            if not self._push_fail_warned:
                self.get_logger().error(f'/set_entity_state call failed: {e}')
                self._push_fail_warned = True
            return
        if not res.success:
            if not self._push_fail_warned:
                self.get_logger().warn(
                    f'/set_entity_state rejected entity "{self.entity_name}" — '
                    f'the car is not moving in Gazebo. Check the spawned model '
                    f'name matches entity_name.')
                self._push_fail_warned = True
        else:
            # Reset the warn latch so a transient startup failure (before the
            # entity finished spawning) does not permanently mute later ones.
            self._push_fail_warned = False
            if not self._push_ok_logged:
                self.get_logger().info(
                    f'Gazebo write path OK — "{self.entity_name}" is tracking '
                    f'the plant (RViz and Gazebo stay in sync).')
                self._push_ok_logged = True

    def _publish_odom(self, stamp):
        X, Y, psi, vx, vy, r, delta, Fx_drive = self.state
        qx, qy, qz, qw = yaw_to_quat(psi)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = X
        odom.pose.pose.position.y = Y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = r
        self.odom_pub.publish(odom)

    def _publish_joints(self, stamp):
        d_fl, d_fr = ackermann_angles(self.state[6], self.model.p)
        js = JointState()
        js.header.stamp = stamp
        js.name = [
            'base_to_front_left_hinge', 'base_to_front_right_hinge',
            'front_left_hinge_to_wheel', 'front_right_hinge_to_wheel',
            'base_to_back_left_wheel', 'base_to_back_right_wheel',
        ]
        w = self.wheel_angle
        js.position = [d_fl, d_fr, w, w, w, w]
        self.joint_pub.publish(js)

        jt = JointTrajectory()
        jt.header.stamp = stamp
        jt.header.frame_id = self.base_frame
        jt.joint_names = js.name
        pt = JointTrajectoryPoint()
        pt.positions = js.position
        jt.points = [pt]
        self.joint_traj_pub.publish(jt)

    def _publish_tf(self, stamp):
        X, Y, psi, *_ = self.state
        qx, qy, qz, qw = yaw_to_quat(psi)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = X
        t.transform.translation.y = Y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def _publish_static_map_odom(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicsPlant()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
