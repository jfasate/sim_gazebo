#!/usr/bin/env python3
"""Dynamic obstacles: prescribed-motion agents driven along a world's centerline.

Spawns N boxes via /spawn_entity, then prescribes their pose every tick via
/set_entity_state — the same "co-sim prescribes pose, physics must not fight it"
pattern the ego car uses. No world edits, no new Gazebo plugins: every world
already declares gazebo_ros_state, and sim_launch already loads
libgazebo_ros_factory.so.

Two behaviours, which is all the scoring teacher needs to have something to say:
  follow  agent drives ALONG the centerline at a lateral offset. Negative speed
          makes it oncoming. This is "another vehicle".
  cross   agent sweeps ACROSS the corridor at a fixed arclength. This is the
          pedestrian that forces a brake and creates occlusion events.

Ground truth is free: gazebo_ros_state already publishes /model_states, so agent
poses and twists come out with the sim. Box extents are the `size` params here.

Usage:
  ros2 run f1tenth_gym_gazebo dynamic_obstacles --ros-args \
      -p centerline:=obstacle_corridor -p n_follow:=3 -p n_cross:=2
"""

import math
import os
import random

import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState, SpawnEntity


# ---------------------------------------------------------------- pure helpers

def resolve_centerline(ref):
    """Bare name -> src/csv_data/<name>_centerline.csv; a value with '/' is a path.

    Matches the reference-resolution convention in lpv_mpc_node / window_republisher.
    """
    if '/' in ref or ref.startswith('~'):
        return os.path.abspath(os.path.expanduser(ref))
    return os.path.join(os.path.abspath(os.path.join('src', 'csv_data')),
                        ref + '_centerline.csv')


def load_centerline(path):
    """'#'-commented, ';'-separated: s_m; x_m; y_m; psi_rad; kappa; vx; ax; wr; wl.

    Returns (s, x, y, psi, w_right, w_left) as float arrays, s strictly increasing.
    """
    a = np.loadtxt(path, delimiter=';', comments='#', dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 9:
        raise ValueError('%s: expected >=9 ";"-separated columns, got %s'
                         % (path, a.shape))
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 7], a[:, 8]


def sample_at(cl, s):
    """Interpolate the centerline at arclength s (wraps). -> (x, y, psi, wr, wl).

    psi is interpolated through its unwrapped form so the seam at ±pi doesn't
    spin an agent around when it crosses.
    """
    s_arr, x, y, psi, wr, wl = cl
    total = s_arr[-1]
    sm = math.fmod(s, total)
    if sm < 0.0:
        sm += total
    return (float(np.interp(sm, s_arr, x)),
            float(np.interp(sm, s_arr, y)),
            float(np.interp(sm, s_arr, np.unwrap(psi))),
            float(np.interp(sm, s_arr, wr)),
            float(np.interp(sm, s_arr, wl)))


def offset_point(x, y, psi, d):
    """Point d metres to the LEFT of (x, y) heading psi (d<0 = right)."""
    return x - d * math.sin(psi), y + d * math.cos(psi)


def triangle(t, period, lo, hi):
    """Triangle wave in [lo, hi] with the given period — a there-and-back sweep."""
    if period <= 0.0:
        return lo
    u = (t / period) % 1.0
    u = 2.0 * u if u < 0.5 else 2.0 * (1.0 - u)
    return lo + (hi - lo) * u


def agent_pose(agent, cl, t):
    """(x, y, yaw, v) for an agent at sim time t. Pure — this is the tested core."""
    if agent['kind'] == 'follow':
        s = agent['s0'] + agent['speed'] * t
        x, y, psi, wr, wl = sample_at(cl, s)
        d = max(-wr + agent['margin'], min(wl - agent['margin'], agent['offset']))
        px, py = offset_point(x, y, psi, d)
        # Driving backwards along s means facing the other way: that's oncoming.
        yaw = psi if agent['speed'] >= 0.0 else psi + math.pi
        return px, py, yaw, abs(agent['speed'])

    x, y, psi, wr, wl = sample_at(cl, agent['s0'])
    lo, hi = -wr + agent['margin'], wl - agent['margin']
    period = 2.0 * (hi - lo) / max(agent['speed'], 1e-3)
    d = triangle(t + agent['phase'] * period, period, lo, hi)
    px, py = offset_point(x, y, psi, d)
    # Crossing the corridor: face along the sweep, flipping at each turnaround.
    fwd = triangle(t + agent['phase'] * period + 1e-3, period, lo, hi) >= d
    yaw = psi + (math.pi / 2.0 if fwd else -math.pi / 2.0)
    return px, py, yaw, agent['speed']


def box_sdf(name, sx, sy, sz, colour):
    """Kinematic, gravity-off box — pose is prescribed, so physics must not fight it."""
    return """<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{n}'>
    <static>0</static>
    <link name='link'>
      <gravity>0</gravity>
      <kinematic>1</kinematic>
      <self_collide>0</self_collide>
      <inertial><mass>1.0</mass>
        <inertia><ixx>0.1</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>0.1</iyy><iyz>0</iyz><izz>0.1</izz></inertia>
      </inertial>
      <collision name='collision'>
        <pose>0 0 {hz} 0 0 0</pose>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
      <visual name='visual'>
        <pose>0 0 {hz} 0 0 0</pose>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material><script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/{c}</name>
        </script></material>
      </visual>
    </link>
  </model>
</sdf>""".format(n=name, sx=sx, sy=sy, sz=sz, hz=sz / 2.0, c=colour)


def build_agents(cl, n_follow, n_cross, seed, speed_range, cross_speed,
                 oncoming_frac, margin, start_gap):
    """Deterministic placement: evenly spread in arclength, jittered by `seed`.

    start_gap keeps agents clear of s=0, where the ego car spawns.
    """
    rng = random.Random(seed)
    total = float(cl[0][-1])
    usable = total - 2.0 * start_gap
    agents = []

    n = max(n_follow, 0) + max(n_cross, 0)
    if n == 0 or usable <= 0.0:
        return agents
    step = usable / n

    for i in range(n):
        s0 = start_gap + step * (i + 0.5) + rng.uniform(-0.25, 0.25) * step
        if i < n_follow:
            oncoming = rng.random() < oncoming_frac
            speed = rng.uniform(*speed_range) * (-1.0 if oncoming else 1.0)
            agents.append(dict(kind='follow', s0=s0, speed=speed, phase=0.0,
                               offset=rng.choice((-0.5, 0.5)) * rng.uniform(0.3, 0.8),
                               margin=margin,
                               name='dyn_follow_%d' % i, colour='Blue'))
        else:
            agents.append(dict(kind='cross', s0=s0, speed=cross_speed,
                               phase=rng.random(), offset=0.0, margin=margin,
                               name='dyn_cross_%d' % (i - n_follow), colour='Red'))
    return agents


# ------------------------------------------------------------------- ROS node

class DynamicObstacles(Node):

    def __init__(self):
        super().__init__('dynamic_obstacles')

        self.declare_parameter('centerline', 'obstacle_corridor')
        self.declare_parameter('n_follow', 3)
        self.declare_parameter('n_cross', 2)
        self.declare_parameter('seed', 0)
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('speed_min', 0.4)
        self.declare_parameter('speed_max', 1.2)
        self.declare_parameter('cross_speed', 0.6)
        self.declare_parameter('oncoming_frac', 0.34)
        self.declare_parameter('margin', 0.35)       # keep agents off the walls
        self.declare_parameter('start_gap', 3.0)     # clear of the ego spawn at s=0
        self.declare_parameter('follow_size', [0.50, 0.30, 0.40])   # another car
        self.declare_parameter('cross_size', [0.25, 0.25, 0.50])    # pedestrian

        g = lambda k: self.get_parameter(k).value

        path = resolve_centerline(g('centerline'))
        if not os.path.exists(path):
            raise RuntimeError('centerline not found: %s (run from the workspace '
                               'root, or pass an explicit path)' % path)
        self.cl = load_centerline(path)

        self.agents = build_agents(
            self.cl, g('n_follow'), g('n_cross'), g('seed'),
            (g('speed_min'), g('speed_max')), g('cross_speed'),
            g('oncoming_frac'), g('margin'), g('start_gap'))
        self.sizes = {'follow': list(g('follow_size')),
                      'cross': list(g('cross_size'))}

        self.get_logger().info(
            'centerline %s (%.1f m), %d agents' % (path, self.cl[0][-1], len(self.agents)))

        self._spawn = self.create_client(SpawnEntity, '/spawn_entity')
        self._state = self.create_client(SetEntityState, '/set_entity_state')
        for cli, name in ((self._spawn, '/spawn_entity'), (self._state, '/set_entity_state')):
            while not cli.wait_for_service(timeout_sec=2.0) and rclpy.ok():
                self.get_logger().info('waiting for %s ...' % name)

        self._spawn_all()
        self._t0 = self.get_clock().now()
        self.create_timer(1.0 / max(g('rate'), 1.0), self._tick)

    def _spawn_all(self):
        for a in self.agents:
            sx, sy, sz = self.sizes[a['kind']]
            x, y, yaw, _ = agent_pose(a, self.cl, 0.0)
            req = SpawnEntity.Request()
            req.name = a['name']
            req.xml = box_sdf(a['name'], sx, sy, sz, a['colour'])
            req.initial_pose = _pose(x, y, yaw, sz)
            self._spawn.call_async(req)
            self.get_logger().info(
                'spawn %s (%s) at s=%.1f speed=%.2f' % (a['name'], a['kind'],
                                                        a['s0'], a['speed']))

    def _tick(self):
        t = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        for a in self.agents:
            x, y, yaw, v = agent_pose(a, self.cl, t)
            sz = self.sizes[a['kind']][2]
            st = EntityState()
            st.name = a['name']
            st.pose = _pose(x, y, yaw, sz)
            st.twist.linear.x = v * math.cos(yaw)
            st.twist.linear.y = v * math.sin(yaw)
            st.reference_frame = 'world'
            req = SetEntityState.Request()
            req.state = st
            # ponytail: one service call per agent per tick (30 Hz x N). Fine to
            # ~10 agents; if it shows up in the RTF, batch via a custom plugin.
            self._state.call_async(req)


def _pose(x, y, yaw, sz):
    from geometry_msgs.msg import Pose
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), 0.0
    p.orientation.z = math.sin(yaw / 2.0)
    p.orientation.w = math.cos(yaw / 2.0)
    return p


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
