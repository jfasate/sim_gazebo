#!/usr/bin/env python3
"""Self-check for the dynamic-obstacle path math (the only non-trivial part).

Fails if an agent ever leaves the corridor, if arclength stops wrapping, or if
placement stops being reproducible from the seed.

  python3 src/f1tenth_gym_gazebo/test/test_dynamic_obstacles.py
"""

import math

import numpy as np

from f1tenth_gym_gazebo.dynamic_obstacles import (
    agent_pose, build_agents, offset_point, sample_at, triangle,
)

R, HALF_W = 5.0, 1.5


def circle_centerline(n=400):
    """A closed circle of radius R with a constant HALF_W corridor."""
    th = np.linspace(0.0, 2.0 * math.pi, n)
    return (R * th,                      # s
            R * np.cos(th), R * np.sin(th),
            th + math.pi / 2.0,          # psi: tangent, CCW
            np.full(n, HALF_W), np.full(n, HALF_W))


def radius_of(x, y):
    return math.hypot(x, y)


def test_sample_wraps():
    cl = circle_centerline()
    total = cl[0][-1]
    a = sample_at(cl, 1.0)
    b = sample_at(cl, total + 1.0)
    c = sample_at(cl, -total + 1.0)
    for i in (0, 1):                     # x, y identical across wraps
        assert abs(a[i] - b[i]) < 1e-6, (a, b)
        assert abs(a[i] - c[i]) < 1e-6, (a, c)


def test_offset_point_is_left_handed():
    # heading +x, offset +1 must move +y (left); -1 must move -y (right).
    x, y = offset_point(0.0, 0.0, 0.0, 1.0)
    assert abs(x) < 1e-9 and abs(y - 1.0) < 1e-9, (x, y)
    x, y = offset_point(0.0, 0.0, 0.0, -1.0)
    assert abs(y + 1.0) < 1e-9, (x, y)


def test_triangle_stays_in_bounds_and_turns_around():
    vals = [triangle(t, 4.0, -1.0, 2.0) for t in np.linspace(0.0, 12.0, 200)]
    assert min(vals) >= -1.0 - 1e-9 and max(vals) <= 2.0 + 1e-9
    assert abs(triangle(0.0, 4.0, -1.0, 2.0) + 1.0) < 1e-9      # starts at lo
    assert abs(triangle(2.0, 4.0, -1.0, 2.0) - 2.0) < 1e-9      # hi at half period
    assert abs(triangle(4.0, 4.0, -1.0, 2.0) + 1.0) < 1e-9      # back to lo


def test_agents_never_leave_the_corridor():
    cl = circle_centerline()
    margin = 0.35
    agents = build_agents(cl, n_follow=3, n_cross=2, seed=7,
                          speed_range=(0.4, 1.2), cross_speed=0.6,
                          oncoming_frac=0.34, margin=margin, start_gap=3.0)
    assert len(agents) == 5
    limit = HALF_W - margin + 1e-6
    for a in agents:
        for t in np.linspace(0.0, 120.0, 600):
            x, y, yaw, v = agent_pose(a, cl, t)
            # On a circle, lateral offset from the centerline is |radius - R|.
            assert abs(radius_of(x, y) - R) <= limit, (a['name'], t, radius_of(x, y))
            assert v >= 0.0 and math.isfinite(yaw)


def test_cross_agents_actually_sweep():
    cl = circle_centerline()
    agents = build_agents(cl, n_follow=0, n_cross=1, seed=3,
                          speed_range=(0.4, 1.2), cross_speed=0.6,
                          oncoming_frac=0.0, margin=0.35, start_gap=3.0)
    offs = [radius_of(*agent_pose(agents[0], cl, t)[:2]) - R
            for t in np.linspace(0.0, 20.0, 300)]
    assert max(offs) - min(offs) > 1.0, 'cross agent barely moved: %r' % (offs[:5],)


def test_follow_agents_advance_along_s():
    cl = circle_centerline()
    agents = build_agents(cl, n_follow=1, n_cross=0, seed=1,
                          speed_range=(0.8, 0.8), cross_speed=0.6,
                          oncoming_frac=0.0, margin=0.35, start_gap=3.0)
    p0 = agent_pose(agents[0], cl, 0.0)[:2]
    p1 = agent_pose(agents[0], cl, 1.0)[:2]
    assert math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 0.5, (p0, p1)


def test_placement_is_reproducible_and_clear_of_spawn():
    cl = circle_centerline()
    kw = dict(n_follow=3, n_cross=2, speed_range=(0.4, 1.2), cross_speed=0.6,
              oncoming_frac=0.34, margin=0.35, start_gap=3.0)
    a = build_agents(cl, seed=42, **kw)
    b = build_agents(cl, seed=42, **kw)
    c = build_agents(cl, seed=43, **kw)
    assert [x['s0'] for x in a] == [x['s0'] for x in b], 'seed is not deterministic'
    assert [x['s0'] for x in a] != [x['s0'] for x in c], 'seed has no effect'
    total = cl[0][-1]
    for x in a:                          # never on top of the ego spawn at s=0
        assert 3.0 <= x['s0'] <= total - 3.0, x


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
    print('all dynamic-obstacle checks passed')
