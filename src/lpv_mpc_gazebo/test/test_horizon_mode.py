"""Phase-1 horizon-mode logic checks (pure numpy, no ROS).

Mirrors the two non-trivial bits of lpv_mpc_node so a regression is caught
cheaply:
  - _resample_open: arc-length resample of an open path to uniform spacing
  - open-vs-closed index handling in _build_reference / _lookahead_ref_speed
Run: python3 test/test_horizon_mode.py   (or via pytest)
"""
import numpy as np


def resample_open(raw, ds_target=0.10):        # mirrors _resample_open
    xy = raw[:, 1:3]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < ds_target:
        return raw
    m = max(2, int(round(total / ds_target)) + 1)
    s_new = np.linspace(0.0, total, m)
    return np.column_stack([
        s_new,
        np.interp(s_new, s, xy[:, 0]),
        np.interp(s_new, s, xy[:, 1]),
        np.zeros(m), np.zeros(m),
        np.interp(s_new, s, raw[:, 5]),
    ])


def idx_advance(wp_idx, adv, n, closed):       # mirrors the wrap/clamp
    return (wp_idx + adv) % n if closed else min(wp_idx + adv, n - 1)


def test_resample_uniform_and_preserves_ends():
    raw = np.zeros((5, 6))
    raw[:, 1] = [0.0, 1.0, 3.0, 3.5, 6.0]      # x, uneven spacing
    raw[:, 2] = 0.0                            # y (straight line)
    raw[:, 5] = [3, 3.5, 4, 4.5, 5]            # vx ramps up
    out = resample_open(raw, ds_target=0.1)
    d = np.linalg.norm(np.diff(out[:, 1:3], axis=0), axis=1)
    assert np.allclose(d, d[0], atol=1e-9)                 # uniform spacing
    assert abs(d[0] - 0.1) < 0.02                          # ~= ds_target
    assert np.isclose(out[0, 1], 0.0) and np.isclose(out[-1, 1], 6.0)   # x ends
    assert np.isclose(out[0, 5], 3.0) and np.isclose(out[-1, 5], 5.0)   # vx ends
    assert np.all(np.diff(out[:, 5]) >= -1e-9)             # vx monotonic up


def test_open_clamps_closed_wraps():
    n = 100
    assert idx_advance(95, 20, n, closed=True) == 15       # closed wraps
    assert idx_advance(95, 20, n, closed=False) == 99      # open clamps to last
    assert idx_advance(10, 5, n, True) == 15               # interior: identical
    assert idx_advance(10, 5, n, False) == 15


def horizon_ok(raw, min_len=1.0, max_len=60.0):   # mirrors _ref_path_callback guard
    seg = np.linalg.norm(np.diff(raw[:, 1:3], axis=0), axis=1)
    total = float(seg.sum())
    n_distinct = 1 + int((seg > 1e-6).sum())
    return not (n_distinct < 2 or total < min_len or total > max_len)


def min_ref_speed(raw_v, speed_scale, closed, topic_min_ref_speed=0.0):
    # mirrors lpv_mpc_node._ingest_waypoints
    if closed:
        return max(float(np.min(raw_v)) * speed_scale, 2.0)
    return max(topic_min_ref_speed, 0.0)


def startup_speed(wp_v_target, closed, topic_startup_speed=0.4):
    # mirrors the low-speed branch in lpv_mpc_node.control_loop
    if closed:
        return 2.0
    ref_cap = float(np.nanmax(wp_v_target)) if len(wp_v_target) else 0.0
    return min(max(topic_startup_speed, 0.0), ref_cap)


def reference_advance_speed(state_speed, closed, topic_reference_speed_floor=0.10):
    # mirrors lpv_mpc_node._build_reference
    if closed:
        return max(state_speed, 1.5)
    return max(state_speed, topic_reference_speed_floor)


def test_degenerate_horizon_rejected():
    """A bad generator (collapsed/tiny/scattered points) must be rejected so the
    controller holds its last reference instead of crashing on an empty array."""
    collapsed = np.zeros((20, 6)); collapsed[:, 1] = 5.0; collapsed[:, 2] = 3.0
    assert not horizon_ok(collapsed)                       # all identical
    tiny = np.zeros((20, 6)); tiny[:, 1] = np.linspace(0, 0.3, 20)
    assert not horizon_ok(tiny)                            # span < 1 m
    huge = np.zeros((20, 6)); huge[:, 1] = np.linspace(0, 120, 20)
    assert not horizon_ok(huge)                            # span > 60 m (scattered)
    good = np.zeros((30, 6)); good[:, 1] = np.linspace(0, 10, 30)
    assert horizon_ok(good)                                # ~10 m forward line


def test_topic_reference_keeps_low_safe_speed():
    """Topic/neural mode must not inherit the CSV racing speed floor."""
    raw_v = np.full(20, 2.0)
    assert min_ref_speed(raw_v, speed_scale=0.1, closed=True) == 2.0
    assert min_ref_speed(raw_v, speed_scale=0.1, closed=False) == 0.0
    assert startup_speed(np.full(20, 0.2), closed=False) == 0.2
    assert startup_speed(np.full(20, 0.8), closed=False) == 0.4
    assert reference_advance_speed(0.04, closed=True) == 1.5
    assert reference_advance_speed(0.04, closed=False) == 0.10


if __name__ == '__main__':
    test_resample_uniform_and_preserves_ends()
    test_open_clamps_closed_wraps()
    test_degenerate_horizon_rejected()
    test_topic_reference_keeps_low_safe_speed()
    print('ok')
