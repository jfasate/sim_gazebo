"""Shared, dependency-light helpers for system-ID (no rclpy / no scipy).

Kept in the `dynamics` subpackage so both the ROS excitation node and the
offline fitter can import it without pulling ROS or SciPy.
"""

import math


def build_timeline(rate, cruise=3.0):
    """Return a list of (v_des, delta) setpoints, one per 1/rate tick.

    Deliberately *gentle*: steering stays in the linear-to-mild-nonlinear
    regime and a slow ramp probes toward the grip limit without forcing a full
    spin, so the identification residual surface stays well behaved.
    """
    dt = 1.0 / rate
    tl = []

    def hold(duration, v, d):
        for _ in range(int(round(duration / dt))):
            tl.append((v, d))

    # 1. Longitudinal steps (straight): observability for k_speed, tau_motor,
    #    c_rr, c_drag.
    hold(1.0, 0.0, 0.0)
    hold(3.0, 2.0, 0.0)
    hold(3.0, 4.0, 0.0)
    hold(3.0, 6.0, 0.0)
    hold(3.0, 2.0, 0.0)     # braking transient
    hold(2.0, 0.0, 0.0)

    # 2. Constant-speed steering steps (moderate) at cruise.
    hold(2.5, cruise, 0.0)
    for d in (0.05, 0.10, 0.15, 0.20):
        hold(1.5, cruise, d)
        hold(1.0, cruise, 0.0)
    for d in (-0.10, -0.20):
        hold(1.5, cruise, d)
        hold(1.0, cruise, 0.0)

    # 3. Slow steering ramp toward the grip limit (probes mu without a spin).
    n_ramp = int(round(4.0 / dt))
    for i in range(n_ramp):
        tl.append((cruise, 0.30 * (i / n_ramp)))
    hold(1.0, cruise, 0.0)

    # 4. Small steering sine sweep (chirp 0.2 -> 1.5 Hz) for transient / Iz.
    hold(1.0, cruise, 0.0)
    n = int(round(5.0 / dt))
    f0, f1, amp = 0.2, 1.5, 0.08
    for i in range(n):
        t = i * dt
        f = f0 + (f1 - f0) * (t / (n * dt))
        tl.append((cruise, amp * math.sin(2.0 * math.pi * f * t)))

    hold(1.0, 0.0, 0.0)
    return tl


def simulate(model, t, cmd_v, cmd_delta, vx0, substeps=2):
    """Replay commands through a DoubleTrackModel.

    Returns three lists (vx, vy, r) sampled at each command tick.
    `model` is a DoubleTrackModel already carrying the candidate params.
    `t`, `cmd_v`, `cmd_delta` are sequences of equal length.
    """
    s = [0.0, 0.0, 0.0, vx0, 0.0, 0.0, 0.0, 0.0]
    n = len(t)
    vx = [0.0] * n
    vy = [0.0] * n
    r = [0.0] * n
    for i in range(n):
        dt = (t[i] - t[i - 1]) if i > 0 else (t[1] - t[0])
        dt = max(1e-3, min(dt, 0.1))
        s = model.step(s, cmd_v[i], cmd_delta[i], dt, substeps=substeps)
        vx[i], vy[i], r[i] = s[3], s[4], s[5]
    return vx, vy, r
