#!/usr/bin/env python3
"""System-ID fitter: recover VehicleParams from a logged excitation run.

Simulation-based least-squares: replay the logged command sequence through the
DoubleTrackModel with candidate parameters and minimise the error between the
predicted and measured response.

Staged identification (decoupled -> robust):
  Stage 1 (longitudinal): fit [k_speed, tau_motor, c_rr, c_drag] to forward
      speed vx on the straight-line segments (steering = 0).
  Stage 2 (lateral): hold Stage-1 params fixed, fit [Iz, Bf, muf, Br, mur] to
      yaw rate r over the steering segments.

A robust `soft_l1` loss down-weights the near-limit / transient outliers so a
few chaotic samples cannot wreck the fit.

Usage:
    ros2 run f1tenth_gym_gazebo sysid_fit /tmp/sysid_log.csv [--out params.yaml]
    /usr/bin/python3 sysid_fit.py --selfcheck      # no hardware needed
"""

import argparse
import sys

import numpy as np
from scipy.optimize import least_squares

from f1tenth_gym_gazebo.dynamics.vehicle import DoubleTrackModel, VehicleParams
from f1tenth_gym_gazebo.dynamics.pacejka import AxleTireParams
from f1tenth_gym_gazebo.dynamics.sysid_common import build_timeline, simulate

# ---- fit-vector layouts ----
LONG_NAMES = ['k_speed', 'tau_motor', 'c_rr', 'c_drag']
LONG_LOWER = [1.0, 0.005, 0.0, 0.0]
LONG_UPPER = [30.0, 0.50, 0.10, 0.50]

LAT_NAMES = ['Iz', 'Bf', 'muf', 'Br', 'mur']
LAT_LOWER = [0.01, 3.0, 0.4, 3.0, 0.4]
LAT_UPPER = [0.30, 25.0, 2.0, 25.0, 2.0]

ALL_NAMES = LONG_NAMES + LAT_NAMES


def make_params(long_vec, lat_vec):
    vp = VehicleParams()
    vp.k_speed, vp.tau_motor, vp.c_rr, vp.c_drag = long_vec
    Iz, Bf, muf, Br, mur = lat_vec
    vp.Iz = Iz
    vp.front_tire = AxleTireParams(B=Bf, C=1.4, E=0.97, mu=muf)
    vp.rear_tire = AxleTireParams(B=Br, C=1.4, E=0.97, mu=mur)
    return vp


def long_default():
    d = VehicleParams()
    return np.array([d.k_speed, d.tau_motor, d.c_rr, d.c_drag])


def lat_default():
    d = VehicleParams()
    return np.array([d.Iz, d.front_tire.B, d.front_tire.mu,
                     d.rear_tire.B, d.rear_tire.mu])


# ---- stage residuals ----
def _res_long(pl, t, cv, cd, meas_vx, mask):
    vp = make_params(pl, lat_default())
    vx, _, _ = simulate(DoubleTrackModel(vp), t, cv, cd, meas_vx[0])
    return (np.asarray(vx) - meas_vx)[mask]


def _res_lat(plat, long_fixed, t, cv, cd, meas_r):
    vp = make_params(long_fixed, plat)
    _, _, r = simulate(DoubleTrackModel(vp), t, cv, cd, 0.0)
    return np.asarray(r) - meas_r


def fit(t, cv, cd, meas_vx, meas_r):
    straight = np.abs(cd) < 1e-3

    # Stage 1: longitudinal on straight segments.
    r1 = least_squares(
        _res_long, long_default(), bounds=(LONG_LOWER, LONG_UPPER),
        args=(t, cv, cd, meas_vx, straight),
        loss='soft_l1', f_scale=0.3, max_nfev=300, verbose=0)

    # Stage 2: lateral, longitudinal held fixed.
    r2 = least_squares(
        _res_lat, lat_default(), bounds=(LAT_LOWER, LAT_UPPER),
        args=(r1.x, t, cv, cd, meas_r),
        loss='soft_l1', f_scale=0.3, max_nfev=400, verbose=0)

    x = np.concatenate([r1.x, r2.x])
    # Report final trajectory RMS on the full model.
    vp = make_params(r1.x, r2.x)
    vx, _, rr = simulate(DoubleTrackModel(vp), t, cv, cd, meas_vx[0])
    rms_vx = float(np.sqrt(np.mean((np.asarray(vx) - meas_vx) ** 2)))
    rms_r = float(np.sqrt(np.mean((np.asarray(rr) - meas_r) ** 2)))
    return x, rms_vx, rms_r


def load_csv(path):
    d = np.genfromtxt(path, delimiter=',', names=True)
    return d['t'], d['cmd_v'], d['cmd_delta'], d['vx'], d['r']


def print_report(x, rms_vx, rms_r):
    print('\n=== fitted parameters ===')
    for name, val in zip(ALL_NAMES, x):
        print(f'  {name:10s} = {val:.5f}')
    print(f'  (traj RMS: vx={rms_vx:.4f} m/s, r={rms_r:.4f} rad/s)')


def write_yaml(x, path):
    with open(path, 'w') as f:
        f.write('# Fitted VehicleParams (from sysid_fit)\n')
        f.write('vehicle_params:\n')
        for name, val in zip(ALL_NAMES, x):
            f.write(f'  {name}: {val:.6f}\n')
        f.write('  # tyre C=1.4, E=0.97 held fixed during the fit\n')
    print(f'  wrote -> {path}')


def selfcheck():
    """Recover known (well-behaved, understeer) params from synthetic data."""
    print('Self-check: synthesising data from known ground-truth params...')
    gt_long = np.array([10.0, 0.08, 0.020, 0.08])
    gt_lat = np.array([0.050, 10.0, 1.05, 12.0, 1.25])  # rear-grippy understeer
    gt = np.concatenate([gt_long, gt_lat])

    rate = 50.0
    tl = build_timeline(rate, cruise=3.0)
    dt = 1.0 / rate
    t = np.arange(len(tl)) * dt
    cv = np.array([v for v, _ in tl])
    cd = np.array([d for _, d in tl])

    vp = make_params(gt_long, gt_lat)
    vx, _, r = simulate(DoubleTrackModel(vp), t, cv, cd, 0.0)
    vx = np.asarray(vx); r = np.asarray(r)
    rng = np.random.default_rng(0)
    vx_m = vx + rng.normal(0, 0.02, vx.shape)
    r_m = r + rng.normal(0, 0.01, r.shape)
    print(f'  (max |r| in maneuver = {np.max(np.abs(r)):.2f} rad/s)')

    x, rms_vx, rms_r = fit(t, cv, cd, vx_m, r_m)
    print_report(x, rms_vx, rms_r)
    print('\n  ground truth vs fit:')
    for name, g, fv in zip(ALL_NAMES, gt, x):
        err = 100.0 * abs(fv - g) / (abs(g) + 1e-9)
        note = ''
        if name in ('c_rr', 'c_drag'):
            note = '  (correlated pair; see total resistance below)'
        print(f'    {name:10s} gt={g:8.4f}  fit={fv:8.4f}  err={err:5.1f}%{note}')

    # The identifiable handling params must be tight; rolling/aero split is
    # only weakly observable, so validate their SUM at a reference speed.
    def resist(c_rr, c_drag, v):
        return c_rr * VehicleParams().m * 9.81 + c_drag * v * v
    v_ref = 5.0
    f_gt = resist(gt[2], gt[3], v_ref)
    f_fit = resist(x[2], x[3], v_ref)
    res_err = 100.0 * abs(f_fit - f_gt) / f_gt
    handling_ok = all(
        100.0 * abs(x[i] - gt[i]) / abs(gt[i]) < 15
        for i, n in enumerate(ALL_NAMES) if n in ('Iz', 'Bf', 'muf', 'Br', 'mur'))
    traj_ok = rms_vx < 0.1 and rms_r < 0.05
    print(f'\n  total resistance @ {v_ref:.0f} m/s: '
          f'gt={f_gt:.3f} N  fit={f_fit:.3f} N  err={res_err:.1f}%')
    print('\n  RESULT:',
          'PASS' if (handling_ok and traj_ok and res_err < 15)
          else 'CHECK (see above)',
          f'| trajectory {"ok" if traj_ok else "BAD"}, '
          f'handling {"ok" if handling_ok else "BAD"}, '
          f'resistance {"ok" if res_err < 15 else "BAD"}')


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?', help='excitation log CSV')
    ap.add_argument('--out', default=None, help='write fitted params YAML')
    ap.add_argument('--selfcheck', action='store_true',
                    help='recover known params from synthetic data')
    args = ap.parse_args(argv)

    if args.selfcheck:
        selfcheck()
        return
    if not args.csv:
        ap.error('provide a CSV path or --selfcheck')

    t, cv, cd, vx, r = load_csv(args.csv)
    print(f'Loaded {len(t)} samples from {args.csv}')
    x, rms_vx, rms_r = fit(t, cv, cd, vx, r)
    print_report(x, rms_vx, rms_r)
    if args.out:
        write_yaml(x, args.out)


if __name__ == '__main__':
    main()
