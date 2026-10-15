"""Double-track (4-wheel) planar vehicle model for the F1Tenth co-sim plant.

Clean-room, numpy/math only. Self-contained in this package.

Model summary
-------------
Planar body (x forward, y left) with 4 wheels. Per step:
  1. Steering servo: actual steer delta slews toward command (rate-limited).
  2. Ackermann geometry: front steer -> distinct left/right wheel angles.
  3. Powertrain: a P speed controller requests a drive force; the actual drive
     force Fx_drive follows it through a first-order motor/ESC lag (tau_motor).
  4. Resistive forces: rolling resistance (c_rr) + aerodynamic drag (c_drag).
  5. Vertical loads Fz per wheel with longitudinal + lateral load transfer.
  6. Per-wheel slip angle from body velocity + yaw rate + wheel position.
  7. Combined-slip Magic Formula -> per-wheel (Fx, Fy) (see pacejka.py).
  8. Sum forces/moments -> body accelerations. Integrate with RK4.

Below `v_blend` the tyre slip angle is ill-defined (division by ~0 speed), so
the model blends smoothly to a kinematic bicycle. This keeps standstill / launch
numerically stable.

State vector (length 8):
    [X, Y, psi, vx, vy, r, delta, Fx_drive]
        X, Y     : world position [m]
        psi      : yaw [rad]
        vx, vy   : body-frame velocity [m/s]
        r        : yaw rate [rad/s]
        delta    : actual front steer angle [rad]
        Fx_drive : actual powertrain longitudinal force [N] (post motor lag)

Input:
    (v_desired [m/s], delta_desired [rad])   from the Ackermann drive command.
"""

import math
from dataclasses import dataclass, field

from .pacejka import AxleTireParams, combined

G_ACCEL = 9.81

# state indices
IX, IY, IPSI, IVX, IVY, IR, IDELTA, IFX = range(8)
STATE_DIM = 8


@dataclass
class VehicleParams:
    # --- geometry / mass (F1Tenth scale) ---
    m: float = 3.74            # mass [kg]
    Iz: float = 0.04712        # yaw inertia [kg m^2]
    lf: float = 0.15875        # CoG -> front axle [m]
    lr: float = 0.17145        # CoG -> rear axle [m]
    track: float = 0.2032      # track width (left<->right) [m]
    h_cg: float = 0.074        # CoG height [m]

    # --- actuation limits ---
    delta_max: float = 0.4189      # |steer| limit [rad]
    delta_rate: float = 3.2        # steer slew rate [rad/s]
    a_max: float = 9.51            # |long accel| limit [m/s^2]
    v_max: float = 20.0
    v_min: float = -5.0

    # --- powertrain ---
    k_speed: float = 8.0           # P gain: Fx_demand = m*k_speed*(v_des - vx)
    tau_motor: float = 0.05        # ESC/motor first-order lag [s]
    rear_wheel_drive: bool = True  # RWD: drive force on rear axle only

    # --- resistive forces ---
    c_rr: float = 0.015            # rolling resistance coeff [-] (F = c_rr*m*g)
    c_drag: float = 0.05           # aero drag coeff [N/(m/s)^2] (F = c_drag*v^2)
    v_roll_eps: float = 0.05       # speed below which rolling resistance is off

    # --- low-speed kinematic blend ---
    v_blend_min: float = 1.0       # below this: pure kinematic
    v_blend_max: float = 2.0       # above this: pure dynamic

    # --- tyres (SEED VALUES — require system-ID on the real car) ---
    front_tire: AxleTireParams = field(
        default_factory=lambda: AxleTireParams(B=11.0, C=1.4, E=0.97, mu=1.05))
    rear_tire: AxleTireParams = field(
        default_factory=lambda: AxleTireParams(B=11.0, C=1.4, E=0.97, mu=1.05))

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


def ackermann_angles(delta: float, p: VehicleParams) -> tuple:
    """Left/right front wheel angles from a single bicycle steer `delta`.

    Ackermann geometry: inner wheel turns sharper than outer.
        cot(delta_o) - cot(delta_i) = track / wheelbase
    Returns (delta_left, delta_right) [rad].
    """
    if abs(delta) < 1e-6:
        return 0.0, 0.0
    L = p.wheelbase
    # Turn radius of the bicycle (rear-axle) reference point.
    R = L / math.tan(abs(delta))
    d_inner = math.atan(L / (R - p.track / 2.0))
    d_outer = math.atan(L / (R + p.track / 2.0))
    if delta > 0.0:  # left turn: left wheel is inner
        return math.copysign(d_inner, delta), math.copysign(d_outer, delta)
    else:            # right turn: right wheel is inner
        return math.copysign(d_outer, delta), math.copysign(d_inner, delta)


class DoubleTrackModel:
    """Four-wheel planar dynamics with combined-slip Pacejka tyres."""

    def __init__(self, params: VehicleParams = None):
        self.p = params or VehicleParams()
        # Cache of last-computed diagnostics (per full step), for logging.
        self.last_saturated = False

    # -- input conditioning -------------------------------------------------
    def _steer_servo(self, delta: float, delta_cmd: float, dt: float) -> float:
        delta_cmd = max(-self.p.delta_max, min(self.p.delta_max, delta_cmd))
        max_step = self.p.delta_rate * dt
        d = max(-max_step, min(max_step, delta_cmd - delta))
        return delta + d

    def _resistive_force(self, vx: float) -> float:
        """Longitudinal resistance (rolling + aero), opposing motion [N]."""
        p = self.p
        f_aero = p.c_drag * vx * abs(vx)
        if abs(vx) < p.v_roll_eps:
            f_roll = 0.0
        else:
            f_roll = math.copysign(p.c_rr * p.m * G_ACCEL, vx)
        return -(f_roll + f_aero)

    def _drive_demand(self, vx: float, v_des: float) -> float:
        """Commanded drive force from the P speed controller [N]."""
        p = self.p
        a_cmd = p.k_speed * (v_des - vx)
        a_cmd = max(-p.a_max, min(p.a_max, a_cmd))
        return p.m * a_cmd

    # -- continuous dynamics ------------------------------------------------
    def _dynamic_deriv(self, s, v_des):
        """State derivative from the full double-track dynamic model."""
        p = self.p
        X, Y, psi, vx, vy, r, delta, Fx_drive = s

        # Powertrain: actual drive force follows demand through motor lag.
        Fx_cmd = self._drive_demand(vx, v_des)
        Fx_drive_dot = (Fx_cmd - Fx_drive) / p.tau_motor

        # Resistive force (rolling + aero) acts at the body.
        F_res = self._resistive_force(vx)

        # Longitudinal accel estimate for load transfer (drive + resistance).
        ax_est = (Fx_drive + F_res) / p.m

        # Per-axle static vertical load, then longitudinal load transfer.
        L = p.wheelbase
        Fz_f = p.m * G_ACCEL * p.lr / L
        Fz_r = p.m * G_ACCEL * p.lf / L
        dFz_long = p.m * ax_est * p.h_cg / L      # + to rear on accel
        Fz_f -= dFz_long
        Fz_r += dFz_long

        # Lateral load transfer (approx 50/50 roll distribution).
        ay_est = vx * r  # centripetal lateral accel estimate
        dFz_lat_f = p.m * ay_est * p.h_cg / p.track * (p.lr / L)
        dFz_lat_r = p.m * ay_est * p.h_cg / p.track * (p.lf / L)

        # Per-wheel Fz (left/right), clamped non-negative. +ay -> load right.
        Fz_fl = max(0.0, Fz_f / 2.0 - dFz_lat_f)
        Fz_fr = max(0.0, Fz_f / 2.0 + dFz_lat_f)
        Fz_rl = max(0.0, Fz_r / 2.0 - dFz_lat_r)
        Fz_rr = max(0.0, Fz_r / 2.0 + dFz_lat_r)

        # Ackermann front wheel angles.
        d_fl, d_fr = ackermann_angles(delta, p)

        # Longitudinal drive-force split across wheels.
        if p.rear_wheel_drive and Fx_drive >= 0.0:
            Fx_fl = Fx_fr = 0.0
            Fx_rl = Fx_rr = Fx_drive / 2.0
        else:  # braking (or AWD): split across all four
            Fx_fl = Fx_fr = Fx_drive / 4.0
            Fx_rl = Fx_rr = Fx_drive / 4.0

        # Wheel slip angles: alpha = steer - atan2(vy_wheel, vx_wheel).
        tw = p.track / 2.0

        def slip(lf_or_lr_signed, y_off, steer):
            vx_w = vx - r * y_off
            vy_w = vy + r * lf_or_lr_signed
            if abs(vx_w) < 1e-4:
                vx_w = math.copysign(1e-4, vx_w) if vx_w != 0 else 1e-4
            return steer - math.atan2(vy_w, vx_w)

        a_fl = slip(p.lf, +tw, d_fl)
        a_fr = slip(p.lf, -tw, d_fr)
        a_rl = slip(-p.lr, +tw, 0.0)
        a_rr = slip(-p.lr, -tw, 0.0)

        # Combined-slip tyre forces (wheel frame: Fx along wheel heading).
        fx_fl, fy_fl, s1 = combined(a_fl, Fx_fl, Fz_fl, p.front_tire)
        fx_fr, fy_fr, s2 = combined(a_fr, Fx_fr, Fz_fr, p.front_tire)
        fx_rl, fy_rl, s3 = combined(a_rl, Fx_rl, Fz_rl, p.rear_tire)
        fx_rr, fy_rr, s4 = combined(a_rr, Fx_rr, Fz_rr, p.rear_tire)
        self.last_saturated = s1 or s2 or s3 or s4

        # Rotate front wheel forces from wheel frame into body frame.
        def to_body(fx, fy, steer):
            c, sn = math.cos(steer), math.sin(steer)
            return fx * c - fy * sn, fx * sn + fy * c

        Fx_fl_b, Fy_fl_b = to_body(fx_fl, fy_fl, d_fl)
        Fx_fr_b, Fy_fr_b = to_body(fx_fr, fy_fr, d_fr)
        # Rear wheels: steer = 0.
        Fx_rl_b, Fy_rl_b = fx_rl, fy_rl
        Fx_rr_b, Fy_rr_b = fx_rr, fy_rr

        # Resistance added to body-x (drag + rolling act opposite motion).
        Fx_body = Fx_fl_b + Fx_fr_b + Fx_rl_b + Fx_rr_b + F_res
        Fy_body = Fy_fl_b + Fy_fr_b + Fy_rl_b + Fy_rr_b

        # Yaw moment about CoG. Front wheels at +lf, rear at -lr; y=+tw/-tw.
        Mz = (
            p.lf * (Fy_fl_b + Fy_fr_b) - p.lr * (Fy_rl_b + Fy_rr_b)
            + tw * (-Fx_fl_b + Fx_fr_b) + tw * (-Fx_rl_b + Fx_rr_b)
        )

        # Body accelerations (with transport terms).
        ax = Fx_body / p.m + vy * r
        ay = Fy_body / p.m - vx * r
        r_dot = Mz / p.Iz

        Xdot = vx * math.cos(psi) - vy * math.sin(psi)
        Ydot = vx * math.sin(psi) + vy * math.cos(psi)
        return [Xdot, Ydot, r, ax, ay, r_dot, 0.0, Fx_drive_dot]

    def _kinematic_deriv(self, s, v_des):
        """Kinematic bicycle derivative — used at low speed (no slip).

        vy and yaw-rate r are algebraic in the kinematic model; here they are
        relaxed toward their kinematic targets with a short time constant so the
        shared state vector stays consistent across the dynamic/kinematic blend
        without introducing a dt dependence.
        """
        p = self.p
        X, Y, psi, vx, vy, r, delta, Fx_drive = s

        # Longitudinal: same powertrain + resistance as the dynamic model.
        Fx_cmd = self._drive_demand(vx, v_des)
        Fx_drive_dot = (Fx_cmd - Fx_drive) / p.tau_motor
        F_res = self._resistive_force(vx)
        ax = (Fx_drive + F_res) / p.m

        L = p.wheelbase
        beta = math.atan2(p.lr * math.tan(delta), L)
        v = vx  # treat vx as speed at low velocity
        Xdot = v * math.cos(psi + beta)
        Ydot = v * math.sin(psi + beta)
        psidot = v * math.cos(beta) * math.tan(delta) / L

        tau = 0.05  # relaxation time constant [s]
        vy_target = v * math.sin(beta)
        vy_dot = (vy_target - vy) / tau
        r_dot = (psidot - r) / tau
        return [Xdot, Ydot, psidot, ax, vy_dot, r_dot, 0.0, Fx_drive_dot]

    def _deriv(self, s, v_des):
        """Blend dynamic + kinematic derivatives by forward speed."""
        p = self.p
        vx = s[IVX]
        speed = abs(vx)
        if speed <= p.v_blend_min:
            w = 0.0
        elif speed >= p.v_blend_max:
            w = 1.0
        else:
            w = (speed - p.v_blend_min) / (p.v_blend_max - p.v_blend_min)

        if w >= 1.0:
            return self._dynamic_deriv(s, v_des)
        if w <= 0.0:
            return self._kinematic_deriv(s, v_des)
        dd = self._dynamic_deriv(s, v_des)
        kd = self._kinematic_deriv(s, v_des)
        return [w * a + (1.0 - w) * b for a, b in zip(dd, kd)]

    # -- integration --------------------------------------------------------
    def step(self, s, v_des, delta_cmd, dt, substeps=4):
        """Advance state `s` by dt using RK4 with `substeps` internal steps.

        Steering servo is applied first (outside the force integration).
        Accepts a length-7 or length-8 state (Fx_drive defaults to 0).
        Returns the new length-8 state list.
        """
        p = self.p
        s = list(s)
        if len(s) == 7:
            s.append(0.0)
        v_des = max(p.v_min, min(p.v_max, v_des))
        h = dt / substeps
        for _ in range(substeps):
            # Slew the actual steer toward the command over this substep.
            s[IDELTA] = self._steer_servo(s[IDELTA], delta_cmd, h)

            k1 = self._deriv(s, v_des)
            s2 = [si + 0.5 * h * ki for si, ki in zip(s, k1)]
            k2 = self._deriv(s2, v_des)
            s3 = [si + 0.5 * h * ki for si, ki in zip(s, k2)]
            k3 = self._deriv(s3, v_des)
            s4 = [si + h * ki for si, ki in zip(s, k3)]
            k4 = self._deriv(s4, v_des)
            s = [si + (h / 6.0) * (a + 2 * b + 2 * c + d)
                 for si, a, b, c, d in zip(s, k1, k2, k3, k4)]
            # Re-clamp steer (integration leaves delta derivative at 0).
            s[IDELTA] = max(-p.delta_max, min(p.delta_max, s[IDELTA]))
        return s
