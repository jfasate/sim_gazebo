"""Combined-slip Pacejka Magic Formula tire model (clean-room, numpy only).

Pure-slip Magic Formula for longitudinal (Fx0) and lateral (Fy0) force, then
combined-slip coupling by the friction-ellipse (similarity) method: a tyre has a
finite grip budget mu*Fz; longitudinal force spends part of it, leaving less for
lateral. This reproduces the dominant combined-slip effects for an F1Tenth car:

  - cornering force saturates at the grip limit (nonlinear vs slip angle),
  - accelerating/braking hard in a corner steals lateral grip (understeer/spin),
  - force demand above mu*Fz -> saturation (wheelspin / lockup regime).

Only a handful of physically meaningful, ID-able parameters per axle
(B, C, D=mu*Fz, E). This deliberately avoids the ~30-coefficient ADAMS parameter
set, for which no F1Tenth-scale values exist.

Magic Formula (single output):
    F = D * sin( C * atan( B*s - E*(B*s - atan(B*s)) ) )
with slip s = kappa (longitudinal) or alpha (lateral, rad).
"""

import math
from dataclasses import dataclass


@dataclass
class AxleTireParams:
    """Magic Formula coefficients for one axle (shared by its two wheels).

    B : stiffness factor  (initial slope ~ B*C*D)
    C : shape factor      (~1.3 lateral, ~1.6 longitudinal typical)
    E : curvature factor  (peak/tail shaping, |E| <= 1)
    mu: peak friction coefficient (peak force = mu * Fz)
    """
    B: float
    C: float
    E: float
    mu: float


def _magic(slip: float, B: float, C: float, E: float, D: float) -> float:
    """Scalar Magic Formula. Returns force for a given slip and peak D."""
    Bs = B * slip
    return D * math.sin(C * math.atan(Bs - E * (Bs - math.atan(Bs))))


def pure_lateral(alpha: float, Fz: float, p: AxleTireParams) -> float:
    """Pure-slip lateral force Fy0 [N] from slip angle alpha [rad]."""
    if Fz <= 0.0:
        return 0.0
    D = p.mu * Fz
    return _magic(alpha, p.B, p.C, p.E, D)


def pure_longitudinal(kappa: float, Fz: float, p: AxleTireParams) -> float:
    """Pure-slip longitudinal force Fx0 [N] from slip ratio kappa [-]."""
    if Fz <= 0.0:
        return 0.0
    D = p.mu * Fz
    return _magic(kappa, p.B, p.C, p.E, D)


def combined(alpha: float, Fx_demand: float, Fz: float,
             p: AxleTireParams) -> tuple:
    """Combined-slip wheel forces by the friction-ellipse method.

    Inputs:
        alpha      : slip angle [rad]
        Fx_demand  : commanded longitudinal force at this wheel [N]
                     (drive/brake force requested by the powertrain model)
        Fz         : vertical load on this wheel [N]
        p          : axle tire parameters

    Returns (Fx, Fy, saturated):
        Fx : realised longitudinal force [N] (clamped to grip budget)
        Fy : realised lateral force [N] (reduced by longitudinal usage)
        saturated : True if the grip budget was exceeded (wheelspin/lockup)

    Method:
        grip budget  G = mu * Fz
        Fx = clamp(Fx_demand, -G, G)
        remaining lateral capacity  Fy_max = sqrt(G^2 - Fx^2)
        Fy = clamp( Fy0(alpha),  -Fy_max, +Fy_max )
    """
    if Fz <= 0.0:
        return 0.0, 0.0, True

    G = p.mu * Fz

    saturated = abs(Fx_demand) >= G
    Fx = max(-G, min(G, Fx_demand))

    # Lateral capacity left after longitudinal force is spent (friction ellipse).
    Fy_cap = math.sqrt(max(G * G - Fx * Fx, 0.0))

    Fy0 = pure_lateral(alpha, Fz, p)
    if abs(Fy0) > Fy_cap:
        Fy0 = math.copysign(Fy_cap, Fy0)
        saturated = True

    return Fx, Fy0, saturated
