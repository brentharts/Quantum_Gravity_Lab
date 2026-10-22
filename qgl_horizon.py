#!/usr/bin/env python3
"""
qgl_horizon.py -- what a pancake singularity does to particle horizons.

The idea under test: "two directions stay open and extended at the singularity, so cosmological horizons are broken
along those axes, allowing communication across arbitrary cylindrical distances".  Checked here against the
metrics we already have.  Short version of what comes out (the tests pin each statement):

* In the pancake model the axis that collapses is R_1 ~ a0 t, the closed Hopf-fibre direction.  A radial null ray
  along it covers a comoving angle  h_1 = int 2 dt / R_1 ~ (2/a0) ln(t / t_in),  which diverges as t_in -> 0.  So
  the *collapsing* direction has no particle horizon.  But that direction is a circle (period 4 pi in the
  standard Euler-angle S^3; the identification is not checked against the paper) whose proper circumference
  2 pi R_1 -> 0, so "arbitrary distance" there means winding ever more times round a vanishing circle.
* The two *open* directions (R_2 = R_3 -> b0, finite) have a finite comoving horizon, h_2 ~ 2 t / b0: light covers
  a proper distance ~ t, exactly as in an ordinary big bang.  Their horizon is not broken.  Two antipodal points
  of the base S^2 first come into causal contact when  int 2 dt / R_2 = pi, which takes t ~ pi b0 / 2.
* Kasner exponents give the general rule: the comoving horizon along axis i is  t^(1-p_i)/(1-p_i)  for p_i < 1 and
  diverges for p_i >= 1.  Vacuum Kasner has p_i <= 1 with equality only for (1, 0, 0), so the pancake is the
  marginal case: the only vacuum exponent set with a divergent horizon, and then only along one axis.
* The model also continues through t = 0 to t < 0, so "the horizon at the singularity" is partly a statement about
  where one starts counting.  With the continuation the null ray's comoving range is finite on the open axes.

Everything here is geometry of our own model; the coupling to any sensor is not involved.  Not done: Mixmaster
horizons (Misner's claim that infinitely many epochs make them divergent in every direction is not tested here).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid

import qgl_kasner as kn
import qgl_pancake as qp

FIBRE_PERIOD = 4.0 * np.pi      # period of the Euler angle psi on the standard S^3 (identification not checked)


# --------------------------------------------------------------------------
#  Kasner: closed forms
# --------------------------------------------------------------------------

def horizon_type(p: float, tol: float = 1e-12) -> str:
    """'finite' for p < 1, 'logarithmic' for p = 1 (int dt/t), 'power' for p > 1 (not reachable in vacuum)."""
    if abs(p - 1.0) <= tol:
        return "logarithmic"
    return "finite" if p < 1.0 else "power"


def kasner_comoving_horizon(p: float, t: float) -> float:
    """int_0^t t'^(-p) dt' = t^(1-p)/(1-p) for p < 1; inf for p >= 1.  Metric a_i = t^p_i, null ray along axis i."""
    return float(t ** (1.0 - p) / (1.0 - p)) if p < 1.0 - 1e-12 else float("inf")


def kasner_proper_horizon(p: float, t: float) -> float:
    """a_i(t) * comoving horizon = t / (1 - p) for p < 1.  Finite and ~ t on every non-collapsing axis."""
    return float(t / (1.0 - p)) if p < 1.0 - 1e-12 else float("inf")


def kasner_horizon_table(us: Sequence[float] = (1.5, 2.0, 3.0, 5.0)) -> list:
    """Per u: exponents, and which axes have a divergent comoving horizon (none, for vacuum Kasner with u > 1)."""
    rows = []
    for u in us:
        p = kn.kasner_exponents(u)
        rows.append(dict(u=u, p=tuple(float(x) for x in p),
                         types=tuple(horizon_type(float(x)) for x in p)))
    rows.append(dict(u=None, p=(1.0, 0.0, 0.0), types=tuple(horizon_type(x) for x in (1.0, 0.0, 0.0))))
    return rows


# --------------------------------------------------------------------------
#  Pancake: numerical horizons on t > 0
# --------------------------------------------------------------------------

def pancake_horizons(sol: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Comoving null-ray angles for t > 0 along a `pancake_solution`, counted from t_in = the first grid time
    (the solution skips |t| < t_start).

        fibre : int_{t_in}^t 2 dt / R_1          (grows like (2/a0) ln(t/t_in); diverges as t_in -> 0)
        base  : int_0^t   2 dt / R_2             (finite; the piece on [0, t_in] is added as 2 t_in / b0)

    plus the proper size of the fibre circle, 2 pi R_1, and the proper distance light has covered along the
    base, int dt = t (a null ray has |dx_proper/dt| = 1 in a comoving frame, so this is just elapsed time).
    Keys: t, fibre, base, fibre_circumference, proper_reach, fibre_turns.
    """
    pos = sol["t"] > 0
    t = sol["t"][pos]
    R1, R2 = sol["R"][pos, 0], sol["R"][pos, 1]
    fibre = cumulative_trapezoid(2.0 / R1, t, initial=0.0)
    base = cumulative_trapezoid(2.0 / R2, t, initial=0.0) + 2.0 * t[0] / sol["params"]["b0"]
    return dict(t=t, fibre=fibre, base=base, fibre_circumference=2.0 * np.pi * R1,
                proper_reach=t.copy(), fibre_turns=fibre / FIBRE_PERIOD)


def fibre_log_coefficient(a0: float) -> float:
    """Leading growth of the fibre horizon: h_1 = (2/a0) ln(t/t_in).  This is the same 2/a0 as in x_winding."""
    return 2.0 / a0


def base_contact_time(hz: Dict[str, np.ndarray], angle: float = np.pi) -> float:
    """First t at which the base null angle reaches `angle` (pi: antipodal points of the base S^2); nan if not reached."""
    idx = np.flatnonzero(hz["base"] >= angle)
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(hz["t"][0])
    t0, t1, h0, h1 = hz["t"][i - 1], hz["t"][i], hz["base"][i - 1], hz["base"][i]
    return float(t0 + (angle - h0) * (t1 - t0) / (h1 - h0))


def fibre_time_to_wrap(a0: float, t: float, turns: float = 1.0) -> float:
    """Latest t_in such that a ray started at t_in has wound `turns` times by time t:  t_in = t exp(-turns 4 pi a0 / 2)."""
    return float(t * np.exp(-turns * FIBRE_PERIOD * a0 / 2.0))


def summary(a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5, t_max: float = 6.0) -> Dict[str, float]:
    sol = qp.pancake_solution(a0, b0, f0, m, t_max=t_max)
    hz = pancake_horizons(sol)
    i1 = int(np.searchsorted(hz["t"], 1.0))
    return dict(t_contact_base=base_contact_time(hz),
                pi_b0_over_2=float(np.pi * b0 / 2.0),
                fibre_turns_t1=float(hz["fibre_turns"][i1]),
                t_in=float(hz["t"][0]),
                fibre_circumference_t1=float(hz["fibre_circumference"][i1]),
                base_angle_t1=float(hz["base"][i1]))


def main() -> None:
    print("Kasner comoving horizon type per axis (t^(1-p)/(1-p) for p<1, divergent for p>=1)")
    for r in kasner_horizon_table():
        label = "pancake" if r["u"] is None else f"u={r['u']}"
        print(f"  {label:8s} p=({r['p'][0]:+.3f}, {r['p'][1]:+.3f}, {r['p'][2]:+.3f})  {r['types']}")
    s = summary()
    print("\nPancake (a0=1, b0=3, f0=2, m=0.5), light from the pancake outward:")
    print(f"  base S^2 antipodal contact at t = {s['t_contact_base']:.2f}   (pi b0/2 = {s['pi_b0_over_2']:.2f} if R_2 stayed b0)")
    print(f"  fibre turns by t=1, counted from t_in={s['t_in']:.0e}: {s['fibre_turns_t1']:.2f}  (grows like ln(1/t_in))")
    print(f"  fibre circumference at t=1: {s['fibre_circumference_t1']:.2f};  base angle covered by t=1: {s['base_angle_t1']:.2f} rad of pi")


if __name__ == "__main__":
    main()
