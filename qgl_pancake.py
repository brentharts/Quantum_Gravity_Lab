#!/usr/bin/env python3
"""
qgl_pancake.py -- the non-singular "pancake" Big Bang of Dechant, Lasenby & Hobson
(arXiv:0809.4335): a massive scalar field in a biaxial Bianchi IX universe.

Metric (their normalisation, kappa = 8 pi G):

    ds^2 = -dt^2 + (1/4) sum_i R_i^2 (omega^i)^2 ,     R_2 = R_3

so R_i = 2 a_i in the convention of qgl_bkl (a_i multiplies sigma^i).  The matter is a scalar field phi with
V = m^2 phi^2 / 2 and a cosmological constant Lambda.  With H_i = R_i'/R_i, S = sum H_i and the slice
Ricci components R3_ii = 2 (R_i^4 - (R_j^2 - R_k^2)^2) / (R_1 R_2 R_3)^2 (the same as qgl_bkl.ricci3 with a = R/2):

    R_i'' = R_i ( -H_i (H_j + H_k) - R3_ii + kappa V + Lambda )
    phi'' = -S phi' - m^2 phi
    constraint:  sum_{i<j} H_i H_j + (1/2) sum_i R3_ii - kappa (phi'^2/2 + V) - Lambda = 0

These were derived independently of the paper (diagonal Bianchi IX, Einstein + scalar) and are checked against
the paper's equations (25)-(28) and its curvature at t = 0 (eq. 31) in tests/test_pancake.py.

The pancake solution is the odd-parity series
    R_1 = t (a0 + a2 t^2 + ...),  R_2 = R_3 = b0 + b2 t^2 + ...,  phi = f0 + f2 t^2 + ...
with a0, b0, f0 free.  R_1 -> 0 linearly while R_2 stays finite, so the volume vanishes at t = 0, but the
curvature invariants stay finite (K(0) is given in closed form by `kretschmann_at_pancake`); the shear diverges
as 1/t^2.  The solution continues through t = 0 with R_1 odd and R_2, phi even.

`pancake_sensor_run` drives the QGL complexity scalar and Jaynes-Cummings sensor with K(t) along the comoving
worldline through the pancake.  That coupling is a modelling choice, as in qgl_curvature.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from unified_sandbox_core import Config
import qgl_bkl as bk
import qgl_curvature as qc


# --------------------------------------------------------------------------
#  Equations
# --------------------------------------------------------------------------

def slice_ricci(R) -> np.ndarray:
    """
    R3_ii for scale factors R_i (shape (..., 3)) in the paper's normalisation; equals bk.ricci3(R / 2).

    The numerator q_i^2 - (q_j - q_k)^2 (q = R^2) is evaluated in factored form (q_i - q_j + q_k)(q_i - q_k + q_j), differencing the
    large equal terms first.
    Near the pancake R_1 -> 0 and the expanded form cancels catastrophically: at |t| = 1e-6 it loses about
    eight digits, which showed up as a 1e-4 violation of the constraint.
    """
    q = np.asarray(R, dtype=float) ** 2
    P = q[..., 0] * q[..., 1] * q[..., 2]
    out = np.empty_like(q)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        out[..., i] = 2.0 * ((q[..., i] - q[..., j]) + q[..., k]) * ((q[..., i] - q[..., k]) + q[..., j]) / P
    return out


def pancake_rhs(y, m: float, Lam: float = 0.0, kappa: float = 1.0) -> np.ndarray:
    """
    Time derivative of the state y = (R1, R2, R3, R1', R2', R3', phi, phi').  Regular at R1 = 0: the
    H_1 = R1'/R1 factor only multiplies R2', R3', which vanish there.
    """
    R, Rd, ph, phd = y[:3], y[3:6], y[6], y[7]
    V = 0.5 * m * m * ph * ph
    r3 = slice_ricci(R)
    H = Rd / R
    Rdd = np.empty(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        # R_i H_i = R_i' , so R_i'' = -R_i' (H_j + H_k) + R_i (...): only H_1 = R_1'/R_1 can be large, and it
        # multiplies R_2', R_3' ~ t
        Rdd[i] = -Rd[i] * (H[j] + H[k]) + R[i] * (-r3[i] + kappa * V + Lam)
    phdd = -np.sum(H) * phd - m * m * ph
    return np.concatenate([Rd, Rdd, [phd, phdd]])


def constraint(y, m: float, Lam: float = 0.0, kappa: float = 1.0) -> float:
    """Hamiltonian constraint (zero on solutions) for the state y."""
    R, Rd, ph, phd = y[:3], y[3:6], y[6], y[7]
    H = Rd / R
    V = 0.5 * m * m * ph * ph
    return float(H[0] * H[1] + H[1] * H[2] + H[0] * H[2] + 0.5 * np.sum(slice_ricci(R))
                 - kappa * (0.5 * phd * phd + V) - Lam)


# --------------------------------------------------------------------------
#  Series about the pancake
# --------------------------------------------------------------------------

def pancake_series(a0: float, b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0) -> Dict[str, float]:
    """
    Next coefficients of the odd-parity series (eq. 30 of the paper), solved from the field equations:

        b2 = (2 Lam b0^2 + kappa m^2 f0^2 b0^2 - 8) / (8 b0),   f2 = -m^2 f0 / 4,
        a2 = a0 (b0 (2 Lam + kappa m^2 f0^2) - 8 b2) / (12 b0)

    The constraint holds identically at t = 0 for these (checked in the tests).
    """
    b2 = (2.0 * Lam * b0 ** 2 + kappa * m * m * f0 ** 2 * b0 ** 2 - 8.0) / (8.0 * b0)
    f2 = -m * m * f0 / 4.0
    a2 = a0 * (b0 * (2.0 * Lam + kappa * m * m * f0 ** 2) - 8.0 * b2) / (12.0 * b0)
    return dict(a0=a0, a2=a2, b0=b0, b2=b2, f0=f0, f2=f2)


def series_state(c: Dict[str, float], t: float) -> np.ndarray:
    """State (R, R', phi, phi') from the series through O(t^2) in R_2, phi and O(t^3) in R_1."""
    R1 = t * (c["a0"] + c["a2"] * t * t)
    R2 = c["b0"] + c["b2"] * t * t
    return np.array([R1, R2, R2, c["a0"] + 3.0 * c["a2"] * t * t, 2.0 * c["b2"] * t, 2.0 * c["b2"] * t,
                     c["f0"] + c["f2"] * t * t, 2.0 * c["f2"] * t])


# --------------------------------------------------------------------------
#  Solution through the pancake
# --------------------------------------------------------------------------

def pancake_solution(a0: float, b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0,
                     t_max: float = 5.0, t_min: Optional[float] = None, t_start: float = 1e-4,
                     n: int = 400) -> Dict[str, np.ndarray]:
    """
    The pancake solution on t in [t_min, t_max] (t_min defaults to -t_max), skipping the interval
    |t| < t_start around the pancake itself, where the series supplies the starting data.  Both halves are
    integrated away from the pancake on a geometric grid (so the neighbourhood of t = 0 is resolved).

    Returns arrays on a time grid t (ascending, no t = 0 point):
        t, R (n, 3), Rd (n, 3), phi, phid, Rdd (n, 3)
    R_1 is negative for t < 0 (odd parity).
    """
    t_min = -t_max if t_min is None else t_min
    if not (t_min < -t_start and t_max > t_start):
        raise ValueError("need t_min < -t_start and t_max > t_start")
    c = pancake_series(a0, b0, f0, m, Lam, kappa)

    def half(sign: float, span: float) -> Dict[str, np.ndarray]:
        grid = np.geomspace(t_start, span, n)
        s0 = series_state(c, sign * t_start)
        # in s = sign * t the state (which holds t-derivatives) obeys dy/ds = sign * rhs(y)
        sol = solve_ivp(lambda s, y: sign * pancake_rhs(y, m, Lam, kappa),
                        [t_start, span], s0, t_eval=grid, method="DOP853", rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise RuntimeError(f"pancake integration failed: {sol.message}")
        return dict(t=sign * sol.t, y=sol.y.T)

    fwd = half(+1.0, t_max)
    bwd = half(-1.0, -t_min)
    t = np.concatenate([bwd["t"][::-1], fwd["t"]])
    y = np.vstack([bwd["y"][::-1], fwd["y"]])
    Rdd = np.array([pancake_rhs(s, m, Lam, kappa)[3:6] for s in y])
    return dict(t=t, R=y[:, :3], Rd=y[:, 3:6], phi=y[:, 6], phid=y[:, 7], Rdd=Rdd,
                params=dict(a0=a0, b0=b0, f0=f0, m=m, Lam=Lam, kappa=kappa))


# --------------------------------------------------------------------------
#  Curvature
# --------------------------------------------------------------------------

def pancake_kretschmann(sol: Dict[str, np.ndarray]) -> np.ndarray:
    """Kretschmann scalar along a solution (geometry only, via qgl_bkl.bianchi9_kretschmann with a = |R|/2)."""
    R = sol["R"]
    sgn = np.sign(R)
    return bk.bianchi9_kretschmann(np.abs(R) / 2.0, sgn * sol["Rd"] / 2.0, sgn * sol["Rdd"] / 2.0)


def kretschmann_at_pancake(b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0) -> float:
    """
    Closed-form Kretschmann scalar at t = 0, from the series (a2 = 2 a0 / (3 b0^2) always):

        e = 2 b2 / b0 = Lam/2 + kappa m^2 f0^2 / 4 - 2/b0^2
        K(0) = 16 e^2 + 80 / b0^4

    Orthonormal-frame pieces: R_0202 = R_0303 and R_1212 = R_1313 equal e (each counted 4 times), R_0101 and
    R_2323 equal 4/b0^2, and the magnetic (Codazzi) part is (-2, 1, 1)/b0^2, which enters with -8.  The
    paper's eq. (31) lists only the *coordinate* components that survive at t = 0; the x-components vanish there
    only because g_xx -> 0, while their orthonormal-frame values are finite, so K needs them too.
    The value is checked against a brute-force Riemann tensor in tests/test_pancake.py.
    """
    e = 0.5 * Lam + 0.25 * kappa * m * m * f0 ** 2 - 2.0 / b0 ** 2
    return float(16.0 * e * e + 80.0 / b0 ** 4)


def shear_squared(sol: Dict[str, np.ndarray]) -> np.ndarray:
    """sigma^2 = (1/2) sigma_ij sigma^ij = (1/6) sum_{i<j} (H_i - H_j)^2; -> 1/(3 t^2) at the pancake."""
    H = sol["Rd"] / sol["R"]
    return ((H[:, 0] - H[:, 1]) ** 2 + (H[:, 0] - H[:, 2]) ** 2 + (H[:, 1] - H[:, 2]) ** 2) / 6.0


# --------------------------------------------------------------------------
#  Sensor through the pancake
# --------------------------------------------------------------------------

def pancake_sensor_run(sol: Dict[str, np.ndarray], t_start: float, t_stop: float, rate_scale: float = 1.0,
                       cfg: Optional[Config] = None, K_ref: Optional[float] = None,
                       K_trans: Optional[float] = None, v_ref: float = 0.5, n: int = 300) -> pd.DataFrame:
    """
    Sensor on the comoving worldline from t_start to t_stop (through the pancake if they straddle zero).
    The sensor clock runs at rate_scale sensor units per unit of cosmic time (comoving proper time = t).  K is
    the Kretschmann scalar interpolated on the solution; K_ref (rho_hat = 1) and K_trans (C_Q = 1/2) default to
    K(0), so C_Q = 1/2 at the pancake: a modelling choice.  Output as qgl_curvature.sensor_run_along plus t.
    """
    p = sol["params"]
    K0 = kretschmann_at_pancake(p["b0"], p["f0"], p["m"], p["Lam"], p["kappa"])
    K_ref = K0 if K_ref is None else K_ref
    K_trans = K0 if K_trans is None else K_trans
    t_full = sol["t"]
    K_full = pancake_kretschmann(sol)
    # tabulate K on a grid that is dense near t = 0; insert the exact value at t = 0 (finite, by the series)
    t_all = np.concatenate([t_full[t_full < 0], [0.0], t_full[t_full > 0]])
    K_all = np.concatenate([K_full[t_full < 0], [K0], K_full[t_full > 0]])
    lo, hi = sorted((t_start, t_stop))
    mask = (t_all >= lo) & (t_all <= hi)
    t_use, K_use = t_all[mask], K_all[mask]
    if len(t_use) > n:
        keep = np.flatnonzero(t_use == 0.0)                      # the exact pancake point always stays
        idx = np.unique(np.concatenate([np.linspace(0, len(t_use) - 1, n).astype(int), keep]))
        t_use, K_use = t_use[idx], K_use[idx]
    if t_stop < t_start:
        t_use, K_use = t_use[::-1], K_use[::-1]
    tau = np.abs(t_use - t_use[0]) * rate_scale
    df = qc.sensor_run_along(tau, K_use, np.zeros_like(tau), cfg=cfg, K_ref=K_ref, K_trans=K_trans, v_ref=v_ref)
    df.insert(1, "t", t_use)
    return df


def main() -> None:
    a0, b0, f0, m = 1.0, 3.0, 2.0, 0.5
    sol = pancake_solution(a0, b0, f0, m, t_max=6.0, n=300)
    K = pancake_kretschmann(sol)
    print(f"pancake: a0={a0}, b0={b0}, f0={f0}, m={m}; K(0) = {kretschmann_at_pancake(b0, f0, m):.6g}")
    i0 = np.argmin(np.abs(sol["t"]))
    for j in (0, len(K) // 4, i0 - 1, i0, len(K) - 1):
        t = sol["t"][j]
        print(f"t={t:+10.3e}  R1={sol['R'][j, 0]:+10.3e}  R2={sol['R'][j, 1]:9.4f}  phi={sol['phi'][j]:7.4f}  "
              f"K={K[j]:.6g}  t^2*shear^2={t * t * shear_squared(sol)[j]:.4f}")
    df = pancake_sensor_run(sol, -4.0, 4.0, rate_scale=4.0)
    print(df.iloc[::50][["t", "tau", "K", "C_Q", "omega_c", "n", "sigma_z"]].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
