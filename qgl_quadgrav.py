#!/usr/bin/env python3
"""
qgl_quadgrav.py -- where would a Weyl^2 term matter?  A curvature probe for quadratic gravity (R + R^2 + Weyl^2).

Context: Stelle (1977) showed quadratic gravity is renormalizable; K. S. Kumar and J. Marto, arXiv:2603.07150, propose
treating its massive spin-2 mode as an inverted-harmonic-oscillator-like instability (QUANTGRAV.md, 2.4).  This module
does NOT implement quadratic gravity and solves none of its field equations.  It only computes, on the geometries the
repository already has, the invariants such a theory is built from:

    Riem^2 = K,   Ric^2 = R_ab R^ab,   R,
    C^2    = K - 2 Ric^2 + R^2/3          (Weyl squared, exact in 4D),
    GB     = K - 4 Ric^2 + R^2            (Gauss-Bonnet density, a total derivative in 4D),

so that the two non-topological quadratic terms are R^2 and C^2.  Findings (all checked in tests/test_quadgrav.py):

  * flat FLRW is conformally flat: C^2 = 0 identically, for any H(t), so a Weyl^2 term has no density to act on in the
    homogeneous isotropic background; only R^2 does.
  * vacuum Kasner / Mixmaster: Ric = 0, so C^2 = K (Kasner: -16 p1 p2 p3 / t^4); R^2 vanishes and Weyl^2 is all there is.
  * the pancake (scalar-field Bianchi IX): C^2 and Ric^2 are both non-zero; the ratio varies along the history.

Frame: orthonormal, signature (-+++), diagonal Bianchi IX (scale factors a_i, first and second derivatives).  For
this class R_0i = 0 (the momentum constraint is empty), and

    R_00 = -sum a_i''/a_i,      R_ii = a_i''/a_i + H_i (S - H_i) + R3_ii,      H_i = a_i'/a_i,  S = sum H_j,
    R = -R_00 + sum R_ii,       Ric^2 = R_00^2 + sum R_ii^2.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

import qgl_bkl as bk


# ---------------------------------------------------------------- algebra of the invariants

def weyl_squared(K, ric_sq, R):
    """C_abcd C^abcd = Riem^2 - 2 Ric^2 + R^2/3 (4D)."""
    K, ric_sq, R = (np.asarray(v, dtype=float) for v in (K, ric_sq, R))
    return K - 2.0 * ric_sq + R ** 2 / 3.0


def gauss_bonnet_density(K, ric_sq, R):
    """Riem^2 - 4 Ric^2 + R^2 (Euler density; topological in 4D)."""
    K, ric_sq, R = (np.asarray(v, dtype=float) for v in (K, ric_sq, R))
    return K - 4.0 * ric_sq + R ** 2


def weyl_fraction(C2, ric_sq):
    """|C^2| / (|C^2| + Ric^2): 1 where the geometry is pure Weyl curvature (vacuum), 0 where it is conformally flat."""
    C2, ric_sq = np.abs(np.asarray(C2, dtype=float)), np.asarray(ric_sq, dtype=float)
    den = C2 + ric_sq
    return np.where(den > 0, C2 / np.where(den > 0, den, 1.0), 0.0)


# ---------------------------------------------------------------- flat FLRW

def flrw_invariants(H, Hdot) -> Dict[str, np.ndarray]:
    """K, Ric^2, R, C^2, GB for flat FLRW from H and Hdot.  R = 6 (Hdot + 2 H^2);
    R_00 = -3 (Hdot + H^2), R_ij = (Hdot + 3 H^2) g_ij, so Ric^2 = 9 (Hdot + H^2)^2 + 3 (Hdot + 3 H^2)^2."""
    H, Hdot = np.asarray(H, dtype=float), np.asarray(Hdot, dtype=float)
    K = 12.0 * ((Hdot + H ** 2) ** 2 + H ** 4)
    R = 6.0 * (Hdot + 2.0 * H ** 2)
    ric_sq = 9.0 * (Hdot + H ** 2) ** 2 + 3.0 * (Hdot + 3.0 * H ** 2) ** 2
    return dict(K=K, ric_sq=ric_sq, R=R, C2=weyl_squared(K, ric_sq, R), GB=gauss_bonnet_density(K, ric_sq, R))


# ---------------------------------------------------------------- diagonal Bianchi IX (Kasner, Mixmaster, pancake)

def bianchi_ricci(a, ad, add):
    """(R_00, R_ii) in the orthonormal frame; R_ii has shape (..., 3)."""
    a, ad, add = (np.asarray(v, dtype=float) for v in (a, ad, add))
    H = ad / a
    S = H.sum(axis=-1, keepdims=True)
    R00 = -np.sum(add / a, axis=-1)
    Rii = add / a + H * (S - H) + bk.ricci3(a)
    return R00, Rii


def bianchi_invariants(a, ad, add) -> Dict[str, np.ndarray]:
    """K, Ric^2, R, C^2, GB for diagonal Bianchi IX from scale factors and their first two time derivatives."""
    R00, Rii = bianchi_ricci(a, ad, add)
    R = -R00 + Rii.sum(axis=-1)
    ric_sq = R00 ** 2 + np.sum(Rii ** 2, axis=-1)
    K = bk.bianchi9_kretschmann(a, ad, add)
    return dict(K=K, ric_sq=ric_sq, R=R, C2=weyl_squared(K, ric_sq, R), GB=gauss_bonnet_density(K, ric_sq, R))


def kasner_invariants(p, t) -> Dict[str, np.ndarray]:
    """Bianchi I limit (no curvature term): a_i = t^p_i.  Use for any exponents; vacuum only if sum p = sum p^2 = 1."""
    p = np.asarray(p, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
    a = t ** p
    ad = p * t ** (p - 1.0)
    add = p * (p - 1.0) * t ** (p - 2.0)
    R00, Rii = bianchi_ricci_flat(a, ad, add)
    R = -R00 + Rii.sum(axis=-1)
    ric_sq = R00 ** 2 + np.sum(Rii ** 2, axis=-1)
    H = ad / a
    K = 4.0 * np.sum((add / a) ** 2, axis=-1)
    for i in range(3):
        j = (i + 1) % 3
        K = K + 4.0 * (H[..., i] * H[..., j]) ** 2
    return dict(K=K, ric_sq=ric_sq, R=R, C2=weyl_squared(K, ric_sq, R), GB=gauss_bonnet_density(K, ric_sq, R))


def bianchi_ricci_flat(a, ad, add):
    """Same as bianchi_ricci but for Bianchi I (flat slices: R3 = 0)."""
    H = ad / a
    S = H.sum(axis=-1, keepdims=True)
    return -np.sum(add / a, axis=-1), add / a + H * (S - H)


# ---------------------------------------------------------------- figure

def plot_probe(path: str) -> None:
    """Three panels: C^2 / K along flat LCDM and vacuum Kasner; C^2, Ric^2 and K across the pancake; |C^2| across a
    Mixmaster bounce."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import qgl_flrw as fl
    import qgl_pancake as qp

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) LCDM vs Kasner
    a_grid = np.geomspace(1e-3, 1e1, 400)
    E2 = lambda a: fl.lcdm_E2(a, Om=0.3, Or=1e-4)
    d = fl.dlnE2_dlna_numeric(E2, a_grid)
    H = np.sqrt(E2(a_grid))
    inv = flrw_invariants(H, 0.5 * H ** 2 * d)
    ax[0].semilogx(a_grid, inv["C2"] / inv["K"], color="tab:blue", label=r"flat $\Lambda$CDM: $C^2/K$")
    ax[0].axhline(1.0, color="tab:red", label=r"vacuum Kasner: $C^2/K = 1$")
    ax[0].set_ylim(-0.2, 1.2)
    ax[0].set_xlabel("scale factor a (LCDM)")
    ax[0].set_title(r"FLRW is conformally flat: $C^2 = 0$ to roundoff", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8, loc="center right")
    # (b) pancake
    sol = qp.pancake_solution(1.0, 3.0, 2.0, 0.5, 0.1, t_max=5.0, n=300)
    sg = np.sign(sol["R"])
    pin = bianchi_invariants(np.abs(sol["R"]) / 2.0, sg * sol["Rd"] / 2.0, sg * sol["Rdd"] / 2.0)
    for key, lab, col in (("K", r"$K$", "k"), ("C2", r"$C^2$", "tab:red"), ("ric_sq", r"$\mathrm{Ric}^2$", "tab:blue")):
        ax[1].plot(sol["t"], pin[key], label=lab, color=col, lw=1.6)
    ax[1].set_yscale("symlog", linthresh=1e-2)
    ax[1].set_xlabel("t (pancake at t = 0)")
    ax[1].set_title("pancake (a0=1, b0=3, f0=2, m=0.5, Lambda=0.1): Weyl and Ricci both finite", fontsize=9)
    ax[1].legend(frameon=False, fontsize=8)
    # (c) Mixmaster
    mm = bk.Mixmaster(3.7, t0=1.0e6, t_bounce=1.0, t_end=1.0e-12)
    t = np.geomspace(mm.t0, mm.t_end, 1500)
    K, Ks = mm.weyl_invariants(t)
    ax[2].loglog(t, np.abs(K), label=r"$|C^2|$", color="tab:red")
    ax[2].loglog(t, np.hypot(K, Ks), label=r"$\sqrt{K^2+K_*^2}$", color="0.4", ls="--")
    ax[2].invert_xaxis()
    ax[2].set_xlabel("time to singularity t")
    ax[2].set_title(r"vacuum Mixmaster: $\mathrm{Ric}=0$, so $C^2 = K$; dips at bounces", fontsize=9)
    ax[2].legend(frameon=False, fontsize=8)
    for q in ax:
        q.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    H = np.array([1.0, 0.5])
    inv = flrw_invariants(H, np.array([-0.3, -0.1]))
    print("flat FLRW  C^2 =", inv["C2"], " (conformally flat)")
    k = kasner_invariants([-1 / 3, 2 / 3, 2 / 3], 1.0)
    print("Kasner (-1/3, 2/3, 2/3) at t=1: K =", k["K"], " Ric^2 =", k["ric_sq"], " C^2 =", k["C2"])


if __name__ == "__main__":
    import os
    import sys
    if "--figure" in sys.argv:
        os.makedirs("figures", exist_ok=True)
        plot_probe(os.path.join("figures", "qgl_quadgrav.png"))
    else:
        main()
