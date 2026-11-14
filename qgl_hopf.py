"""
qgl_hopf.py - the Hopf fibration / Berger sphere reading of the biaxial pancake, on the numerical solution.

The spatial slice of the biaxial Bianchi IX pancake (R_2 = R_3) is S^3 with the metric

    (1/4) [ R_1^2 (d psi + cos(th) d phi)^2 + R_2^2 (d th^2 + sin^2(th) d phi^2) ] ,

i.e. the Hopf fibration S^1 -> S^3 -> S^2 with the fibre scaled by s = R_1 / R_2 (a Berger sphere) and an overall scale
R_2.  Near the pancake s -> 0: the Hopf fibres shrink to zero length while the base S^2 stays at radius R_2 / 2.

This module follows that picture along the real solution of qgl_pancake.pancake_solution (not only its leading order):

  * s(t) = R_1 / R_2,
  * the slice Ricci curvature of the solution against the Berger closed form (2 s^2, 4 - 2 s^2, 4 - 2 s^2) / R_2^2
    (identical by construction of the metric, so this is a check of qgl_bkl.ricci3, not a new result),
  * the winding of a light ray round the fibre, int 2 dt / R_1 / (4 pi) turns, against the leading order
    (2 / (4 pi a0)) ln(t / t_in),
  * the local number of e-folds of t per fibre turn, 2 pi R_1 / t, which tends to 2 pi a0.

What it does not claim: the pancake solution of qgl_pancake is leading order, not a solution of the field equations to all
orders; the S^3 (period 4 pi) identification is checked in tests/test_s3_identifications.py, whether the paper uses S^3
or SO(3) is not.  The Hopf-fibration repository (github.com/brentharts/HopfFibration, hopf_berger.py, hopf_pancake.py)
has the same formulas written independently; its cross-check against this module is listed in the README.

    python3 qgl_hopf.py            print a summary
    python3 qgl_hopf.py --figure   write figures/qgl_hopf.png
    python3 qgl_hopf.py --figure3d write figures/qgl_pancake_3d.png (the slice in 3D, at and just after t = 0)
"""
from __future__ import annotations

from typing import Dict

import numpy as np

import qgl_horizon as hz
import qgl_pancake as qp

_PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]   # Okabe-Ito


def berger_parameter(sol: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """t > 0 part of a pancake solution: t, R_1, R_2 and the squashing s = R_1 / R_2 (s = 1 is the round S^3)."""
    pos = sol["t"] > 0
    R = sol["R"][pos]
    return dict(t=sol["t"][pos], R1=R[:, 0], R2=R[:, 1], s=R[:, 0] / R[:, 1])


def berger_ricci_closed(s, R2):
    """Ricci curvatures (fibre, base, base) of the Berger sphere of squashing s and base scale R_2:
    (2 s^2, 4 - 2 s^2, 4 - 2 s^2) / R_2^2.  Unit S^3 at s = R_2 = 1."""
    s = np.asarray(s, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    return np.stack([2 * s * s, 4 - 2 * s * s, 4 - 2 * s * s], axis=-1) / (R2 ** 2)[..., None]


def slice_ricci_along(sol: Dict[str, np.ndarray]) -> np.ndarray:
    """Ricci curvatures of the slice at every t > 0 of the solution, from qgl_pancake.slice_ricci, shape (n, 3)."""
    pos = sol["t"] > 0
    return np.array([qp.slice_ricci(r) for r in sol["R"][pos]])


def fibre_turns_leading(a0: float, t, t_in: float):
    """Leading-order turns of the fibre by light between t_in and t: (2 / a0) ln(t / t_in) / (4 pi)."""
    return hz.fibre_log_coefficient(a0) * np.log(np.asarray(t) / t_in) / hz.FIBRE_PERIOD


def efolds_per_turn(R1, t):
    """Local e-folds of t per turn of the fibre: d ln t = 4 pi R_1 / (2 t) = 2 pi R_1 / t.  Tends to 2 pi a0 as t -> 0.
    (The Hopf-fibration repository's hopf_berger.levels_per_turn is the same number from the other side.)"""
    return 2.0 * np.pi * np.asarray(R1) / np.asarray(t)


def summary(a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5, t_max: float = 6.0) -> Dict[str, float]:
    sol = qp.pancake_solution(a0, b0, f0, m, t_max=t_max)
    b = berger_parameter(sol)
    ric = slice_ricci_along(sol)
    scalar = ric.sum(axis=1)
    h = hz.pancake_horizons(sol)
    i_round = int(np.argmax(b["s"] >= 1.0)) if np.any(b["s"] >= 1.0) else -1
    return dict(s_first=float(b["s"][0]), s_last=float(b["s"][-1]),
                t_round=float(b["t"][i_round]) if i_round >= 0 else float("nan"),
                scalar_first_times_b0sq=float(scalar[0] * b0 ** 2),
                efolds_per_turn_first=float(efolds_per_turn(b["R1"][0], b["t"][0])),
                two_pi_a0=2 * np.pi * a0,
                fibre_turns_last=float(h["fibre_turns"][-1]),
                fibre_turns_leading_last=float(fibre_turns_leading(a0, b["t"][-1], b["t"][0])))


def plot_hopf(path: str, a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5) -> None:
    """Three panels: squashing s(t); slice Ricci curvature, true and leading order; fibre turns and e-folds per turn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sol = qp.pancake_solution(a0, b0, f0, m, t_max=6.0, n=800)
    b = berger_parameter(sol)
    t = b["t"]
    ric = slice_ricci_along(sol)
    h = hz.pancake_horizons(sol)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    a = ax[0]
    a.plot(t, b["s"], color=_PAL[1], lw=1.8, label=r"$s=R_1/R_2$ (numerical solution)")
    a.plot(t, a0 * t / b0, color=_PAL[1], lw=1.2, ls="--", label=r"leading order $a_0 t/b_0$")
    a.axhline(1.0, color="0.4", lw=0.9)
    a.text(t[0] * 1.5, 1.12, r"$s=1$: round $S^3$", fontsize=8, color="0.3")
    a.set(xscale="log", yscale="log", xlabel="t after the pancake", ylabel="squashing of the Hopf fibre",
          title="The fibre collapses, the base does not")
    a.legend(frameon=False, fontsize=8, loc="lower right")

    a = ax[1]
    names = ["fibre", "base", "base"]
    for k, (col, ls) in enumerate(zip((_PAL[1], _PAL[0]), ("-", "-"))):
        a.plot(t, ric[:, k] * b0 ** 2, color=col, lw=1.8, ls=ls, label=f"Ricci, {names[k]} direction (solution)")
    clo = berger_ricci_closed(b["s"], b["R2"]) * b0 ** 2
    a.plot(t, clo[:, 0], color="k", lw=1.0, ls=":", label=r"Berger closed form $(2s^2, 4-2s^2)/R_2^2$")
    a.plot(t, clo[:, 1], color="k", lw=1.0, ls=":")
    a.plot(t, ric.sum(axis=1) * b0 ** 2, color=_PAL[2], lw=1.6, label=r"scalar curvature (finite at the pancake, $\to 8/b_0^2$)")
    a.set(xscale="log", xlabel="t after the pancake", ylabel=r"curvature $\times\, b_0^2$",
          title="Slice curvature stays finite")
    a.legend(frameon=False, fontsize=8, loc="center left")

    a = ax[2]
    a.plot(t, h["fibre_turns"], color=_PAL[1], lw=1.8, label=f"turns of a light ray round the fibre (from $t_{{in}}={t[0]:.0e}$)")
    a.plot(t, fibre_turns_leading(a0, t, t[0]), color=_PAL[1], lw=1.1, ls="--", label=r"leading order $(2/a_0)\ln(t/t_{in})/4\pi$")
    a.set(xscale="log", xlabel="t after the pancake", ylabel="turns of the fibre", title="Winding and e-folds per turn")
    a2 = a.twinx()
    a2.plot(t, efolds_per_turn(b["R1"], t), color=_PAL[0], lw=1.6, label=r"e-folds of $t$ per turn, $2\pi R_1/t$")
    a2.axhline(2 * np.pi * a0, color=_PAL[0], lw=0.9, ls=":")
    a2.set_ylabel("e-folds of t per fibre turn")
    l1, n1 = a.get_legend_handles_labels()
    l2, n2 = a2.get_legend_handles_labels()
    a.legend(l1 + l2, n1 + n2, frameon=False, fontsize=8, loc="upper right")
    for q in ax:
        q.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3D picture of the slice at and near the pancake

def fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly evenly spread unit vectors, shape (n, 3)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def tangent_frame(n: np.ndarray):
    """Two orthonormal vectors spanning the tangent plane of the unit sphere at n."""
    n = np.asarray(n, dtype=float)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def fibre_circle(n: np.ndarray, base_radius: float, fibre_radius: float, m: int = 48) -> np.ndarray:
    """The Hopf fibre over the base point `base_radius * n`, drawn as a circle of radius `fibre_radius` in the tangent
    plane there (the unit-tangent-bundle picture of S^3 / Z_2 = SO(3) over S^2), shape (m, 3).  This is an illustration
    in R^3, not an isometric embedding: the fibre circle has the right length 2 pi R_1 and the base sphere the right
    radius R_2 / 2, but a fibre would not really lie in the tangent plane."""
    e1, e2 = tangent_frame(n)
    u = np.linspace(0.0, 2.0 * np.pi, m)
    centre = base_radius * np.asarray(n, dtype=float)
    return centre + fibre_radius * (np.cos(u)[:, None] * e1 + np.sin(u)[:, None] * e2)


def pancake_slice_radii(t: float, a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5,
                        Lam: float = 0.1) -> Dict[str, float]:
    """(R_1, R_2) of the biaxial pancake at t >= 0: exactly (0, b_0) at t = 0, otherwise from qgl_pancake."""
    if t <= 0.0:
        return dict(R1=0.0, R2=float(b0), s=0.0)
    y = qp.pancake_state_at(t, a0, b0, f0, m, Lam)
    return dict(R1=float(y[0]), R2=float(y[1]), s=float(y[0] / y[1]))


def plot_pancake_3d(path: str, a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5, Lam: float = 0.1,
                    times=(0.0, 0.15, 0.4, 1.0), n_fibres: int = 30) -> None:
    """The pancake slice in 3D.  Four snapshots (t = 0 and three later times, left to right, top row then bottom-left):
    the base S^2 of radius R_2 / 2 with the Hopf fibre over `n_fibres` base points drawn as circles of radius R_1 (true
    proper size).  At t = 0 the fibres have shrunk to points and only the base sphere is left: a 3-space that has
    collapsed to a 2-sphere, with finite curvature.  Bottom right: the history in (R_1, R_2, phi) space, odd R_1 through
    the pancake at (0, b_0, f_0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9])
    pts = fibonacci_sphere(n_fibres)
    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 48), np.linspace(0, np.pi, 24))
    cpal = plt.get_cmap("viridis")
    lim = 0.5 * b0 + 1.2
    for k, t in enumerate(times):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        st = pancake_slice_radii(t, a0, b0, f0, m, Lam)
        base = 0.5 * st["R2"]
        ax.plot_surface(base * np.cos(uu) * np.sin(vv), base * np.sin(uu) * np.sin(vv), base * np.cos(vv),
                        color="0.85", alpha=0.35, linewidth=0, shade=True)
        if st["R1"] > 0.0:
            for p in pts:
                c = fibre_circle(p, base, st["R1"])
                ax.plot(c[:, 0], c[:, 1], c[:, 2], color=cpal(0.5 * (p[2] + 1.0)), lw=1.0)
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.55)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=35)
        scal = (8.0 - 2.0 * st["s"] ** 2) / st["R2"] ** 2
        ax.set_title(f"t = {t:g}:  R$_1$ = {st['R1']:.3f},  R$_2$ = {st['R2']:.3f}\n"
                     f"fibre length 2$\\pi$R$_1$ = {2 * np.pi * st['R1']:.2f};  slice scalar curvature {scal:.3f}",
                     fontsize=9)
    sol = qp.pancake_solution(a0, b0, f0, m, Lam, t_max=5.0, n=300)
    near = np.abs(sol["t"]) <= 2.5
    t_, R_, ph_ = sol["t"][near], sol["R"][near], sol["phi"][near]
    ax = fig.add_subplot(gs[1, 0:2], projection="3d")
    ax.plot(R_[:, 0], R_[:, 1], ph_, color=_PAL[1], lw=2.0)
    ax.scatter([0.0], [b0], [f0], color="k", s=45, zorder=5, label="pancake, t = 0")
    yy, zz = np.meshgrid(np.linspace(R_[:, 1].min() * 0.98, R_[:, 1].max() * 1.02, 2),
                         np.linspace(ph_.min(), ph_.max(), 2))
    ax.plot_surface(np.zeros_like(yy), yy, zz, color="0.7", alpha=0.25, linewidth=0)
    ax.set_xlabel("R$_1$ (odd in t)"), ax.set_ylabel("R$_2$ = R$_3$"), ax.set_zlabel(r"$\phi$")
    ax.locator_params(nbins=4)
    ax.view_init(elev=20, azim=-55)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("history for |t| <= 2.5 in (R$_1$, R$_2$, $\\phi$): R$_1$ changes sign at the plane R$_1$ = 0", fontsize=9)
    a2 = fig.add_subplot(gs[1, 2:4])
    a2.plot(t_, R_[:, 0], color=_PAL[1], lw=1.8, label="R$_1$ (fibre scale; fibre length = 2$\\pi$R$_1$)")
    a2.plot(t_, R_[:, 1], color=_PAL[0], lw=1.8, label="R$_2$ = R$_3$ (base scale; base radius R$_2$/2)")
    a2.plot(t_, np.abs(R_[:, 0]) * R_[:, 1] ** 2 / b0 ** 2, color=_PAL[2], lw=1.4, ls="--",
            label="volume $\\propto |R_1| R_2^2$ (in units of $b_0^2$)")
    a2.axvline(0.0, color="0.4", lw=0.9)
    a2.set_xlabel("t"), a2.set_title("one scale factor goes linearly through zero; the other stays near b$_0$", fontsize=9)
    a2.legend(frameon=False, fontsize=8, loc="center left")
    a2.grid(alpha=0.25)
    k0 = qp.kretschmann_at_pancake(b0, f0, m, Lam, 1.0, a0)
    fig.suptitle(f"Pancake at t = 0 in 3D: the closed direction collapses, the volume vanishes, curvature is finite "
                 f"(Kretschmann K(0) = {k0:.3f};  a0 = {a0:g}, b0 = {b0:g}, f0 = {f0:g}, m = {m:g}, "
                 f"$\\Lambda$ = {Lam:g})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    s = summary()
    print("Pancake as a Berger sphere (a0=1, b0=3, f0=2, m=0.5):")
    print(f"  s = R1/R2: {s['s_first']:.2e} at the first grid time, {s['s_last']:.3f} at t=6; round (s=1) at t = {s['t_round']:.2f}")
    print(f"  slice scalar curvature x b0^2 at the first grid time: {s['scalar_first_times_b0sq']:.6f}   (8 as s -> 0)")
    print(f"  e-folds of t per fibre turn at the first grid time: {s['efolds_per_turn_first']:.4f}   (2 pi a0 = {s['two_pi_a0']:.4f})")
    print(f"  fibre turns by t=6: {s['fibre_turns_last']:.3f}; leading order {s['fibre_turns_leading_last']:.3f}")


if __name__ == "__main__":
    import os
    import sys
    if "--figure" in sys.argv:
        os.makedirs("figures", exist_ok=True)
        plot_hopf(os.path.join("figures", "qgl_hopf.png"))
    elif "--figure3d" in sys.argv:
        os.makedirs("figures", exist_ok=True)
        plot_pancake_3d(os.path.join("figures", "qgl_pancake_3d.png"))
    else:
        main()
