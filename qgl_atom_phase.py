#!/usr/bin/env python3
"""
qgl_atom_phase.py -- the gravitational phase of a freely falling atom in a light-pulse (Mach-Zehnder) interferometer.

Textbook model (Kasevich-Chu): three laser pulses at t = 0, T, 2T split, redirect and recombine an atom of mass m; the
effective wavevector is k.  In a uniform acceleration g the phase is  phi = k g T^2.  This module computes it two ways:

  * analytically (phase_uniform), and in the Colella-Overhauser-Werner form m g delta T / hbar with arm separation
    delta = hbar k T / m (phase_cow), where the mass cancels;
  * numerically from the classical arm trajectories: the propagation phase (S_upper - S_lower)/hbar from the action,
    plus the laser phases phi_L(0) - phi_L(T, z_u) - phi_L(T, z_l) + phi_L(2T) with phi_L(z) = -k z, plus the separation
    phase -(p_u + p_l) dz / 2 hbar (zero unless a gradient keeps the arms from meeting exactly).  The sign convention
    is chosen so a downward acceleration g gives +k g T^2.  With a gradient the propagation and separation phases are each
    ~1e-2 rad but cancel, leaving the laser term, -(7/12) k gamma T^3 (g T - v_r) for an atom released from rest (the
    v_r^2 correction is ~0.2%).  The numerical route also handles a linear gradient
    g(z) = g0 + gamma z (acceleration -g(z) along z) and an equivalence-principle ratio eta = m_grav / m_inert.

What this is not: a model of the Quantum Galileo Interferometer experiment (QUANTGRAV.md, 2.3).  I read only press
coverage of that work, so its geometry (one arm held by magnetic fields) and numbers are not reproduced here.  Nor is
there a mapping from this phase to the Jaynes-Cummings sensor of `unified_*`: an overlay would be invented, not derived.
Consistent with the equivalence principle is all this shows; the phase says nothing about gravity being quantum.
"""
from __future__ import annotations

import numpy as np

HBAR = 1.054571817e-34            # J s
M_RB87 = 1.443160648e-25          # kg
LAMBDA_RB = 780.241e-9            # m, Rb D2 line
G_EARTH = 9.80665                 # m / s^2


def k_eff(wavelength: float = LAMBDA_RB) -> float:
    """Two-photon (counter-propagating Raman/Bragg) effective wavevector 2 * 2 pi / lambda."""
    return 2.0 * 2.0 * np.pi / wavelength


def recoil_velocity(m: float, k: float) -> float:
    return HBAR * k / m


def phase_uniform(k: float, g: float, T: float) -> float:
    """phi = k g T^2."""
    return k * g * T ** 2


def arm_separation(m: float, k: float, T: float) -> float:
    """delta = hbar k T / m: the largest separation of the two arms (recoil velocity times T)."""
    return recoil_velocity(m, k) * T


def phase_cow(m: float, g: float, k: float, T: float) -> float:
    """Colella-Overhauser-Werner form: m g delta T / hbar.  Equals k g T^2 because delta is proportional to 1/m."""
    return m * g * arm_separation(m, k, T) * T / HBAR


def _arm_path(m: float, k: float, T: float, g0: float, gamma: float, eta: float, z0: float, v0: float,
              upper: bool, n: int):
    """Position/velocity at t = T and 2T and the action, for one arm.  Acceleration a(z) = -eta (g0 + gamma z); the
    kicks are +-v_r at the pulses.  Upper arm: +v_r at 0, -2 v_r... i.e. velocity v0 + v_r after pulse 1 and v0 after
    pulse 2.  Lower arm: v0 after pulse 1 and v0 + v_r after pulse 2.  Lagrangian L = m v^2/2 - eta m (g0 z + gamma z^2/2)."""
    vr = recoil_velocity(m, k)

    def segment(z, v, dur):
        # velocity-Verlet with a fine step; also accumulates the action
        h = dur / n
        S = 0.0
        a = lambda zz: -eta * (g0 + gamma * zz)
        for _ in range(n):
            a0 = a(z)
            z_new = z + v * h + 0.5 * a0 * h * h
            v_new = v + 0.5 * (a0 + a(z_new)) * h
            # Simpson-free midpoint action using the endpoint average of L
            L0 = 0.5 * m * v * v - eta * m * (g0 * z + 0.5 * gamma * z * z)
            L1 = 0.5 * m * v_new * v_new - eta * m * (g0 * z_new + 0.5 * gamma * z_new * z_new)
            S += 0.5 * (L0 + L1) * h
            z, v = z_new, v_new
        return z, v, S

    v1 = v0 + (vr if upper else 0.0)
    z_T, v_T, S1 = segment(z0, v1, T)
    v2 = v_T + (-vr if upper else +vr)
    z_2T, v_2T, S2 = segment(z_T, v2, T)
    return dict(z_T=z_T, z_2T=z_2T, v_2T=v_2T, S=S1 + S2)


def mz_phase_numeric(m: float = M_RB87, k: float = None, T: float = 0.1, g0: float = G_EARTH, gamma: float = 0.0,
                     eta: float = 1.0, z0: float = 0.0, v0: float = 0.0, n: int = 2000) -> dict:
    """Numerical Mach-Zehnder phase.  Returns dict(total, propagation, laser).  At gamma = 0 propagation is ~0 and
    laser = eta k g0 T^2."""
    k = k_eff() if k is None else k
    up = _arm_path(m, k, T, g0, gamma, eta, z0, v0, True, n)
    lo = _arm_path(m, k, T, g0, gamma, eta, z0, v0, False, n)
    phi_L = lambda z: -k * z
    laser = phi_L(z0) - phi_L(up["z_T"]) - phi_L(lo["z_T"]) + phi_L(0.5 * (up["z_2T"] + lo["z_2T"]))
    prop = (up["S"] - lo["S"]) / HBAR
    dz = up["z_2T"] - lo["z_2T"]
    sep = -0.5 * m * (up["v_2T"] + lo["v_2T"]) * dz / HBAR          # separation phase: the arms need not meet exactly
    return dict(total=prop + laser + sep, propagation=prop, laser=laser, separation=sep, separation_end=dz)


def eotvos_ratio_from_phases(dphi: float, k: float, g: float, T: float) -> float:
    """eta_1 - eta_2 from the differential phase of two species (same k, T): dphi = (eta_1 - eta_2) k g T^2."""
    return dphi / phase_uniform(k, g, T)


# ---------------------------------------------------------------- figure

def plot_phase(path: str) -> None:
    """(a) phase vs T: numerical against k g T^2; (b) phase vs mass at fixed k (flat) against arm separation (falls);
    (c) the gradient correction: phi(gamma) - phi(0) vs gamma."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = k_eff()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    Ts = np.array([0.02, 0.04, 0.08, 0.16, 0.32])
    num = np.array([mz_phase_numeric(T=t, n=800)["total"] for t in Ts])
    ax[0].loglog(Ts, phase_uniform(k, G_EARTH, Ts), color="tab:blue", label=r"$k g T^2$")
    ax[0].loglog(Ts, num, "o", color="tab:red", label="action + laser phases")
    ax[0].set_xlabel("T (s)"), ax[0].set_ylabel("phase (rad)")
    ax[0].set_title("Mach-Zehnder phase scales as T^2", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)
    masses = np.array([39, 87, 133]) * 1.66053907e-27
    ax[1].bar(range(3), [phase_cow(m, G_EARTH, k, 0.1) for m in masses], color="tab:blue", width=0.35,
              label="phase (K, Rb, Cs)")
    ax2 = ax[1].twinx()
    ax2.bar(np.arange(3) + 0.4, [arm_separation(m, k, 0.1) * 1e3 for m in masses], color="tab:orange", width=0.35,
            label="arm separation (mm)")
    ax[1].set_xticks(np.arange(3) + 0.2), ax[1].set_xticklabels(["K-39", "Rb-87", "Cs-133"])
    ax[1].set_ylabel("phase (rad)"), ax2.set_ylabel("arm separation (mm)")
    ax[1].set_title("mass cancels: phase is the same, the arms are not", fontsize=9)
    gam = np.array([-4e-6, -2e-6, 0.0, 2e-6, 4e-6])
    base = mz_phase_numeric(T=0.1, gamma=0.0, n=800)["total"]
    dif = np.array([mz_phase_numeric(T=0.1, gamma=gg, n=800)["total"] - base for gg in gam])
    ax[2].plot(gam * 1e6, dif, "o-", color="tab:green")
    ax[2].set_xlabel(r"gradient $\gamma = \partial g/\partial z$ ($10^{-6}\,\mathrm{s^{-2}}$)")
    ax[2].set_ylabel(r"$\phi(\gamma) - \phi(0)$ (rad)")
    ax[2].plot(gam * 1e6, -(7.0 / 12.0) * k * gam * 0.1 ** 3 * (G_EARTH * 0.1 - recoil_velocity(M_RB87, k)),
               "--", color="k", label=r"$-\frac{7}{12} k\gamma T^3 (gT - v_r)$")
    ax[2].legend(frameon=False, fontsize=8)
    ax[2].set_title("gradient correction (T = 0.1 s, atom released from rest)", fontsize=9)
    for q in ax:
        q.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    k, T = k_eff(), 0.1
    print(f"k_eff = {k:.4e} 1/m;  T = {T} s;  k g T^2 = {phase_uniform(k, G_EARTH, T):.6e} rad")
    r = mz_phase_numeric(T=T)
    print(f"numerical: total {r['total']:.6e}  propagation {r['propagation']:.3e}  laser {r['laser']:.6e}")
    print(f"COW form m g delta T / hbar = {phase_cow(M_RB87, G_EARTH, k, T):.6e}")


if __name__ == "__main__":
    import os
    import sys
    if "--figure" in sys.argv:
        os.makedirs("figures", exist_ok=True)
        plot_phase(os.path.join("figures", "qgl_atom_phase.png"))
    else:
        main()
