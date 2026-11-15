#!/usr/bin/env python3
"""
qgl_cq.py -- what backreaction on the classical orbit would cost: a decoherence-diffusion trade-off for a qubit that
pushes on a classical momentum.

Context.  QGL drives a quantum sensor (a Jaynes-Cummings atom, `unified_*`) with a classical orbit and lets nothing flow
back (QGL.md, 1.6).  If the sensor were to push on the orbit, the joint dynamics would be classical-quantum (CQ).
Oppenheim, Sparaciari, Soda and Weller-Davies (Nat. Commun. 14, 7910 (2023), arXiv:2203.01982) prove that completely
positive, Markovian CQ dynamics with such a force must both decohere the quantum system and make the classical variable
diffuse, with a trade-off: weak decoherence costs large diffusion.  This module builds a toy that shows the trade-off
in one place, and measures it.  It is a port of G11 of brentharts/quantum-gravity (python/qg/cq_tradeoff.py), extended
with a measurement efficiency eta.

Model.  A qubit's sigma_z is measured continuously with strength k and efficiency eta (stochastic master equation):

    d rho = k (sigma_z rho sigma_z - rho) dt + sqrt(eta k) (sigma_z rho + rho sigma_z - 2 <sigma_z> rho) dW,
    record  dy = 2 sqrt(eta k) <sigma_z> dt + dW,

and the record is fed back as a force on a classical momentum p:

    dp = F / (2 sqrt(eta k)) dy = F <sigma_z> dt + F / (2 sqrt(eta k)) dW.

Then, exactly,
    coherence decays at Gamma = 2 k                      (rho_01 on average; the dissipator does not know eta),
    p drifts by F <sigma_z> and diffuses, Var(dp | drift) = F^2 / (4 eta k) dt = 2 D2 dt,  D2 = F^2 / (8 eta k),
    so  Gamma * D2 = F^2 / (4 eta)  >=  F^2 / 4,  with equality only for a perfect record (eta = 1).

Weak measurement (small k) is slow decoherence and large diffusion, strong measurement the reverse; the product is
pinned.  Throwing information away (eta < 1) raises the product, never lowers it.

What is checked here and what is not (tests/test_cq.py).  The closed forms above, and that a simulation of the SME plus
feedback reproduces the coherence decay exp(-2 k t), the drift F t from a sigma_z eigenstate, and the noise variance
F^2 T / (4 eta k).  The last is close to algebra: the noise in p is the noise in the record, scaled.  What is not
checked: that this protocol is optimal among all CQ dynamics, or that the constant F^2/4 equals the constant in
Oppenheim et al.  Their bound is written with coefficients D0, D1, D2 of a Kramers-Moyal expansion, and the factors of 2
depend on conventions that I did not reconcile (for a sigma_z coupling the paper's decay rate is D0 (L(a) - L(b))^2 / 2,
which is 2 D0 here; I read the paper's text through a fetch, not line by line).  So the claim is the identity for this
protocol and the direction of the inequality in eta, not saturation of a published constant.

Not claimed: a model of gravity.  There is no spacetime here, no metric, no Newton constant and no mapping to the
Jaynes-Cummings sensor or the orbit of `qgl_*`.  It shows what "backreaction" would have to cost in a toy.

Run:  python3 qgl_cq.py            table
      python3 qgl_cq.py --figure   writes figures/qgl_cq.png
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------- closed forms

def coherence_rate(k: float) -> float:
    """Gamma = 2 k: decay rate of the off-diagonal element averaged over the record."""
    return 2.0 * k


def diffusion_D2(F: float, k: float, eta: float = 1.0) -> float:
    """D2 = F^2 / (8 eta k), defined by Var(p_noise) = 2 D2 t."""
    return F * F / (8.0 * eta * k)


def tradeoff_product(F: float, eta: float = 1.0) -> float:
    """Gamma * D2 = F^2 / (4 eta): independent of k."""
    return F * F / (4.0 * eta)


def min_diffusion(F: float, gamma: float, eta: float = 1.0) -> float:
    """D2 needed at a given decoherence rate Gamma: F^2 / (4 eta Gamma).  At eta = 1 this is the boundary the protocol
    cannot cross; for eta < 1 it lies above it."""
    return F * F / (4.0 * eta * gamma)


# ---------------------------------------------------------------- simulation

def _run(k, F, T, dt, n_traj, seed, z0, eta, n_out):
    rng = np.random.default_rng(seed)
    steps = int(round(T / dt))
    z = np.full(n_traj, float(z0))
    c = np.full(n_traj, 0.5 * math.sqrt(max(0.0, 1.0 - z0 * z0)))   # rho_01, real, stays real
    p = np.zeros(n_traj)
    s = math.sqrt(eta * k)
    every = max(1, steps // n_out) if n_out else 0
    t_out, c_out = [0.0], [float(c.mean())]
    for i in range(1, steps + 1):
        dW = rng.normal(0.0, math.sqrt(dt), n_traj)
        dy = 2.0 * s * z * dt + dW
        p += (F / (2.0 * s)) * dy
        c = c - 2.0 * k * c * dt - 2.0 * s * z * c * dW
        z = np.clip(z + 2.0 * s * (1.0 - z * z) * dW, -1.0, 1.0)
        if every and i % every == 0:
            t_out.append(i * dt)
            c_out.append(float(c.mean()))
    return c, p, np.array(t_out), np.array(c_out)


def simulate(k, F=1.0, T=2.0, dt=2e-4, n_traj=4000, seed=0, z0=0.0, eta=1.0):
    """Euler-Maruyama for the SME plus feedback.  Start in |+> (z0 = 0) to see the coherence decay, or in a sigma_z
    eigenstate (z0 = +-1, which the measurement leaves fixed) to isolate the diffusion from the drift.
    Returns dict: coherence (mean rho_01 at T), var_over_T (Var(p_T)/T), mean_p."""
    c, p, _, _ = _run(k, F, T, dt, n_traj, seed, z0, eta, 0)
    return {"coherence": float(c.mean()), "var_over_T": float(p.var() / T), "mean_p": float(p.mean())}


def coherence_curve(k, T=2.0, dt=1e-3, n_traj=4000, seed=0, eta=1.0, n_out=40):
    """Mean rho_01(t) from |+> and the theory 0.5 exp(-2 k t)."""
    _, _, t, c = _run(k, 1.0, T, dt, n_traj, seed, 0.0, eta, n_out)
    return t, c, 0.5 * np.exp(-coherence_rate(k) * t)


# ---------------------------------------------------------------- figure

def plot_tradeoff(path: str) -> None:
    """(a) the trade-off plane (Gamma, D2): the eta = 1 boundary, eta < 1 curves above it, simulated points.
    (b) coherence decay from |+> for three measurement strengths against 0.5 exp(-2 k t)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = 1.0
    ks = (0.05, 0.2, 0.8)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    g = np.geomspace(0.05, 3.0, 200)
    ax[0].fill_between(g, 1e-3, min_diffusion(F, g), color="0.88", label="not reachable by this protocol")
    for eta, col in ((1.0, "tab:blue"), (0.5, "tab:orange"), (0.25, "tab:red")):
        ax[0].loglog(g, min_diffusion(F, g, eta), color=col, label=rf"$\Gamma D_2 = F^2/(4\eta)$, $\eta={eta}$")
    for eta, col, mk in ((1.0, "tab:blue", "o"), (0.5, "tab:orange", "s")):
        pts = []
        for i, k in enumerate(ks):
            r = simulate(k, F=F, T=1.0, dt=2e-4, n_traj=6000, seed=11 + i, z0=1.0, eta=eta)
            pts.append((coherence_rate(k), r["var_over_T"] / 2.0))
        pts = np.array(pts)
        ax[0].loglog(pts[:, 0], pts[:, 1], mk, color=col, mfc="none", ms=9, label=rf"simulated, $\eta={eta}$")
    ax[0].set_xlabel(r"decoherence rate $\Gamma = 2k$")
    ax[0].set_ylabel(r"diffusion $D_2$ of the classical momentum")
    ax[0].set_ylim(1e-2, 1e1)
    ax[0].set_title(r"weak decoherence costs large diffusion ($F=1$)")
    ax[0].legend(fontsize=8, loc="lower left")
    for k, col in zip(ks, ("tab:green", "tab:purple", "tab:brown")):
        t, c, th = coherence_curve(k, T=2.0, dt=1e-3, n_traj=6000, seed=int(100 * k) + 3)
        ax[1].plot(t, th, "-", color=col, label=rf"$0.5\,e^{{-2kt}}$, $k={k}$")
        ax[1].plot(t, c, ".", color=col)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"mean $\rho_{01}$ from $|+\rangle$")
    ax[1].set_title("coherence decay (dots: simulation)")
    ax[1].legend(fontsize=8)
    for q in ax:
        q.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    F, T = 1.0, 2.0
    print("perfect record (eta = 1), F = 1")
    print(" k      Gamma=2k   D2 (sim)   F^2/(8k)   Gamma*D2   coherence(T) sim / theory")
    for k in (0.05, 0.2, 0.8):
        coh = simulate(k, F=F, T=T, z0=0.0, seed=1)["coherence"]
        var = simulate(k, F=F, T=T, z0=1.0, seed=int(100 * k) + 1)["var_over_T"]
        D2 = var / 2.0
        print(f"{k:<6} {coherence_rate(k):<10.3f} {D2:<10.4f} {diffusion_D2(F, k):<10.4f} {coherence_rate(k) * D2:<10.4f} "
              f"{coh:.4f} / {0.5 * math.exp(-2 * k * T):.4f}")
    print("Gamma * D2 = F^2/4 = 0.25 at every k.  With eta < 1 it is F^2/(4 eta): 0.5 at eta = 0.5.")


if __name__ == "__main__":
    import os
    import sys
    if "--figure" in sys.argv:
        os.makedirs("figures", exist_ok=True)
        plot_tradeoff(os.path.join("figures", "qgl_cq.png"))
    else:
        main()
