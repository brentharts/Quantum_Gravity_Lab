"""Gravitational phase of a freely falling atom (qgl_atom_phase): k g T^2, mass independence, Eotvos scaling, gradient.

Not a model of the Quantum Galileo experiment (press coverage only); these are the textbook light-pulse checks."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qgl_atom_phase as ap

K = ap.k_eff()
AMU = 1.66053907e-27


def test_k_eff_value():
    assert K == pytest.approx(4 * np.pi / 780.241e-9)


def test_numerical_phase_is_k_g_T2_and_propagation_vanishes():
    r = ap.mz_phase_numeric(T=0.1, n=1000)
    assert r["total"] == pytest.approx(ap.phase_uniform(K, ap.G_EARTH, 0.1), rel=1e-9)
    assert abs(r["propagation"]) < 1e-9 * abs(r["total"])               # uniform g: the action difference is zero
    assert abs(r["separation_end"]) < 1e-12                             # the arms recombine exactly


@pytest.mark.parametrize("T", [0.02, 0.05, 0.2])
def test_phase_scales_as_T_squared(T):
    a = ap.mz_phase_numeric(T=T, n=600)["total"]
    b = ap.mz_phase_numeric(T=2 * T, n=600)["total"]
    assert b / a == pytest.approx(4.0, rel=1e-9)


def test_phase_is_linear_in_g():
    a = ap.mz_phase_numeric(g0=9.0, n=600)["total"]
    b = ap.mz_phase_numeric(g0=18.0, n=600)["total"]
    assert b / a == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize("z0,v0", [(0.0, 0.0), (1.3, 0.0), (0.0, 2.0), (-0.7, -1.1)])
def test_uniform_g_phase_does_not_depend_on_initial_position_or_velocity(z0, v0):
    r = ap.mz_phase_numeric(T=0.05, z0=z0, v0=v0, n=800)
    assert r["total"] == pytest.approx(ap.phase_uniform(K, ap.G_EARTH, 0.05), rel=1e-9)


@pytest.mark.parametrize("amu", [39, 87, 133])
def test_mass_cancels_at_fixed_k(amu):
    m = amu * AMU
    assert ap.phase_cow(m, ap.G_EARTH, K, 0.1) == pytest.approx(ap.phase_uniform(K, ap.G_EARTH, 0.1), rel=1e-12)
    r = ap.mz_phase_numeric(m=m, T=0.1, n=800)
    assert r["total"] == pytest.approx(ap.phase_uniform(K, ap.G_EARTH, 0.1), rel=1e-9)


def test_arm_separation_does_depend_on_mass():
    d = [ap.arm_separation(a * AMU, K, 0.1) for a in (39, 87, 133)]
    assert d[0] > d[1] > d[2]
    assert d[0] / d[1] == pytest.approx(87 / 39, rel=1e-12)


def test_eotvos_ratio_recovers_the_injected_violation():
    eta = 1.0 + 3e-7
    T = 0.1
    phi1 = ap.mz_phase_numeric(m=39 * AMU, T=T, eta=eta, n=800)["total"]
    phi2 = ap.mz_phase_numeric(m=87 * AMU, T=T, eta=1.0, n=800)["total"]
    assert ap.eotvos_ratio_from_phases(phi1 - phi2, K, ap.G_EARTH, T) == pytest.approx(3e-7, rel=1e-4)


def test_equal_species_give_zero_differential_phase():
    a = ap.mz_phase_numeric(m=39 * AMU, T=0.1, n=800)["total"]
    b = ap.mz_phase_numeric(m=87 * AMU, T=0.1, n=800)["total"]
    assert ap.eotvos_ratio_from_phases(a - b, K, ap.G_EARTH, 0.1) == pytest.approx(0.0, abs=1e-12)


def test_gradient_term_matches_the_released_from_rest_formula():
    """phi(gamma) - phi(0) = -(7/12) k gamma T^3 (g T - v_r), up to a v_r^2 correction of ~0.2% (checked numerically)."""
    T = 0.1
    base = ap.mz_phase_numeric(T=T, gamma=0.0, n=2000)["total"]
    vr = ap.recoil_velocity(ap.M_RB87, K)
    for gam in (2e-6, -3e-6):
        d = ap.mz_phase_numeric(T=T, gamma=gam, n=2000)["total"] - base
        pred = -(7.0 / 12.0) * K * gam * T ** 3 * (ap.G_EARTH * T - vr)
        assert d == pytest.approx(pred, rel=5e-3)


def test_gradient_phase_is_odd_in_gamma_and_propagation_cancels_separation():
    T = 0.1
    base = ap.mz_phase_numeric(T=T, n=2000)["total"]
    p = ap.mz_phase_numeric(T=T, gamma=2e-6, n=2000)
    m = ap.mz_phase_numeric(T=T, gamma=-2e-6, n=2000)
    assert (p["total"] - base) == pytest.approx(-(m["total"] - base), rel=1e-3)
    assert abs(p["propagation"]) > 1e-3                                  # each is ~1e-2 rad ...
    assert p["propagation"] + p["separation"] == pytest.approx(0.0, abs=1e-5)   # ... and they cancel
