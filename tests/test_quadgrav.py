"""Weyl^2 probe for quadratic gravity (qgl_quadgrav).  No field equations of quadratic gravity are solved; these tests
check the curvature invariants the probe reports, each against an independent route."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_bkl as bk
import qgl_flrw as fl
import qgl_kasner as qk
import qgl_pancake as qp
import qgl_quadgrav as qg


# ---------------------------------------------------------------- algebra

def test_weyl_squared_of_de_sitter_is_zero():
    H = 0.7
    inv = qg.flrw_invariants(H, 0.0)
    assert inv["K"] == pytest.approx(24 * H ** 4) and inv["R"] == pytest.approx(12 * H ** 2)
    assert inv["C2"] == pytest.approx(0.0, abs=1e-12)
    assert inv["GB"] == pytest.approx(24 * H ** 4)                       # 24 H^4: Euler density of dS


@pytest.mark.parametrize("seed", range(4))
def test_flrw_is_conformally_flat_for_any_H_and_Hdot(seed):
    rng = np.random.default_rng(seed)
    H, Hd = rng.normal(size=50), rng.normal(size=50)
    inv = qg.flrw_invariants(H, Hd)
    assert inv["C2"] == pytest.approx(0.0, abs=1e-10 * (1 + np.max(np.abs(inv["K"]))))
    assert inv["K"] == pytest.approx(fl.flrw_kretschmann(H, Hd), rel=1e-13)      # agrees with the existing helper


def test_lcdm_history_has_no_weyl_curvature():
    a = np.geomspace(1e-3, 10, 200)
    E2 = lambda x: fl.lcdm_E2(x, Om=0.3, Or=1e-4)
    d = fl.dlnE2_dlna_numeric(E2, a)
    H = np.sqrt(E2(a))
    inv = qg.flrw_invariants(H, 0.5 * H ** 2 * d)
    assert np.max(np.abs(inv["C2"]) / inv["K"]) < 1e-8


def test_weyl_fraction_limits():
    assert qg.weyl_fraction(2.0, 0.0) == pytest.approx(1.0)
    assert qg.weyl_fraction(0.0, 3.0) == pytest.approx(0.0)
    assert qg.weyl_fraction(0.0, 0.0) == 0.0
    assert qg.weyl_fraction(-1.0, 1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------- vacuum Kasner / Mixmaster: Ric = 0, C^2 = K

@pytest.mark.parametrize("u", [1.5, 2.0, 3.7])
def test_vacuum_kasner_is_ricci_flat_and_c2_equals_closed_form_k(u):
    p = np.array(qk.kasner_exponents(u), dtype=float)
    t = np.array([0.3, 1.0, 5.0])
    inv = qg.kasner_invariants(p, t)
    assert inv["ric_sq"] == pytest.approx(0.0, abs=1e-12)
    assert inv["R"] == pytest.approx(0.0, abs=1e-12)
    assert inv["C2"] == pytest.approx(qk.kretschmann_closed_form(p, t), rel=1e-12)
    assert inv["GB"] == pytest.approx(inv["K"], rel=1e-12)             # Ric = 0: Euler density is Riem^2


def test_non_vacuum_kasner_has_ricci_curvature():
    inv = qg.kasner_invariants([0.5, 0.5, 0.5], 1.0)                  # sum p = 3/2, not a vacuum solution
    assert inv["ric_sq"][0] > 1e-3


@pytest.fixture(scope="module")
def mm():
    return bk.Mixmaster(3.7, t0=1.0e6, t_bounce=1.0, t_end=1.0e-30)


@pytest.mark.parametrize("t", [1e4, 30.0, 3.0, 0.1, 1e-5])
def test_mixmaster_ricci_vanishes_and_c2_is_the_weyl_k(mm, t):
    a, ad, add = (v[0] for v in mm.scale_factors(t))
    inv = qg.bianchi_invariants(a, ad, add)
    K, _ = mm.weyl_invariants(t)
    assert inv["C2"] == pytest.approx(float(K[0]), rel=1e-6, abs=1e-9 * abs(float(K[0])))
    assert inv["ric_sq"] < 1e-8 * abs(float(K[0]))                            # vacuum: Ric^2 is roundoff next to K


# ---------------------------------------------------------------- pancake: Einstein equations R_ab = kappa (d_a phi d_b phi + g_ab V) + Lambda g_ab

@pytest.fixture(scope="module")
def pancake():
    return qp.pancake_solution(1.0, 3.0, 2.0, 0.5, 0.1, t_max=5.0, n=200)


def _pancake_invariants(s):
    sg = np.sign(s["R"])
    return qg.bianchi_ricci(np.abs(s["R"]) / 2.0, sg * s["Rd"] / 2.0, sg * s["Rdd"] / 2.0), \
        qg.bianchi_invariants(np.abs(s["R"]) / 2.0, sg * s["Rd"] / 2.0, sg * s["Rdd"] / 2.0)


def test_pancake_ricci_matches_the_einstein_equations(pancake):
    """With V = m^2 phi^2 / 2:  R_ii = kappa V + Lambda,  R_00 = kappa (phi'^2 - V) - Lambda."""
    p = pancake["params"]
    (R00, Rii), _ = _pancake_invariants(pancake)
    V = 0.5 * p["m"] ** 2 * pancake["phi"] ** 2
    for i in range(3):
        assert Rii[:, i] == pytest.approx(p["kappa"] * V + p["Lam"], rel=1e-7, abs=1e-8)
    assert R00 == pytest.approx(p["kappa"] * (pancake["phid"] ** 2 - V) - p["Lam"], rel=1e-7, abs=1e-8)


def test_pancake_has_both_weyl_and_ricci_curvature(pancake):
    _, inv = _pancake_invariants(pancake)
    assert np.all(np.isfinite(inv["C2"])) and np.all(inv["ric_sq"] > 0)
    assert np.max(np.abs(inv["C2"])) > 1e-2
    f = qg.weyl_fraction(inv["C2"], inv["ric_sq"])
    assert 0.0 <= f.min() and f.max() < 1.0                                 # never pure vacuum with the scalar present


def test_pancake_c2_at_the_pancake_matches_the_closed_forms(pancake):
    """At t = 0 the existing closed form gives K; Ricci is R_ii = kappa V + Lambda, R_00 = kappa (phi'^2 - V) - Lambda."""
    p = pancake["params"]
    s = qp.pancake_solution(p["a0"], p["b0"], p["f0"], p["m"], p["Lam"], t_max=1.0, t_start=1e-5, n=40)
    sg = np.sign(s["R"])
    inv = qg.bianchi_invariants(np.abs(s["R"]) / 2.0, sg * s["Rd"] / 2.0, sg * s["Rdd"] / 2.0)
    i = int(np.argmin(np.abs(s["t"])))
    assert abs(s["t"][i]) < 1e-4
    assert inv["K"][i] == pytest.approx(qp.kretschmann_at_pancake(p["b0"], p["f0"], p["m"], p["Lam"], p["kappa"], p["a0"]),
                                        rel=1e-4)
    Rii = p["kappa"] * 0.5 * p["m"] ** 2 * p["f0"] ** 2 + p["Lam"]
    # phi'(0) = 0 (odd parity series: f1 = 0), so R_00 = -kappa V - Lambda and Ric^2 = R_00^2 + 3 R_ii^2
    R00 = -Rii
    assert inv["ric_sq"][i] == pytest.approx(R00 ** 2 + 3 * Rii ** 2, rel=1e-4)
