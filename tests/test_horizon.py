"""Tests for qgl_horizon: horizons at Kasner and pancake singularities."""
import numpy as np
import pytest

import qgl_horizon as hz
import qgl_pancake as qp
import qgl_kasner as kn

A0, B0, F0, M = 1.0, 3.0, 2.0, 0.5


@pytest.fixture(scope="module")
def sol():
    return qp.pancake_solution(A0, B0, F0, M, t_max=6.0)


@pytest.fixture(scope="module")
def horizons(sol):
    return hz.pancake_horizons(sol)


def test_kasner_horizon_closed_form_matches_quadrature():
    from scipy.integrate import quad
    for p in (-0.3, 0.0, 0.5, 0.9):
        num = quad(lambda s: s ** (-p), 0, 2.0)[0]
        assert hz.kasner_comoving_horizon(p, 2.0) == pytest.approx(num, rel=1e-8)
        assert hz.kasner_proper_horizon(p, 2.0) == pytest.approx(2.0 ** (1 - p) * 2.0 ** p / (1 - p), rel=1e-12)


def test_only_the_pancake_exponent_set_has_a_divergent_horizon():
    for r in hz.kasner_horizon_table(us=(1.01, 1.5, 2.0, 4.0, 10.0)):
        if r["u"] is None:
            assert r["types"] == ("logarithmic", "finite", "finite")
        else:
            assert all(t == "finite" for t in r["types"])
    assert np.isinf(hz.kasner_comoving_horizon(1.0, 1.0))


def test_pancake_fibre_horizon_grows_like_log(sol):
    """h_1 = (2/a0) ln(t/t_in) at small t; the slope in ln t is 2/a0."""
    s = qp.pancake_solution(A0, B0, F0, M, t_max=6.0)
    h = hz.pancake_horizons(s)
    t = h["t"]
    sel = (t > 2e-4) & (t < 2e-2)
    slope = np.polyfit(np.log(t[sel]), h["fibre"][sel], 1)[0]
    assert slope == pytest.approx(hz.fibre_log_coefficient(A0), rel=2e-3)
    # and moving t_in down adds more: no limit
    assert h["fibre"][-1] > 2.0 / A0 * np.log(t[-1] / t[0]) * 0.95 - 1.0


def test_fibre_slope_scales_with_inverse_a0():
    for a0 in (0.5, 2.0):
        s = qp.pancake_solution(a0, B0, F0, M, t_max=0.5)
        h = hz.pancake_horizons(s)
        sel = (h["t"] > 2e-4) & (h["t"] < 1e-2)
        slope = np.polyfit(np.log(h["t"][sel]), h["fibre"][sel], 1)[0]
        assert slope == pytest.approx(2.0 / a0, rel=5e-3)


def test_open_directions_have_a_finite_horizon_of_order_t(horizons):
    """Base angle ~ 2t/b0 at small t, so the proper reach is ~ t; no divergence as t_in -> 0."""
    t, base = horizons["t"], horizons["base"]
    sel = (t > 1e-3) & (t < 1e-1)
    assert np.allclose(base[sel] / t[sel], 2.0 / B0, rtol=1e-2)
    assert base[0] == pytest.approx(2.0 * t[0] / B0, rel=1e-6)


def test_fibre_circle_shrinks_while_base_stays_finite(sol, horizons):
    assert horizons["fibre_circumference"][0] < 1e-2 * horizons["fibre_circumference"][len(horizons["t"]) // 2]
    assert sol["R"][sol["t"] > 0][0, 1] == pytest.approx(B0, rel=1e-3)


def test_base_contact_time_is_after_pi_b0_over_two_scale(horizons):
    tc = hz.base_contact_time(horizons)
    assert np.isfinite(tc)
    assert 0.5 * np.pi * B0 / 2 < tc < 2.0 * np.pi * B0 / 2   # order pi b0/2; R_2 grows, so a little later
    # contact angle monotone
    assert np.all(np.diff(horizons["base"]) > 0)
    assert np.isnan(hz.base_contact_time(horizons, angle=100.0))


def test_whole_fibre_is_in_contact_from_arbitrarily_early(horizons):
    """The fibre horizon exceeds one period 4 pi after t_in e^(2 pi a0), i.e. turns >= 1 almost at once."""
    t_wrap = hz.fibre_time_to_wrap(A0, 1.0, turns=1.0)
    assert t_wrap == pytest.approx(np.exp(-2 * np.pi), rel=1e-12)
    i = int(np.searchsorted(horizons["t"], 1.0))
    # counted from 1e-4, the ray has made many turns by t=1; a smaller t_in gives more
    assert horizons["fibre_turns"][i] > 0.5
    assert horizons["fibre_turns"][-1] > horizons["fibre_turns"][i]


def test_summary_runs_and_is_finite():
    s = hz.summary()
    assert all(np.isfinite(v) for v in s.values())
