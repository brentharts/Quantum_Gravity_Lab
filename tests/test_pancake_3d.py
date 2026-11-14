"""The 3D pancake picture (qgl_hopf.plot_pancake_3d): the geometry helpers behind it and the values it annotates."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_hopf as hp
import qgl_pancake as qp

P = dict(a0=1.0, b0=3.0, f0=2.0, m=0.5, Lam=0.1)


def test_fibonacci_sphere_points_are_unit_and_spread_out():
    p = hp.fibonacci_sphere(30)
    assert p.shape == (30, 3)
    assert np.linalg.norm(p, axis=1) == pytest.approx(1.0, abs=1e-12)
    d = np.linalg.norm(p[:, None] - p[None], axis=2) + 10 * np.eye(30)
    assert d.min() > 0.3                                         # no two points nearly coincide
    assert np.abs(p.mean(axis=0)).max() < 0.05                   # centred on the origin


@pytest.mark.parametrize("seed", range(4))
def test_tangent_frame_is_orthonormal_and_tangent(seed):
    n = np.random.default_rng(seed).normal(size=3)
    n /= np.linalg.norm(n)
    e1, e2 = hp.tangent_frame(n)
    assert np.linalg.norm(e1) == pytest.approx(1.0) and np.linalg.norm(e2) == pytest.approx(1.0)
    assert abs(e1 @ e2) < 1e-12 and abs(e1 @ n) < 1e-12 and abs(e2 @ n) < 1e-12


def test_tangent_frame_handles_the_x_axis():
    e1, e2 = hp.tangent_frame(np.array([1.0, 0.0, 0.0]))
    assert abs(e1[0]) < 1e-12 and abs(e2[0]) < 1e-12


def test_fibre_circle_has_the_right_centre_radius_and_length():
    n = np.array([0.0, 0.6, 0.8])
    c = hp.fibre_circle(n, base_radius=1.5, fibre_radius=0.4, m=400)
    centre = 1.5 * n
    assert np.linalg.norm(c - centre, axis=1) == pytest.approx(0.4, abs=1e-12)
    assert np.abs((c - centre) @ n).max() < 1e-12                 # lies in the tangent plane
    assert np.linalg.norm(np.diff(c, axis=0), axis=1).sum() == pytest.approx(2 * np.pi * 0.4, rel=1e-3)
    assert np.allclose(c[0], c[-1])                               # closed


def test_fibre_circle_shrinks_to_a_point():
    c = hp.fibre_circle(np.array([0.0, 0.0, 1.0]), 1.5, 0.0)
    assert np.allclose(c, [0.0, 0.0, 1.5])


def test_slice_at_t_zero_is_the_pancake():
    st = hp.pancake_slice_radii(0.0, **P)
    assert st == dict(R1=0.0, R2=3.0, s=0.0)


def test_slice_radii_follow_the_series_just_after_the_pancake():
    st = hp.pancake_slice_radii(0.01, **P)
    assert st["R1"] == pytest.approx(P["a0"] * 0.01, rel=1e-3)   # R_1 = a0 t + O(t^3)
    assert st["R2"] == pytest.approx(P["b0"], rel=1e-3)
    assert st["s"] == pytest.approx(P["a0"] * 0.01 / P["b0"], rel=1e-3)


def test_slice_curvature_is_finite_and_tends_to_eight_over_b0_squared():
    for t in (0.0, 0.15, 0.4, 1.0):
        st = hp.pancake_slice_radii(t, **P)
        scal = (8 - 2 * st["s"] ** 2) / st["R2"] ** 2
        assert np.isfinite(scal)
        if t == 0.0:
            assert scal == pytest.approx(8 / P["b0"] ** 2)
        direct = float(np.sum(qp.slice_ricci(np.array([max(st["R1"], 1e-12), st["R2"], st["R2"]]))))
        assert scal == pytest.approx(direct, rel=1e-6)           # Berger closed form = slice Ricci of the solution


def test_figure_is_written(tmp_path):
    out = tmp_path / "pancake3d.png"
    hp.plot_pancake_3d(str(out), **P, times=(0.0, 0.15, 0.4, 1.0), n_fibres=8)
    assert out.exists() and out.stat().st_size > 20_000
