"""Extra qgl_qsn checks: Z[tau] coordinates, the Fibonacci gap word, and cell adjacency (spin-network input).

Findings pinned here (all computed, not from a source):
  * every icosagrid vertex coordinate is (sqrt(3)/2)(a + b tau)/2 with integers a, b (min_planes 4 and 3);
  * tetrahedral cells of the five tetragrids never share a triangle, at any size or across tetragrids, so the rule
    'nodes = tetrahedra, links = shared triangles' gives an empty graph; cells touch along edges and at vertices only.
"""
import itertools

import numpy as np
import pytest

import qgl_qsn as q

OFF = q.fibonacci_offsets(12)


def test_zphi_recovers_known_values():
    pts = np.array([[0.0, q.ZTAU_UNIT * q.TAU / 2, q.ZTAU_UNIT * (3 + 2 * q.TAU) / 2]])
    ab = q.zphi_coordinates(pts)
    assert ab is not None
    assert ab[0].tolist() == [[0, 0], [0, 1], [3, 2]]


def test_zphi_rejects_a_generic_point():
    assert q.zphi_coordinates(np.array([[0.1234567, 0.0, 0.0]])) is None


def test_20g_cluster_coordinates_are_in_z_tau():
    ab = q.zphi_coordinates(q.cluster_20g(q.core_scale(OFF)))
    assert ab is not None and ab.shape == (61, 3, 2)


@pytest.mark.parametrize("min_planes", [4, 3])
def test_icosagrid_vertices_are_in_z_tau(min_planes):
    v = q.icosagrid_vertices(OFF, 6.0, min_planes=min_planes)
    ab = q.zphi_coordinates(v)
    assert ab is not None and len(v) > 500
    back = q.ZTAU_UNIT * (ab[..., 0] + ab[..., 1] * q.TAU) / q.ZTAU_DEN
    assert np.allclose(back, v, atol=1e-8)


def test_gap_word_has_golden_frequency_and_no_ss():
    w = q.chain_gap_word(q.fibonacci_offsets(400))
    assert set(w) == {"L", "S"}
    assert w.count("L") / w.count("S") == pytest.approx(q.TAU, abs=0.02)
    assert "SS" not in w                                  # Fibonacci word: the short gap is always isolated


def test_uniform_chain_has_a_single_gap_length():
    assert set(q.chain_gap_word(q.uniform_offsets(10))) == {"S"}      # every gap is the minimum


@pytest.fixture(scope="module")
def cells():
    _, _, c, owner = q.icosagrid_graph(OFF, 9.0)
    return c, owner


def test_no_two_cells_share_a_triangle(cells):
    c, _ = cells
    counts = q.shared_vertex_counts(c)
    assert 3 not in counts.values() and 4 not in counts.values()
    assert q.cell_adjacency(c, 3) == []


def test_cells_do_touch_along_edges(cells):
    c, _ = cells
    links = q.cell_adjacency(c, 2)
    assert len(links) > 0
    deg = q.degree_sequence(len(c), links)
    assert deg.max() >= 1
    for a, b in links[:50]:                                # a shared edge has the length of both cells' edges
        common = [p for p in c[a] if any(np.allclose(p, r, atol=1e-6) for r in c[b])]
        assert len(common) == 2


def test_uniform_tetragrid_tetrahedra_share_no_triangle():
    t = q.five_tetrahedra()[0]
    c = q.tetra_cells(t, q.uniform_offsets(6), 5.0)
    assert len(c) > 10
    assert q.cell_adjacency(c, 3) == []


def test_degree_sequence_counts_both_ends():
    assert q.degree_sequence(4, [(0, 1), (1, 2)]).tolist() == [1, 2, 1, 0]
