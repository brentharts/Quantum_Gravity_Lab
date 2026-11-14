#!/usr/bin/env python3
"""
qgl_qsn.py -- tetragrids, the 20-tetrahedra cluster (20G) and the Fibonacci icosagrid ("quasicrystalline spin network").

Source for the construction: F. Fang and K. Irwin, "An Icosahedral Quasicrystal and E8 derived quasicrystals",
arXiv:1511.07786, as summarised from the text I could retrieve (formulas were partly garbled in extraction).

Taken from that source:
  * Tetragrid: four plane families with unit normals along the four 3-fold axes of a tetrahedron; planes x.e_k = x_N.
    At uniform spacing (x_N = T N) the tetragrid is an FCC lattice (tetrahedra and octahedra).
  * Fibonacci spacing: x_N = T (N + alpha + (1/rho) floor(N/mu + beta)), with rho = mu = tau for the Fibonacci case.
  * Golden rotation between tetragrids: arccos((3 tau - 1)/4), about 15.52 degrees.
  * 4G: four tetrahedra of one orientation meeting at a point (cuboctahedron triangles joined to its centre).
    20G: five 4Gs related by 5-fold rotations (compound of five cuboctahedra), 61 vertices.
  * Icosagrid: ten plane families along the ten 3-fold axes of an icosahedron, described as a golden composition of
    five tetragrids.

My choices (inferred, not from the source):
  * alpha = 0 and beta = 1/2.  These make the chain symmetric under x -> -x (checked in the tests); the source asks for
    180-degree symmetry at the origin but I could not read the values it uses.
  * The five tetragrids are the five tetrahedra of one chirality in the compound of five tetrahedra inscribed in the
    dodecahedron (their 20 vertices are the 20 dodecahedron vertices).
  * Which points count as vertices.  grid_vertices(min_planes=m) keeps the points lying on at least m planes of a
    tetragrid; m = 4 (all four families meet) and m = 3 (any triple point) are both provided and reported separately.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

TAU = (1.0 + 5.0 ** 0.5) / 2.0
GOLDEN_ROTATION_DEG = float(np.degrees(np.arccos((3.0 * TAU - 1.0) / 4.0)))


# ---------------------------------------------------------------- icosahedral frame

def dodecahedron_vertices() -> np.ndarray:
    """The 20 vertices of the dodecahedron (norm sqrt 3), shape (20, 3).  They are the 20 three-fold directions."""
    v = [p for p in itertools.product((1.0, -1.0), repeat=3)]
    for s1, s2 in itertools.product((1.0, -1.0), repeat=2):
        v.append((0.0, s1 / TAU, s2 * TAU))
        v.append((s1 / TAU, s2 * TAU, 0.0))
        v.append((s2 * TAU, 0.0, s1 / TAU))
    return np.array(v)


def icosahedral_axes() -> np.ndarray:
    """Unit vectors along the ten 3-fold axes, one per line, shape (10, 3)."""
    dv = dodecahedron_vertices() / np.sqrt(3.0)
    axes: List[np.ndarray] = []
    for v in dv:
        if not any(np.allclose(v, -a) or np.allclose(v, a) for a in axes):
            axes.append(v)
    return np.array(axes)


def inscribed_tetrahedra() -> List[Tuple[int, int, int, int]]:
    """All regular tetrahedra whose vertices are dodecahedron vertices, as index 4-tuples (there are ten)."""
    dv = dodecahedron_vertices()
    d = np.linalg.norm(dv[:, None] - dv[None, :], axis=2)
    edge = 2.0 * np.sqrt(2.0)                     # face diagonal of the inscribed cube (side 2)
    out = []
    for q in itertools.combinations(range(20), 4):
        if all(abs(d[a, b] - edge) < 1e-9 for a, b in itertools.combinations(q, 2)):
            out.append(q)
    return out


def five_tetrahedra() -> np.ndarray:
    """Unit vertex directions of the five tetrahedra of one chirality, shape (5, 4, 3).

    The chirality is the one containing the cube-aligned tetrahedron (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)."""
    dv = dodecahedron_vertices()
    tets = inscribed_tetrahedra()
    idx = {tuple(np.round(p, 6)): i for i, p in enumerate(dv)}
    t0 = tuple(sorted(idx[p] for p in [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)]))
    chosen = [t0]
    used = set(t0)
    rest = [t for t in tets if t != t0 and not (set(t) & used)]

    def cover(sel: List[Tuple[int, ...]], taken: set, pool: List[Tuple[int, ...]]):
        if len(sel) == 5:
            return sel
        for t in pool:
            if not (set(t) & taken):
                got = cover(sel + [t], taken | set(t), pool)
                if got:
                    return got
        return None

    sel = cover(chosen, used, rest)
    assert sel is not None and len(sel) == 5
    return np.array([dv[list(t)] / np.sqrt(3.0) for t in sel])


# ---------------------------------------------------------------- spacing

def fibonacci_offsets(n_max: int, T: float = 1.0, alpha: float = 0.0, beta: float = 0.5,
                      rho: float = TAU, mu: float = TAU) -> Dict[int, float]:
    """x_N = T (N + alpha + (1/rho) floor(N/mu + beta)) for N = -n_max..n_max."""
    return {n: T * (n + alpha + np.floor(n / mu + beta) / rho) for n in range(-n_max, n_max + 1)}


def uniform_offsets(n_max: int, T: float = 1.0) -> Dict[int, float]:
    return {n: T * n for n in range(-n_max, n_max + 1)}


# ---------------------------------------------------------------- 4G / 20G

def cuboctahedron(tetra: np.ndarray) -> np.ndarray:
    """The 12 unit vertices (circumradius 1) of the cuboctahedron whose triangle normals include the tetrahedron
    directions: +-(e_i - e_j) normalised, shape (12, 3)."""
    out = []
    for i, j in itertools.combinations(range(4), 2):
        v = tetra[i] - tetra[j]
        v = v / np.linalg.norm(v)
        out.append(v)
        out.append(-v)
    return np.array(out)


def cluster_20g(scale: float = 1.0) -> np.ndarray:
    """Vertices of the 20G: the origin and the 12 cuboctahedron vertices of each of the five 4Gs."""
    pts = [np.zeros(3)]
    for t in five_tetrahedra():
        pts.extend(cuboctahedron(t) * scale)
    return np.array(pts)


def relative_rotations(tetra_a: np.ndarray, tetra_b: np.ndarray) -> List[np.ndarray]:
    """All proper rotations mapping the vertex set of tetra_a onto that of tetra_b (twelve of them)."""
    out = []
    for perm in itertools.permutations(range(4)):
        pb = tetra_b[list(perm)]
        h = tetra_a.T @ pb                         # Kabsch
        u, _, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        if np.linalg.det(r) > 0 and np.allclose(tetra_a @ r.T, pb, atol=1e-9):
            if not any(np.allclose(r, q, atol=1e-9) for q in out):
                out.append(r)
    return out


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def min_relative_angle_deg(tetra_a: np.ndarray, tetra_b: np.ndarray) -> float:
    return min(rotation_angle_deg(r) for r in relative_rotations(tetra_a, tetra_b))


# ---------------------------------------------------------------- grids

def grid_vertices(normals: np.ndarray, offsets: Dict[int, float], radius: float, min_planes: int = 4,
                  tol: float = 1e-9) -> np.ndarray:
    """Points within `radius` of the origin lying on at least `min_planes` planes of the families `normals`
    (one unit normal per family, the same offset table for every family)."""
    x = np.array(sorted(offsets.values()))
    k = len(normals)
    found = {}
    for fam in itertools.combinations(range(k), 3):
        e3 = normals[list(fam)]
        if abs(np.linalg.det(e3)) < 1e-9:
            continue
        inv = np.linalg.inv(e3)
        grid = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
        pts = grid @ inv.T
        pts = pts[np.linalg.norm(pts, axis=1) <= radius + tol]
        if len(pts) == 0:
            continue
        proj = pts @ normals.T                               # (n, k)
        pos = np.searchsorted(x, proj)
        near = np.zeros_like(proj, dtype=bool)
        for shift in (-1, 0):
            cand = x[np.clip(pos + shift, 0, len(x) - 1)]
            near |= np.abs(cand - proj) < 1e-7
        keep = near.sum(axis=1) >= min_planes
        for p in pts[keep]:
            found[tuple(np.round(p, 7))] = p
    return np.array(list(found.values())) if found else np.zeros((0, 3))


def icosagrid_vertices(offsets: Dict[int, float], radius: float, min_planes: int = 4) -> np.ndarray:
    """Union of the vertex sets of the five tetragrids (my reading of 'golden composition of five tetragrids')."""
    allp = {}
    for t in five_tetrahedra():
        for p in grid_vertices(t, offsets, radius, min_planes):
            allp[tuple(np.round(p, 7))] = p
    return np.array(list(allp.values())) if allp else np.zeros((0, 3))


def contains(points: np.ndarray, subset: np.ndarray, tol: float = 1e-6) -> Tuple[int, int]:
    """(number of `subset` points found in `points`, len(subset))."""
    if len(points) == 0:
        return 0, len(subset)
    d = np.linalg.norm(subset[:, None, :] - points[None, :, :], axis=2).min(axis=1)
    return int(np.sum(d < tol)), len(subset)


def core_scale(offsets: Dict[int, float]) -> float:
    """Circumradius of the central cuboctahedron when the first planes N = +-1 carry its vertices: x_1 / (u.e_1)."""
    t = five_tetrahedra()[0]
    u = cuboctahedron(t)[0]                       # (e_0 - e_1) normalised
    return offsets[1] / float(u @ t[0])


# ---------------------------------------------------------------- tetrahedral cells and the edge graph

def tetra_cells(normals: np.ndarray, offsets: Dict[int, float], radius: float, eps: float = 1e-9) -> List[np.ndarray]:
    """Regular-tetrahedron cells of one tetragrid, as (4, 3) vertex arrays, with every vertex within `radius`.

    Because the four unit normals sum to zero, an 'up' tetrahedron {p.e_k >= a_k for all k} (one plane per family) is
    bounded when h = -sum(a_k) > 0, has p.e_j running from a_j to a_j + h, and is a cell of the arrangement exactly
    when no plane of family k lies strictly inside (a_k, a_k + h).  A 'down' tetrahedron {p.e_k <= b_k} is the mirror
    case with h = sum(b_k) and the range (b_k - h, b_k).  A vertex with p.e_k = y_k for all k is p = (3/4) sum y_k e_k."""
    x = np.array(sorted(offsets.values()))
    m = len(x)
    after = np.append(np.diff(x), np.inf)
    before = np.insert(np.diff(x), 0, np.inf)
    idx = np.array(np.meshgrid(*([np.arange(m)] * 4), indexing="ij")).reshape(4, -1).T
    xs = x[idx]                                                # (n, 4) offsets
    total = xs.sum(axis=1)
    cells: List[np.ndarray] = []
    for sign, gap in ((+1, after), (-1, before)):
        h = -total if sign > 0 else total
        ok = (h > eps) & np.all(gap[idx] >= h[:, None] - eps, axis=1)
        for a, hh in zip(xs[ok], h[ok]):
            verts = []
            for j in range(4):
                y = a.copy()
                y[j] = -(a.sum() - a[j])
                verts.append(0.75 * (y @ normals))
            verts = np.array(verts)
            if np.all(np.linalg.norm(verts, axis=1) <= radius + eps):
                cells.append(verts)
    return cells


def icosagrid_graph(offsets: Dict[int, float], radius: float):
    """Vertices, edges and cells of the union of the tetrahedral cells of the five tetragrids.

    Returns (vertices (n, 3), edges list of (i, j), cells list of (4, 3) arrays, cell_tetragrid list of ints)."""
    vid: Dict[Tuple[float, ...], int] = {}
    verts: List[np.ndarray] = []
    edges = set()
    all_cells: List[np.ndarray] = []
    owner: List[int] = []
    for k, t in enumerate(five_tetrahedra()):
        for cell in tetra_cells(t, offsets, radius):
            all_cells.append(cell)
            owner.append(k)
            ids = []
            for p in cell:
                key = tuple(np.round(p, 7) + 0.0)
                if key not in vid:
                    vid[key] = len(verts)
                    verts.append(p)
                ids.append(vid[key])
            for a, b in itertools.combinations(ids, 2):
                edges.add((min(a, b), max(a, b)))
    return np.array(verts), sorted(edges), all_cells, owner


def graph_from_cells(cells: List[np.ndarray], edge_length: float = None, tol: float = 1e-6):
    """Vertices and edges of the given cells, optionally only the cells whose edge length is `edge_length`."""
    vid: Dict[Tuple[float, ...], int] = {}
    verts: List[np.ndarray] = []
    edges = set()
    for cell in cells:
        if edge_length is not None and abs(np.linalg.norm(cell[0] - cell[1]) - edge_length) > tol:
            continue
        ids = []
        for p in cell:
            key = tuple(np.round(p, 7) + 0.0)
            if key not in vid:
                vid[key] = len(verts)
                verts.append(p)
            ids.append(vid[key])
        for a, b in itertools.combinations(ids, 2):
            edges.add((min(a, b), max(a, b)))
    return np.array(verts), sorted(edges)


def graph_statistics(verts: np.ndarray, edges, interior: float) -> Dict[str, object]:
    """Degree range over vertices with |p| <= interior, number of distinct edge directions and edge lengths."""
    deg = np.zeros(len(verts), dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    inner = np.linalg.norm(verts, axis=1) <= interior
    dirs, lens = set(), set()
    for a, b in edges:
        d = verts[b] - verts[a]
        n = np.linalg.norm(d)
        u = d / n
        first = next(c for c in u if abs(c) > 1e-9)
        u = u * np.sign(first)
        dirs.add(tuple(np.round(u, 6) + 0.0))
        lens.add(round(float(n), 6))
    return {"n_vertices": len(verts), "n_edges": len(edges), "deg_min": int(deg[inner].min()),
            "deg_max": int(deg[inner].max()), "n_directions": len(dirs), "lengths": sorted(lens),
            "n_interior": int(inner.sum())}


# ---------------------------------------------------------------- Z[tau] coordinates

ZTAU_UNIT = 3.0 ** 0.5 / 2.0       # global irrational factor of the icosahedral frame used here
ZTAU_DEN = 2                       # common denominator


def zphi_coordinates(points: np.ndarray, unit: float = ZTAU_UNIT, den: int = ZTAU_DEN, b_max: int = 60,
                     tol: float = 1e-8):
    """Write every coordinate as x = unit * (a + b tau) / den with integers a, b.

    Returns an integer array of shape (n, 3, 2) holding (a, b), or None if some coordinate has no such form with
    |b| <= b_max (Z[tau] is dense in R, so b_max is kept small and tol tight to make a chance fit unlikely)."""
    pts = np.asarray(points, dtype=float)
    y = pts.reshape(-1) / unit * den
    bs = np.arange(-b_max, b_max + 1)
    a = np.rint(y[:, None] - bs[None, :] * TAU)
    res = np.abs(a + bs[None, :] * TAU - y[:, None])
    best = res.argmin(axis=1)
    ok = res[np.arange(len(y)), best] < tol
    if not np.all(ok):
        return None
    ab = np.stack([a[np.arange(len(y)), best], bs[best]], axis=1).astype(int)
    return ab.reshape(pts.shape[0], 3, 2)


# ---------------------------------------------------------------- chain gaps

def chain_gap_word(offsets: Dict[int, float], tol: float = 1e-9) -> str:
    """'L'/'S' word of the gaps between consecutive planes of a chain (two gap lengths expected)."""
    x = np.array(sorted(offsets.values()))
    gaps = np.diff(x)
    short = gaps.min()
    return "".join("S" if abs(g - short) < tol else "L" for g in gaps)


# ---------------------------------------------------------------- tetrahedron adjacency (spin-network input)

def shared_vertex_counts(cells: List[np.ndarray]) -> Dict[Tuple[int, int], int]:
    """For every pair of cells (i < j) that touch, the number of vertices they share (1 = vertex, 2 = edge, 3 = face)."""
    where: Dict[Tuple[float, ...], List[int]] = {}
    for i, c in enumerate(cells):
        for p in c:
            where.setdefault(tuple(np.round(p, 6) + 0.0), []).append(i)
    counts: Dict[Tuple[int, int], int] = {}
    for ids in where.values():
        for a, b in itertools.combinations(sorted(set(ids)), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def cell_adjacency(cells: List[np.ndarray], shared: int) -> List[Tuple[int, int]]:
    """Links between cells sharing exactly `shared` vertices (3 = a common triangle, i.e. the spin-network rule
    'nodes = tetrahedra, links = shared triangles'; 2 = a common edge)."""
    return sorted(k for k, v in shared_vertex_counts(cells).items() if v == shared)


def degree_sequence(n_nodes: int, links: List[Tuple[int, int]]) -> np.ndarray:
    d = np.zeros(n_nodes, dtype=int)
    for a, b in links:
        d[a] += 1
        d[b] += 1
    return d


# ---------------------------------------------------------------- figure

def plot_chain(path: str, radius: float = 12.0) -> None:
    """Four panels: H4 shells of the E8 roots, the Fibonacci chain against uniform spacing, the 20G cluster inside the
    icosagrid vertices, and the vertex-degree histogram of the largest-cell edge graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    import qgl_e8

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    # (a) E8 roots in the H4 subspace
    roots = qgl_e8.e8_roots()
    p_phys, _ = qgl_e8.h4_projection()
    img = roots @ p_phys.T
    rad = np.round(np.linalg.norm(img, axis=1), 6)
    shells = sorted(set(rad))
    a = ax[0, 0]
    for r_, col in zip(shells, ("tab:blue", "tab:orange")):
        m = rad == r_
        a.scatter(img[m, 0], img[m, 1], s=10, color=col, label=f"{int(m.sum())} roots, radius {r_:.4f}")
    a.set_aspect("equal")
    a.set_title(f"E8 roots in the H4 subspace: radius ratio {shells[-1] / shells[0]:.4f} (tau = {TAU:.4f})", fontsize=9)
    a.legend(fontsize=8, frameon=False)
    # (b) chains
    a = ax[0, 1]
    fib = np.array(sorted(fibonacci_offsets(12).values()))
    uni = np.array(sorted(uniform_offsets(12).values()))
    a.vlines(fib, 0.1, 0.9, color="tab:red", lw=1.4, label="Fibonacci spacing")
    a.vlines(uni, -0.9, -0.1, color="tab:gray", lw=1.4, label="uniform spacing")
    a.set_yticks([])
    a.set_title("plane offsets x_N: gaps T and tau T (Fibonacci word) vs a lattice", fontsize=9)
    a.set_ylim(-1.0, 1.5)
    a.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    # (c) 20G inside the icosagrid vertices
    off = fibonacci_offsets(12)
    v = icosagrid_vertices(off, 6.0)
    c20 = cluster_20g(core_scale(off))
    a = ax[1, 0]
    a.scatter(v[:, 0], v[:, 1], s=3, color="0.6", label=f"icosagrid vertices ({len(v)}, r <= 6)")
    a.scatter(c20[:, 0], c20[:, 1], s=26, color="tab:red", label="20G cluster (61 points)")
    a.set_aspect("equal")
    a.set_title("20G cluster inside the Fibonacci icosagrid (xy projection)", fontsize=9)
    a.legend(fontsize=8, frameon=False, loc="upper right")
    # (d) degree histogram, largest cells
    verts, edges, cells, _ = icosagrid_graph(off, radius)
    big = max(round(float(np.linalg.norm(c[0] - c[1])), 6) for c in cells)
    gv, ge = graph_from_cells(cells, edge_length=big)
    deg = degree_sequence(len(gv), ge)
    inner = np.linalg.norm(gv, axis=1) <= radius - 2 * big
    h = Counter(deg[inner].tolist())
    a = ax[1, 1]
    a.bar(list(h.keys()), list(h.values()), color="tab:green")
    a.set_xlabel("vertex degree")
    a.set_ylabel("vertices")
    a.set_title(f"largest-cell edge graph, interior vertices: degrees {min(h)}-{max(h)}", fontsize=9)
    for q in ax.ravel():
        q.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"golden rotation arccos((3 tau-1)/4) = {GOLDEN_ROTATION_DEG:.4f} deg")
    t = five_tetrahedra()
    print("min relative rotation between tetragrid 0 and 1..4:",
          [round(min_relative_angle_deg(t[0], t[k]), 4) for k in range(1, 5)])
    print("20G vertices:", len(cluster_20g()))
    import os
    import sys
    if "--figure" in sys.argv:
        os.makedirs("figures", exist_ok=True)
        plot_chain(os.path.join("figures", "qgl_qsn.png"))
