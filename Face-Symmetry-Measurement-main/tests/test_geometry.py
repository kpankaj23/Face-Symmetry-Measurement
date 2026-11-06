"""Unit tests for the symmetry mathematics (no detector needed)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fsm import geometry as G  # noqa: E402
from fsm import mesh as MS  # noqa: E402
from fsm.quality import gate_decision, risk_score, DEFAULT_GATE  # noqa: E402

PERM = MS.ibug68_permutation()


def symmetric_face(seed=0):
    """A random but perfectly symmetric 68-point configuration about x = 0."""
    rng = np.random.default_rng(seed)
    P = np.zeros((68, 2))
    for a, b in MS.IBUG68_PAIRS:
        x, y = rng.uniform(10, 60), rng.uniform(-80, 80)
        P[a], P[b] = (-x, y), (x, y)
    for k in MS.IBUG68_CENTRAL:
        P[k] = (0.0, rng.uniform(-80, 80))
    return P


def similarity(P, angle_deg, scale, t):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return scale * P @ R.T + np.asarray(t)


def score(P):
    return G.asymmetry(P, PERM, MS.IBUG68_REGIONS, MS.IBUG68_INNER,
                       MS.IBUG68_OUTER_EYE).global_score


def test_permutation_is_involution():
    assert np.all(PERM[PERM] == np.arange(68))
    mp = MS.mp_permutation()
    assert np.all(mp[mp] == np.arange(MS.MP_N))
    # 220 bilateral pairs + 28 midline vertices in the canonical model
    assert np.sum(mp[:468] == np.arange(468)) == 28


def test_reflection_is_its_own_inverse():
    P = np.random.default_rng(1).normal(size=(50, 2))
    Q = G.reflect_points(G.reflect_points(P, 0.7, 3.0), 0.7, 3.0)
    assert np.allclose(P, Q)
    assert np.allclose(G.reflect_vertical(P, 2.0)[:, 0], 4.0 - P[:, 0])


def test_symmetric_face_scores_zero():
    assert score(symmetric_face()) < 1e-6


@pytest.mark.parametrize("angle", [-30, -10, 0, 7, 25])
def test_invariance_to_similarity_transform(angle):
    rng = np.random.default_rng(2)
    P = symmetric_face(3) + rng.normal(scale=1.5, size=(68, 2))   # mildly asymmetric
    s0 = score(P)
    s1 = score(similarity(P, angle, 2.3, (40, -15)))
    assert s0 > 1e-3
    assert abs(s0 - s1) < 1e-4 * max(1, s0) + 1e-6


def test_fitted_axis_recovers_tilt():
    P = similarity(symmetric_face(4), 12.0, 1.0, (5, 5))
    r = G.asymmetry(P, PERM, MS.IBUG68_REGIONS, MS.IBUG68_INNER, MS.IBUG68_OUTER_EYE)
    assert abs(abs(G.axis_tilt_deg(r.theta)) - 12.0) < 0.05


def test_known_displacement_gives_known_score():
    """Moving one eye corner by d pixels horizontally changes its pair residual by ~d."""
    P = symmetric_face(5)
    iod = G.iod(P, MS.IBUG68_OUTER_EYE)
    Q = P.copy()
    Q[39, 0] += 2.0                                # inner corner of one eye, outward shift
    r = G.asymmetry(Q, PERM, MS.IBUG68_REGIONS, MS.IBUG68_INNER, MS.IBUG68_OUTER_EYE)
    # the two partners together deviate by ~2 px from mirror symmetry -> a ~= 1 px / IOD each
    assert r.per_point[39] == pytest.approx(r.per_point[42], rel=1e-6)
    assert r.per_point[39] * iod == pytest.approx(1.0, abs=0.15)
    assert r.region_scores["Eyes"] > r.region_scores["Mouth"]


def test_baseline_kdtree_zero_on_symmetric_mesh():
    """With a vertical, perfectly symmetric mesh the KD-tree baseline gives ~0."""
    V = MS.canonical_vertices()[:, :2] * np.array([20.0, -20.0]) + 300
    L = np.vstack([V, np.tile(V[:1], (10, 1))])        # pad iris rows
    assert G.baseline_kdtree_asymmetry(L)["global_score"] < 1e-6


def test_baseline_not_rotation_invariant_but_proposed_is():
    """The motivating property: vertical-midline methods change under roll."""
    V = MS.canonical_vertices()[:, :2] * np.array([20.0, -20.0])
    rng = np.random.default_rng(6)
    L = np.vstack([V, np.tile(V[:1], (10, 1))]) + rng.normal(scale=0.5, size=(478, 2))
    b0 = G.baseline_kdtree_asymmetry(L)["global_score"]
    b1 = G.baseline_kdtree_asymmetry(similarity(L, 10, 1, (0, 0)))["global_score"]
    assert b1 > 1.5 * b0


def test_unflip_roundtrip():
    """Flipping coordinates and re-indexing twice returns the original landmarks."""
    rng = np.random.default_rng(7)
    L = rng.uniform(0, 500, size=(478, 2))
    perm = MS.mp_permutation()
    W = 640
    Lf = L.copy()
    Lf[:, 0] = W - Lf[:, 0]
    Lf = Lf[perm]                                     # what a perfect detector would return
    assert np.allclose(G.unflip_landmarks(Lf, W, perm), L)


def test_mapping_is_mirror_consistent():
    m = MS.mp68_mapping()
    assert np.all(MS.mp_permutation()[m] == m[PERM])


def test_gate_levels_and_risk():
    ok = {"iod_px": 120, "sharpness": 2.5, "brightness": 120, "clipped": 0.0,
          "abs_yaw": 2, "abs_pitch": 3, "mirror_incons": 0.01, "n_faces": 1}
    assert gate_decision(ok, DEFAULT_GATE)[0] == "accept"
    assert gate_decision({**ok, "abs_yaw": 15}, DEFAULT_GATE)[0] == "caution"
    st, why = gate_decision({**ok, "abs_yaw": 40}, DEFAULT_GATE)
    assert st == "reject" and why[0]["signal"] == "abs_yaw"
    assert gate_decision({**ok, "brightness": 20}, DEFAULT_GATE)[0] == "reject"
    assert risk_score(ok, DEFAULT_GATE) < 0
    assert risk_score({**ok, "abs_yaw": 20}, DEFAULT_GATE) == pytest.approx(1.0)
