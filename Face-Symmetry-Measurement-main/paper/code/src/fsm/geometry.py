"""
geometry.py -- the symmetry mathematics. No image processing and no UI in this file,
so everything here is unit-tested directly (tests/test_geometry.py).

Notation
--------
P      : (n, 2) landmark coordinates in pixels (x right, y down).
perm   : mirror permutation, perm[i] is the bilateral partner of point i
         (perm[i] == i for midline points).
axis   : the symmetry line  {x : n . x = rho}, with unit normal n = (cos t, sin t).
refl(x) = x - 2 (n . x - rho) n   is the reflection about that line.

Proposed measure (used by variants M1-M3)
-----------------------------------------
Instead of assuming a vertical midline through two chosen landmarks, we estimate the
symmetry axis that best explains the configuration:

    (t*, rho*) = argmin_{t, rho}  sum_i || refl(P[perm[i]]) - P[i] ||^2

The residual r_i = refl(P[perm[i]]) - P[i] is the displacement between a point and the
reflection of its partner. Its half-length |r_i| / 2 is the distance of point i from the
closest perfectly symmetric configuration (P + refl(P[perm])) / 2. We report

    a_i = |r_i| / (2 * IOD)          (per-point asymmetry, fraction of inter-ocular dist.)
    A   = mean_i a_i                 (global; over a chosen subset, e.g. the inner face)
    A_r = mean_{i in region r} a_i   (regional)

Because the axis is fitted, A is invariant to in-plane rotation, translation and scale
of the landmarks (verified in tests). This is the 2D analogue of the Procrustes
"object symmetry" decomposition of geometric morphometrics (Klingenberg et al. 2002),
restricted to a pure reflection.

Baseline measure (variant M0, re-implemented from nashokan/FacialAsymmetry)
---------------------------------------------------------------------------
Vertical midline x = c through the mean x of mesh vertices 33 and 263; every vertex left
of the line is reflected and matched to its nearest right-side vertex with a KD-tree;
score = mean matched distance / face width (|x_454 - x_234|).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------------------
# Reflection primitives
# --------------------------------------------------------------------------------------
def reflect_points(P: np.ndarray, theta: float, rho: float) -> np.ndarray:
    """Reflect points P about the line n.x = rho with n = (cos theta, sin theta)."""
    n = np.array([np.cos(theta), np.sin(theta)])
    s = P @ n - rho
    return P - 2.0 * s[:, None] * n[None, :]


def reflect_vertical(P: np.ndarray, c: float) -> np.ndarray:
    """Reflect about the vertical line x = c, i.e. (x, y) -> (2c - x, y)."""
    Q = P.copy()
    Q[:, 0] = 2.0 * c - Q[:, 0]
    return Q


# --------------------------------------------------------------------------------------
# Optimal symmetry axis
# --------------------------------------------------------------------------------------
def _axis_init(P: np.ndarray, perm: np.ndarray) -> tuple[float, float]:
    """
    Closed-form initial guess. For each bilateral pair the vector between partners is
    (ideally) parallel to the axis normal, and their midpoint lies on the axis.
    """
    idx = np.where(perm != np.arange(len(perm)))[0]
    idx = idx[idx < perm[idx]]
    d = P[idx] - P[perm[idx]]
    # Orientation-free average of the pair directions (double-angle trick).
    ang = np.arctan2(d[:, 1], d[:, 0])
    w = np.linalg.norm(d, axis=1)
    theta = 0.5 * np.arctan2(np.sum(w * np.sin(2 * ang)), np.sum(w * np.cos(2 * ang)))
    n = np.array([np.cos(theta), np.sin(theta)])
    mids = 0.5 * (P[idx] + P[perm[idx]])
    rho = float(np.mean(mids @ n))
    return float(theta), rho


def fit_symmetry_axis(P: np.ndarray, perm: np.ndarray, weights: np.ndarray | None = None
                      ) -> tuple[float, float]:
    """Least-squares symmetry axis (theta, rho); see module docstring."""
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    Pp = P[perm]

    def cost(v):
        R = reflect_points(Pp, v[0], v[1]) - P
        return float(np.sum(w * np.sum(R * R, axis=1)))

    x0 = np.array(_axis_init(P, perm))
    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 2000})
    return float(res.x[0]), float(res.x[1])


def axis_segment(theta: float, rho: float, P: np.ndarray) -> np.ndarray:
    """Two end points of the symmetry axis spanning the vertical extent of P (for drawing)."""
    n = np.array([np.cos(theta), np.sin(theta)])
    u = np.array([-n[1], n[0]])                     # direction along the axis
    c = rho * n                                     # closest point of the line to the origin
    t = (P - c) @ u
    return np.stack([c + u * (t.min() - 10), c + u * (t.max() + 10)])


# --------------------------------------------------------------------------------------
# Proposed asymmetry measure
# --------------------------------------------------------------------------------------
@dataclass
class AsymmetryResult:
    per_point: np.ndarray          # a_i, fraction of IOD, shape (n,)
    global_score: float            # mean a_i over `subset`
    region_scores: dict            # region name -> mean a_i
    theta: float                   # fitted axis
    rho: float
    iod: float                     # normaliser in pixels
    residuals: np.ndarray          # r_i in pixels, shape (n, 2)


def iod(P: np.ndarray, eye_pair: tuple[int, int]) -> float:
    """Inter-ocular distance between the two outer eye corners."""
    return float(np.linalg.norm(P[eye_pair[0]] - P[eye_pair[1]]))


def asymmetry(P: np.ndarray, perm: np.ndarray, regions: dict, subset: list[int],
              eye_pair: tuple[int, int], fit_on: list[int] | None = None) -> AsymmetryResult:
    """
    Compute per-point, global and regional asymmetry with a fitted symmetry axis.

    fit_on : indices used to fit the axis (default: `subset`). Using the inner face
             avoids letting the ill-defined jaw contour tilt the axis.
    """
    fit_on = subset if fit_on is None else fit_on
    w = np.zeros(len(P))
    w[fit_on] = 1.0
    # A pair only contributes if both partners are in the fitting set.
    w = w * w[perm]
    theta, rho = fit_symmetry_axis(P, perm, weights=w)
    R = reflect_points(P[perm], theta, rho) - P
    scale = iod(P, eye_pair)
    a = np.linalg.norm(R, axis=1) / (2.0 * scale)
    regs = {k: float(np.mean(a[v])) for k, v in regions.items()}
    return AsymmetryResult(a, float(np.mean(a[subset])), regs, theta, rho, scale, R)


def axis_tilt_deg(theta: float) -> float:
    """Angle between the fitted symmetry axis and the image vertical, in degrees."""
    # The normal of a vertical axis is horizontal (theta = 0 or pi).
    t = np.degrees(theta) % 180.0
    return float(t if t <= 90 else t - 180.0)


# --------------------------------------------------------------------------------------
# Baseline measure (re-implementation of the public FacialAsymmetry method)
# --------------------------------------------------------------------------------------
def baseline_kdtree_asymmetry(L: np.ndarray, mid_a: int = 33, mid_b: int = 263,
                              cheek_a: int = 234, cheek_b: int = 454,
                              margin: float = 1.5, perm: np.ndarray | None = None
                              ) -> dict:
    """
    Reflected nearest-neighbour matching on the dense mesh with a vertical midline.

    If `perm` is given, the KD-tree is replaced by the fixed mesh correspondence
    (used in the correspondence ablation); everything else is unchanged.
    Returns the global score (mean distance / face width) and the matched pairs.
    """
    c = 0.5 * (L[mid_a, 0] + L[mid_b, 0])
    width = abs(L[cheek_b, 0] - L[cheek_a, 0]) or 1.0
    left = np.where(L[:, 0] < c - margin)[0]
    right = np.where(L[:, 0] > c + margin)[0]
    if len(left) == 0 or len(right) == 0:
        return {"global_score": float("nan"), "pairs": np.zeros((0, 2), int),
                "distances": np.zeros(0), "midline_x": c, "face_width": width}
    refl_left = reflect_vertical(L[left], c)
    if perm is None:
        _, j = cKDTree(L[right]).query(refl_left)
        partners = right[j]
    else:
        partners = perm[left]
    d = np.linalg.norm(refl_left - L[partners], axis=1)
    return {"global_score": float(np.mean(d) / width),
            "pairs": np.stack([left, partners], 1), "distances": d,
            "midline_x": c, "face_width": width}


def mapping_cross_structure_rate(pairs: np.ndarray, perm: np.ndarray) -> float:
    """
    Fraction of KD-tree matches whose partner is NOT the anatomical mirror vertex.
    A diagnostic of how often nearest-neighbour matching pairs unrelated structures.
    """
    if len(pairs) == 0:
        return float("nan")
    return float(np.mean(perm[pairs[:, 0]] != pairs[:, 1]))


# --------------------------------------------------------------------------------------
# Mirror-consistency (flip test)
# --------------------------------------------------------------------------------------
def unflip_landmarks(Lf: np.ndarray, width: int, perm: np.ndarray) -> np.ndarray:
    """
    Map landmarks detected on a horizontally flipped image back to the original frame.

    MediaPipe returns continuous coordinates (x_norm * W, pixel centres at i + 0.5), so a
    point at x' in the flipped image is at W - x' in the original. Flipping also
    swaps anatomical sides, so the vertex detected as i in the flipped image is the
    original's vertex perm[i]; we therefore re-index with the mirror permutation.
    """
    U = Lf.copy()
    U[:, 0] = width - U[:, 0]
    out = np.empty_like(U)
    out[perm] = U
    return out


def mirror_inconsistency(L: np.ndarray, L_unflipped: np.ndarray, scale: float,
                         idx: list[int] | None = None) -> float:
    """Mean distance between the two detections, as a fraction of `scale` (e.g. IOD)."""
    idx = slice(None) if idx is None else idx
    return float(np.mean(np.linalg.norm(L[idx] - L_unflipped[idx], axis=1)) / scale)
