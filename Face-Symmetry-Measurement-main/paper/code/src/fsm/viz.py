"""
viz.py -- drawing helpers shared by the Streamlit app and the paper figures.
All functions take a BGR image and return a new BGR image (input is never modified).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as G
from . import mesh as MS

REGION_COLORS = {          # BGR, colour-blind-friendly (Okabe-Ito)
    "Brows": (0, 159, 230),
    "Eyes": (233, 180, 86),
    "Nose": (115, 158, 0),
    "Mouth": (0, 94, 213),
    "Jaw": (167, 121, 204),
}
STATUS_COLORS = {"accept": (80, 160, 40), "caution": (0, 170, 240), "reject": (40, 40, 220)}


def _thick(img):
    return max(1, int(round(img.shape[1] / 400)))


def draw_mesh(img, L, color=(255, 255, 255), radius=None):
    out = img.copy()
    r = radius or max(1, _thick(img) - 1)
    for x, y in L:
        cv2.circle(out, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
    return out


def draw_axis(img, res: G.AsymmetryResult, P, color=(255, 0, 255)):
    out = img.copy()
    a, b = G.axis_segment(res.theta, res.rho, P)
    cv2.line(out, tuple(int(v) for v in a), tuple(int(v) for v in b), color,
             _thick(img) + 1, cv2.LINE_AA)
    return out


def draw_pairs(img, P, perm=None, color=(255, 255, 0)):
    """Connect each bilateral pair of the 68-point layout."""
    out = img.copy()
    perm = MS.ibug68_permutation() if perm is None else perm
    for i, j in enumerate(perm):
        if i < j and i in MS.IBUG68_INNER:
            cv2.line(out, tuple(P[i].astype(int)), tuple(P[j].astype(int)), color, 1, cv2.LINE_AA)
    return out


def draw_mirrored(img, res: G.AsymmetryResult, P):
    """Original points (green) and reflections of their partners (red), joined by residuals."""
    out = img.copy()
    perm = MS.ibug68_permutation()
    Q = G.reflect_points(P[perm], res.theta, res.rho)
    t = _thick(img)
    for i in MS.IBUG68_INNER:
        cv2.line(out, tuple(P[i].astype(int)), tuple(Q[i].astype(int)), (128, 128, 128), 1,
                 cv2.LINE_AA)
        cv2.circle(out, tuple(P[i].astype(int)), t + 1, (60, 200, 60), -1, cv2.LINE_AA)
        cv2.circle(out, tuple(Q[i].astype(int)), t + 1, (60, 60, 230), -1, cv2.LINE_AA)
    return out


def draw_heatmap(img, res: G.AsymmetryResult, P, vmax=0.06, valid: dict | None = None):
    """Colour each 68-point by its per-point asymmetry (fraction of IOD); greys out invalid regions."""
    out = (0.55 * img).astype(np.uint8) if img.mean() < 200 else img.copy()
    cmap = cv2.applyColorMap(np.uint8(np.clip(np.linspace(0, 1, 256) * 255, 0, 255))[:, None],
                             cv2.COLORMAP_INFERNO)[:, 0, :]
    t = _thick(img)
    region_of = {i: k for k, v in MS.IBUG68_REGIONS.items() for i in v}
    for i, (x, y) in enumerate(P):
        reg = region_of[i]
        if valid is not None and not valid.get(reg, True):
            col = (140, 140, 140)
        else:
            col = tuple(int(c) for c in cmap[int(np.clip(res.per_point[i] / vmax, 0, 1) * 255)])
        cv2.circle(out, (int(x), int(y)), 2 * t + 2, col, -1, cv2.LINE_AA)
    return out


def draw_regions(img, P):
    out = img.copy()
    t = _thick(img)
    for k, idx in MS.IBUG68_REGIONS.items():
        for i in idx:
            cv2.circle(out, tuple(P[i].astype(int)), 2 * t + 1, REGION_COLORS[k], -1, cv2.LINE_AA)
    return out


def draw_kdtree_matches(img, L, pairs, perm, max_lines=400):
    """Baseline KD-tree matches: correct anatomical partner (green) vs wrong partner (red)."""
    out = img.copy()
    for li, ri in pairs[:max_lines]:
        col = (60, 200, 60) if perm[li] == ri else (60, 60, 230)
        cv2.line(out, tuple(L[li].astype(int)), tuple(L[ri].astype(int)), col, 1, cv2.LINE_AA)
    return out


def status_banner(img, status, text=None):
    out = img.copy()
    h = max(24, img.shape[0] // 14)
    cv2.rectangle(out, (0, 0), (img.shape[1], h), STATUS_COLORS[status], -1)
    cv2.putText(out, (text or status.upper()), (8, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                h / 40, (255, 255, 255), max(1, h // 18), cv2.LINE_AA)
    return out
