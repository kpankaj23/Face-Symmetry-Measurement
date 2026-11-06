"""
mesh.py -- landmark topologies, bilateral correspondences and anatomical regions.

Two landmark layouts are used in this project:

1. The 478-point MediaPipe Face Landmarker mesh (468 face vertices + 10 iris points).
   Its bilateral correspondence is NOT guessed per image: it is derived once from
   Google's canonical face model (configs/canonical_face_model.obj), which is exactly
   left-right symmetric. Vertex i is paired with the vertex whose canonical position
   is the mirror image (-x, y, z) of vertex i.

2. The 68-point iBUG / 300W layout used by the LFPW human annotations. Its bilateral
   pairs are fixed by the published annotation scheme (e.g. point 37 <-> 46 for the
   outer eye corners, 1-indexed). All reported asymmetry measures are computed on this
   layout so that detector-based measurements can be compared with measurements from
   human annotations of the same points.

The MediaPipe->68 correspondence (which mesh vertex plays the role of each iBUG point)
is *fitted on the LFPW training split* by experiments/fit_mp68_mapping.py and stored in
configs/mp68_mapping.json. It is constrained to be mirror-consistent (see that script).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(ROOT, "configs")

# --------------------------------------------------------------------------------------
# 68-point iBUG layout (0-indexed)
# --------------------------------------------------------------------------------------
# Bilateral pairs (left, right) following the 300W markup. Points not listed are on the
# facial midline and map to themselves under reflection.
IBUG68_PAIRS = (
    # jaw line 0..16 (8 = chin is central)
    [(i, 16 - i) for i in range(8)]
    # eyebrows 17..21 <-> 26..22
    + [(17 + i, 26 - i) for i in range(5)]
    # lower nose 31,32 <-> 35,34 (33 = subnasale is central; 27..30 bridge is central)
    + [(31, 35), (32, 34)]
    # eyes: 36..41 (one eye) <-> 45..42,47,46 (other eye)
    + [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)]
    # outer lips: 48<->54, 49<->53, 50<->52, 59<->55, 58<->56 (51, 57 central)
    + [(48, 54), (49, 53), (50, 52), (59, 55), (58, 56)]
    # inner lips: 60<->64, 61<->63, 67<->65 (62, 66 central)
    + [(60, 64), (61, 63), (67, 65)]
)
IBUG68_CENTRAL = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]

# Anatomical regions of the 68-point layout used for regional reporting.
IBUG68_REGIONS = {
    "Brows": list(range(17, 27)),
    "Eyes": list(range(36, 48)),
    "Nose": list(range(27, 36)),
    "Mouth": list(range(48, 68)),
    "Jaw": list(range(0, 17)),
}
# Inner-face points (everything except the jaw contour). The jaw contour of 300W is a
# "silhouette" whose exact position along the cheek is ill-defined, and MediaPipe's
# face oval does not correspond to it point-for-point, so headline numbers use the
# inner 51 points and the jaw is reported separately.
IBUG68_INNER = list(range(17, 68))

# Outer eye corners: inter-ocular distance (IOD) normaliser, as in the 300W NME metric.
IBUG68_OUTER_EYE = (36, 45)


def ibug68_permutation() -> np.ndarray:
    """Return perm[i] = index of the mirror partner of point i (central points -> self)."""
    perm = np.arange(68)
    for a, b in IBUG68_PAIRS:
        perm[a], perm[b] = b, a
    return perm


# --------------------------------------------------------------------------------------
# 478-point MediaPipe mesh
# --------------------------------------------------------------------------------------
MP_N = 478
MP_IRIS = list(range(468, 478))  # 468 = one iris centre, 469-472 ring; 473 = other centre, 474-477 ring


@lru_cache(maxsize=1)
def canonical_vertices() -> np.ndarray:
    """Load the 468 canonical face-model vertices (x, y, z) shipped with MediaPipe."""
    path = os.path.join(CONFIG_DIR, "canonical_face_model.obj")
    verts = [list(map(float, ln.split()[1:4])) for ln in open(path) if ln.startswith("v ")]
    return np.asarray(verts)


@lru_cache(maxsize=1)
def mp_permutation() -> np.ndarray:
    """
    Mirror permutation of the 478-point mesh.

    For the 468 face vertices: reflect the canonical model (x -> -x) and find the vertex
    at the reflected position (the canonical model is exactly symmetric, max error 0).
    Iris centres 468 <-> 473 are paired; iris ring points are paired by their order
    (469..472 <-> 474..477), which is verified empirically by the flip test in tests/.
    """
    V = canonical_vertices()
    M = V.copy()
    M[:, 0] *= -1.0
    _, idx = cKDTree(V).query(M)
    perm = np.arange(MP_N)
    perm[:468] = idx
    # iris: centre<->centre; ring points paired so that horizontal neighbours swap
    iris_a = [468, 469, 470, 471, 472]
    iris_b = [473, 476, 475, 474, 477]  # see tests/test_mesh.py::test_iris_flip_pairing
    for a, b in zip(iris_a, iris_b):
        perm[a], perm[b] = b, a
    return perm


def mp_bilateral_pairs() -> np.ndarray:
    """(K, 2) array of mesh-vertex pairs (i, mirror(i)) with i < mirror(i)."""
    perm = mp_permutation()
    return np.array([(i, j) for i, j in enumerate(perm) if i < j], dtype=int)


@lru_cache(maxsize=1)
def mp68_mapping() -> np.ndarray:
    """
    Load the fitted MediaPipe-vertex index for each of the 68 iBUG points.
    Falls back to a commonly used hand-made mapping if the fitted file is absent.
    """
    path = os.path.join(CONFIG_DIR, "mp68_mapping.json")
    if os.path.exists(path):
        with open(path) as f:
            return np.asarray(json.load(f)["mp_index_for_ibug68"], dtype=int)
    return np.asarray(HANDMADE_MP68, dtype=int)


# A widely circulated hand-made MediaPipe->68 correspondence (used only as a starting
# point / fallback; the fitted mapping replaces it).
HANDMADE_MP68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
]


def mp_to_68(lm478: np.ndarray) -> np.ndarray:
    """Select the 68 mesh vertices that play the role of the iBUG points."""
    return lm478[mp68_mapping()]
