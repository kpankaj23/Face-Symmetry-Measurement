"""
Fit the MediaPipe-vertex <-> iBUG-68 correspondence on the LFPW TRAINING split.

For every iBUG point k and mesh vertex v we compute the median (over training images)
of ||L_v - GT_k|| / IOD_gt. The mapping is chosen under a mirror constraint so that the
correspondence itself cannot create left-right asymmetry:
  * bilateral pair (a, b):  pick v minimising E[a, v] + E[b, mirror(v)]; a->v, b->mirror(v)
  * midline point k:        pick v among midline vertices (mirror(v) == v)
The fitted mapping is then evaluated on the TEST split (NME, inner 51 points) and
compared with the commonly circulated hand-made mapping.

Writes configs/mp68_mapping.json and results/tables/mapping_nme.csv
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fsm import mesh as MS  # noqa: E402
from fsm.mesh import ROOT  # noqa: E402


def load(split):
    with open(os.path.join(ROOT, "results", "raw", f"detections_{split}.pkl"), "rb") as f:
        return [r for r in pickle.load(f) if r["ok"]]


def nme(mapping, rows, idx):
    """Per-image NME (inter-ocular normalised mean error) on point subset idx."""
    out = []
    for r in rows:
        P, G = r["L"][mapping], r["gt68"]
        d = np.linalg.norm(G[36] - G[45])
        out.append(np.mean(np.linalg.norm(P[idx] - G[idx], axis=1)) / d)
    return np.array(out)


train, test = load("train"), load("test")
perm = MS.mp_permutation()
# E[k, v]: median normalised error of vertex v as a stand-in for iBUG point k
errs = []
for r in train:
    G = r["gt68"]
    d = np.linalg.norm(G[36] - G[45])
    L = r["L"][:468]
    errs.append(np.linalg.norm(G[:, None, :] - L[None, :, :], axis=2) / d)
E = np.median(np.stack(errs), axis=0)                       # (68, 468)

mapping = np.zeros(68, dtype=int)
central_v = np.where(perm[:468] == np.arange(468))[0]
for a, b in MS.IBUG68_PAIRS:
    cost = E[a] + E[b, perm[:468]]
    v = int(np.argmin(cost))
    mapping[a], mapping[b] = v, int(perm[v])
for k in MS.IBUG68_CENTRAL:
    mapping[k] = int(central_v[np.argmin(E[k, central_v])])

# Sanity: the mapping must commute with mirroring.
P68 = MS.ibug68_permutation()
assert np.all(perm[mapping] == mapping[P68]), "mapping is not mirror-consistent"

hand = np.asarray(MS.HANDMADE_MP68)
rows = []
for name, m in [("hand-made", hand), ("fitted (train)", mapping)]:
    for subset, idx in [("inner51", MS.IBUG68_INNER), ("jaw17", list(range(17))),
                        ("all68", list(range(68)))]:
        e = nme(m, test, idx)
        rows.append({"mapping": name, "points": subset, "test_NME_mean_pct": 100 * e.mean(),
                     "test_NME_median_pct": 100 * np.median(e), "n_images": len(e)})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "tables", "mapping_nme.csv"), index=False)
print(df.round(3).to_string(index=False))
hand_sym = bool(np.all(perm[hand] == hand[P68]))
print("hand-made mapping mirror-consistent:", hand_sym)

with open(os.path.join(ROOT, "configs", "mp68_mapping.json"), "w") as f:
    json.dump({"mp_index_for_ibug68": mapping.tolist(),
               "fitted_on": "LFPW trainset (%d images)" % len(train),
               "handmade_mirror_consistent": hand_sym}, f, indent=1)
