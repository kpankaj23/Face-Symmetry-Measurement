"""
pipeline.py -- end-to-end measurement. Callable from experiments and from the app.

`measure(img)` runs the detector at most three times (original, mirrored, and the
baseline's aligned re-detection) and returns every quantity needed by the four
compared variants, so experiments evaluate all variants on exactly the same detections:

  M0  Baseline     re-implementation of nashokan/FacialAsymmetry: eye-corner affine
                   alignment + re-detection, vertical midline, KD-tree reflected
                   nearest-neighbour matching on the 478-point mesh, / face width.
  M1  Fixed-pair   fixed anatomical correspondence on the 68-point layout, fitted
                   symmetry axis (rotation invariant, no re-detection), / IOD.
  M2  +Gate        M1 + accept/caution/reject gate on size, blur, exposure, pose.
  M3  +Mirror      M2 with mirror-consistent landmarks: the landmark set is the average
                   of the detection on I and the un-flipped detection on flip(I); the
                   mirror inconsistency is added to the gate and used for per-region
                   validity.  (Proposed full system.)
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import geometry as G
from . import mesh as MS
from .detector import detect, pose_from_transform
from .quality import gate_decision, image_quality, load_gate, risk_score

PERM68 = MS.ibug68_permutation()
PERM478 = MS.mp_permutation()

# Per-region mirror-inconsistency limit (fraction of IOD) above which a region is marked
# "unreliable" in M3. Calibrated on the training split (calibrate_gate.py writes it into
# gate_thresholds.json under "region_mirror_limit").
DEFAULT_REGION_LIMIT = 0.05


def align_like_baseline(img: np.ndarray, L: np.ndarray, size=(600, 600)) -> np.ndarray:
    """Baseline alignment: similarity transform taking vertices 33, 263 to fixed positions."""
    w, h = size
    src = np.float32([L[33], L[263]])
    dst = np.float32([[w * 0.33, h * 0.33], [w * 0.67, h * 0.33]])
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return img if M is None else cv2.warpAffine(img, M, size)


def asym68(P68: np.ndarray) -> G.AsymmetryResult:
    """Proposed measure on a 68-point configuration (axis fitted on the inner face)."""
    return G.asymmetry(P68, PERM68, MS.IBUG68_REGIONS, subset=MS.IBUG68_INNER,
                       eye_pair=MS.IBUG68_OUTER_EYE, fit_on=MS.IBUG68_INNER)


def raw_detect(img: np.ndarray, run_baseline: bool = True, run_mirror: bool = True) -> dict:
    """
    All detector calls for one image (original, mirrored, baseline re-detection).
    Kept separate from the geometry so that experiments can cache detections.
    """
    t0 = time.perf_counter()
    raw: dict = {"ok": False, "failure": None}
    if img is None or img.size == 0:
        raw["failure"] = "unreadable image"
        return raw
    det = detect(img)
    if det is None:
        raw["failure"] = "no face detected"
        return raw
    raw.update(ok=True, width=img.shape[1], height=img.shape[0], L=det.landmarks,
               T=det.transform, n_faces=det.n_faces,
               quality=image_quality(img, det.landmarks[:468]))
    if run_mirror:
        detf = detect(cv2.flip(img, 1))
        raw["Lf"] = None if detf is None else detf.landmarks       # in flipped-image frame
    if run_baseline:
        det2 = detect(align_like_baseline(img, det.landmarks))
        raw["L_aligned"] = None if det2 is None else det2.landmarks
    raw["detect_s"] = time.perf_counter() - t0
    return raw


def measure_from_raw(raw: dict) -> dict:
    """Geometry for every variant from cached detections (no detector calls)."""
    out: dict = {"ok": raw.get("ok", False), "failure": raw.get("failure")}
    if not out["ok"]:
        return out
    L = raw["L"]
    P = MS.mp_to_68(L)
    out.update(n_faces=raw["n_faces"], lm478=L, lm68=P, transform=raw["T"],
               pose=pose_from_transform(raw["T"]), quality=raw["quality"],
               iod_px=G.iod(P, MS.IBUG68_OUTER_EYE), single=asym68(P))

    # ---- mirror-consistency (M3) ----------------------------------------------------
    if "Lf" in raw:
        if raw["Lf"] is None:
            out["mirror"] = None
        else:
            U = G.unflip_landmarks(raw["Lf"], raw["width"], PERM478)
            Pu = MS.mp_to_68(U)
            Pm = 0.5 * (P + Pu)
            inner = MS.IBUG68_INNER
            out["mirror"] = {
                "lm478_unflipped": U, "lm68_unflipped": Pu, "lm68_avg": Pm,
                "lm478_avg": 0.5 * (L + U),
                "incons": G.mirror_inconsistency(P, Pu, out["iod_px"], inner),
                "region_incons": {k: G.mirror_inconsistency(P, Pu, out["iod_px"], v)
                                  for k, v in MS.IBUG68_REGIONS.items()},
                "score_flip": asym68(Pu).global_score,
                "avg": asym68(Pm),
            }

    # ---- baseline (M0) ----------------------------------------------------------------
    if "L_aligned" in raw:
        if raw["L_aligned"] is None:
            out["baseline"] = None
        else:
            La = raw["L_aligned"]
            b = G.baseline_kdtree_asymmetry(La)
            b["cross_structure_rate"] = G.mapping_cross_structure_rate(b["pairs"], PERM478)
            b["fixed_pair_score"] = G.baseline_kdtree_asymmetry(La, perm=PERM478)["global_score"]
            # Same KD-tree measure on the *un-aligned* detection (rotation study)
            b["noalign_score"] = G.baseline_kdtree_asymmetry(L)["global_score"]
            b["lm478_aligned"] = La
            out["baseline"] = b
    out["detect_s"] = raw.get("detect_s")
    return out


def measure(img: np.ndarray, run_baseline: bool = True, run_mirror: bool = True) -> dict:
    """Measure everything for one BGR image. Never raises on a bad image: reports failure."""
    return measure_from_raw(raw_detect(img, run_baseline, run_mirror))


def signals_of(m: dict, use_mirror: bool) -> dict:
    """Collect the gate signals of a measurement."""
    s = {"iod_px": m["iod_px"], **m["quality"],
         "abs_yaw": abs(m["pose"]["yaw"]), "abs_pitch": abs(m["pose"]["pitch"]),
         "n_faces": m["n_faces"]}
    if use_mirror:
        s["mirror_incons"] = (m["mirror"]["incons"] if m.get("mirror") else np.inf)
    return s


def variant_outputs(m: dict, gate: dict | None = None) -> dict:
    """
    Turn one measurement into the four variants' user-facing outputs:
    score (or NaN if not produced), status, risk and region scores.
    """
    gate_cfg = load_gate() if gate is None else gate
    region_limit = DEFAULT_REGION_LIMIT
    try:
        import json, os
        with open(os.path.join(MS.CONFIG_DIR, "gate_thresholds.json")) as f:
            region_limit = json.load(f).get("region_mirror_limit", DEFAULT_REGION_LIMIT)
    except Exception:
        pass
    v = {}
    if not m.get("ok"):
        for k in ("M0", "M1", "M2", "M3"):
            v[k] = {"score": np.nan, "status": "reject", "risk": np.inf, "regions": {}}
        return v
    b = m.get("baseline")
    v["M0"] = {"score": b["global_score"] if b else np.nan, "status": "accept",
               "risk": 0.0, "regions": {}}
    s1 = m["single"]
    v["M1"] = {"score": s1.global_score, "status": "accept", "risk": 0.0,
               "regions": dict(s1.region_scores)}
    sig2 = signals_of(m, use_mirror=False)
    st2, why2 = gate_decision(sig2, gate_cfg, use_mirror=False)
    v["M2"] = {"score": s1.global_score, "status": st2, "reasons": why2,
               "risk": risk_score(sig2, gate_cfg, use_mirror=False),
               "regions": dict(s1.region_scores)}
    mir = m.get("mirror")
    sig3 = signals_of(m, use_mirror=True)
    st3, why3 = gate_decision(sig3, gate_cfg, use_mirror=True)
    if mir is None:
        st3 = "reject"
        why3 = why3 + [{"signal": "mirror", "value": np.nan, "level": "reject",
                        "why": "no face found in the mirrored image"}]
        v["M3"] = {"score": np.nan, "status": st3, "reasons": why3, "risk": np.inf,
                   "regions": {}, "region_valid": {}}
    else:
        a = mir["avg"]
        valid = {k: bool(mir["region_incons"][k] <= region_limit) for k in a.region_scores}
        v["M3"] = {"score": a.global_score, "status": st3, "reasons": why3,
                   "risk": risk_score(sig3, gate_cfg, use_mirror=True),
                   "regions": dict(a.region_scores), "region_valid": valid,
                   "stability_band": abs(m["single"].global_score - mir["score_flip"])}
    return v


def symmetry_index(A: float, alpha: float = 10.0) -> float:
    """
    Optional readable 0-100 index, S = 100 / (1 + alpha * 100 * A), A as fraction of IOD.
    alpha is chosen so that the LFPW-train median maps near 50 (see calibrate_gate.py).
    This is a monotone re-expression of A, NOT a validated clinical or aesthetic score.
    """
    return float(100.0 / (1.0 + alpha * 100.0 * A)) if np.isfinite(A) else float("nan")
