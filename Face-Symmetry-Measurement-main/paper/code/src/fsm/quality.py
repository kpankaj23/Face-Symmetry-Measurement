"""
quality.py -- observable image-quality signals and the accept / caution / reject gate.

Every signal is something the system can actually measure on the input; we do not
invent a per-landmark confidence (MediaPipe's Face Landmarker returns none: its
`visibility` and `presence` fields are None).

Signals
-------
iod_px         inter-ocular distance in pixels (face size)
sharpness      log10 variance of the Laplacian of the face crop resized to 256 px width
brightness     mean luma (0-255) of the face crop
clipped        fraction of face-crop pixels that are near-black (<16) or near-white (>239)
yaw, pitch     head rotation magnitudes (degrees) from the facial transformation matrix
mirror_incons  mean |L - unflip(detect(flip(I)))| / IOD on the inner face (proposed)
n_faces        number of faces found

Thresholds are stored in configs/gate_thresholds.json. Pose and size limits are set a
priori from the measurement's assumptions; image-statistic limits are calibrated on the
LFPW *training* split by experiments/calibrate_gate.py (never on the test split).
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .mesh import CONFIG_DIR

GATE_PATH = os.path.join(CONFIG_DIR, "gate_thresholds.json")

# Defaults (overwritten by the calibrated file when present).
# direction "max": larger is worse;  "min": smaller is worse.
DEFAULT_GATE = {
    "iod_px":        {"direction": "min", "caution": 60.0, "reject": 30.0},
    "sharpness":     {"direction": "min", "caution": 1.6, "reject": 1.0},
    "brightness_lo": {"direction": "min", "caution": 60.0, "reject": 35.0},
    "brightness_hi": {"direction": "max", "caution": 200.0, "reject": 225.0},
    "clipped":       {"direction": "max", "caution": 0.10, "reject": 0.25},
    "abs_yaw":       {"direction": "max", "caution": 10.0, "reject": 20.0},
    "abs_pitch":     {"direction": "max", "caution": 15.0, "reject": 25.0},
    "mirror_incons": {"direction": "max", "caution": 0.035, "reject": 0.06},
}

# Plain-language explanations shown in the app.
EXPLAIN = {
    "iod_px": "face is small in the image (few pixels between the eyes)",
    "sharpness": "face region looks blurred / out of focus",
    "brightness_lo": "face region is under-exposed (too dark)",
    "brightness_hi": "face region is over-exposed (too bright)",
    "clipped": "many face pixels are clipped to black or white",
    "abs_yaw": "head is turned left/right; 2D left-right comparison is distorted",
    "abs_pitch": "head is tilted up/down; vertical proportions are distorted",
    "mirror_incons": "landmarks are not reproducible on the mirrored image "
                     "(detector is unsure of the geometry)",
    "n_faces": "more than one face found; the first face was analysed",
}


def load_gate() -> dict:
    if os.path.exists(GATE_PATH):
        with open(GATE_PATH) as f:
            return json.load(f)["thresholds"]
    return DEFAULT_GATE


def face_crop(img: np.ndarray, pts: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """Axis-aligned crop around the landmarks with a relative margin, clipped to the image."""
    h, w = img.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    x1, y1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return img
    return img[y0:y1, x0:x1]


def image_quality(img: np.ndarray, pts: np.ndarray) -> dict:
    """Sharpness / exposure statistics of the face region (independent of the detector)."""
    crop = face_crop(img, pts)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Resize to a fixed width so that the focus measure does not depend on face size.
    scale = 256.0 / max(1, gray.shape[1])
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return {
        "sharpness": float(np.log10(lap_var + 1e-6)),
        "brightness": float(g.mean()),
        "clipped": float(np.mean((g < 16) | (g > 239))),
    }


def gate_decision(signals: dict, gate: dict | None = None, use_mirror: bool = True
                  ) -> tuple[str, list[dict]]:
    """
    Combine signals into accept / caution / reject. The worst signal decides.
    Returns (status, reasons) where each reason names the signal, value and level.
    """
    gate = load_gate() if gate is None else gate
    vals = dict(signals)
    vals["brightness_lo"] = vals.get("brightness")
    vals["brightness_hi"] = vals.get("brightness")
    order = {"accept": 0, "caution": 1, "reject": 2}
    status, reasons = "accept", []
    for name, rule in gate.items():
        if name == "mirror_incons" and not use_mirror:
            continue
        v = vals.get(name)
        if v is None or not np.isfinite(v):
            continue
        bad = (lambda t: v > t) if rule["direction"] == "max" else (lambda t: v < t)
        level = "reject" if bad(rule["reject"]) else "caution" if bad(rule["caution"]) else None
        if level:
            reasons.append({"signal": name, "value": float(v), "level": level,
                            "why": EXPLAIN.get(name, name)})
            if order[level] > order[status]:
                status = level
    if signals.get("n_faces", 1) > 1:
        reasons.append({"signal": "n_faces", "value": signals["n_faces"], "level": "caution",
                        "why": EXPLAIN["n_faces"]})
        status = max(status, "caution", key=order.get)
    return status, reasons


def risk_score(signals: dict, gate: dict | None = None, use_mirror: bool = True) -> float:
    """
    Continuous risk used for risk-coverage curves: for each signal, how far it is past its
    caution threshold in units of the (caution -> reject) interval; the maximum is taken.
    0 = at the caution threshold, 1 = at the reject threshold, negative = comfortably fine.
    """
    gate = load_gate() if gate is None else gate
    vals = dict(signals)
    vals["brightness_lo"] = vals.get("brightness")
    vals["brightness_hi"] = vals.get("brightness")
    out = []
    for name, rule in gate.items():
        if name == "mirror_incons" and not use_mirror:
            continue
        v = vals.get(name)
        if v is None or not np.isfinite(v):
            continue
        span = (rule["reject"] - rule["caution"]) or 1e-9
        out.append((v - rule["caution"]) / span)   # sign handles both directions
    return float(max(out)) if out else 0.0
