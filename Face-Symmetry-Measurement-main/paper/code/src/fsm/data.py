"""
data.py -- reading the LFPW (300W subset) images and 68-point annotations.

Expected layout (see datasets/README.md for how to obtain the data):
    datasets/lfpw/trainset/image_XXXX.png + image_XXXX.pts   (811 images)
    datasets/lfpw/testset/image_XXXX.png  + image_XXXX.pts   (224 images)
The .pts files follow the iBUG format: a header, then 68 "x y" lines inside braces.
iBUG coordinates are 1-based (Matlab convention); we subtract 1 to obtain 0-based pixels.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .mesh import ROOT

DATA_ROOT = os.environ.get("FSM_DATA", os.path.join(ROOT, "datasets", "lfpw"))
SPLITS = {"train": "trainset", "test": "testset"}


def read_pts(path: str) -> np.ndarray:
    """Parse an iBUG .pts file into a (68, 2) float array of 0-based pixel coordinates."""
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    start = lines.index("{") + 1
    end = lines.index("}")
    pts = np.array([list(map(float, ln.split())) for ln in lines[start:end]])
    assert pts.shape == (68, 2), f"{path}: expected 68 points, got {pts.shape}"
    return pts - 1.0


def list_split(split: str) -> list[tuple[str, str]]:
    """Sorted list of (image_path, pts_path) for a split ('train' or 'test')."""
    d = os.path.join(DATA_ROOT, SPLITS[split])
    items = []
    for img in sorted(glob.glob(os.path.join(d, "*.png"))):
        pts = os.path.splitext(img)[0] + ".pts"
        if os.path.exists(pts):
            items.append((img, pts))
    return items


def crop_to_annotation(img, gt68, expand: float = 0.6):
    """
    Crop the image around the annotated face (benchmark protocol: 300W provides a face
    box per image). Many LFPW images contain several people; without this crop the
    detector may analyse a different person than the one annotated.
    The box is the landmark extent enlarged by `expand` on every side (relative to its
    size), clipped to the image. Returns (crop, gt68 shifted into crop coordinates).
    """
    h, w = img.shape[:2]
    x0, y0 = gt68.min(0)
    x1, y1 = gt68.max(0)
    bw, bh = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, x0 - expand * bw)), int(max(0, y0 - expand * bh))
    X1, Y1 = int(min(w, x1 + expand * bw)), int(min(h, y1 + expand * bh))
    return img[Y0:Y1, X0:X1].copy(), gt68 - np.array([X0, Y0], dtype=float)
