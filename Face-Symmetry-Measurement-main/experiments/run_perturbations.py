"""
Controlled perturbation study on the LFPW TEST split (experiments E4-E7 + face size).

Every test image (cropped to the annotated face, exactly as in the clean evaluation) is
perturbed with fixed, deterministic settings; the full pipeline is re-run, and the
annotation is transformed with the image where geometry changes (rotation, scaling).
Since the true face geometry does not change, the reference asymmetry A_ref is
unchanged (it is invariant to rotation and scale by construction).

Settings (no randomness; occluder positions come from the annotation):
  brightness  pixel values x {0.25, 0.4, 0.6, 1.5, 2.0, 2.5}  (clipped to [0, 255])
  blur        Gaussian, sigma = s * IOD / 100,  s in {1, 2, 4, 6, 8}
  occlusion   mid-grey box over one eye / the nose / the mouth (annotation box + 35 %)
  rotation    in-plane, {-30, -20, -10, -5, 5, 10, 20, 30} deg, canvas enlarged (no crop)
  downscale   image resized by {0.5, 0.3, 0.2, 0.12}
Writes results/tables/perturbation_rows.csv
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

from common import TABLES, load_cache, row_for
from fsm.data import crop_to_annotation, list_split, read_pts
from fsm.pipeline import raw_detect


def rotate(img, gt, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
    return out, gt @ M[:, :2].T + M[:, 2]


def occlude(img, gt, idx, pad=0.35):
    out = img.copy()
    x0, y0 = gt[idx].min(0)
    x1, y1 = gt[idx].max(0)
    px, py = pad * (x1 - x0) + 4, pad * (y1 - y0) + 4
    cv2.rectangle(out, (int(x0 - px), int(y0 - py)), (int(x1 + px), int(y1 + py)),
                  (128, 128, 128), -1)
    return out, gt


PERTURBATIONS = (
    [("brightness", k, lambda im, g, k=k: (np.clip(im.astype(np.float32) * k, 0, 255)
                                          .astype(np.uint8), g))
     for k in (0.25, 0.4, 0.6, 1.5, 2.0, 2.5)]
    + [("blur", s, lambda im, g, s=s: (cv2.GaussianBlur(
        im, (0, 0), s * np.linalg.norm(g[36] - g[45]) / 100.0), g))
       for s in (1, 2, 4, 6, 8)]
    + [("occlusion", name, lambda im, g, idx=idx: occlude(im, g, idx))
       for name, idx in (("one eye", list(range(36, 42))), ("nose", list(range(27, 36))),
                         ("mouth", list(range(48, 68))))]
    + [("rotation", a, lambda im, g, a=a: rotate(im, g, a))
       for a in (-30, -20, -10, -5, 5, 10, 20, 30)]
    + [("downscale", f, lambda im, g, f=f: (cv2.resize(im, None, fx=f, fy=f,
                                                       interpolation=cv2.INTER_AREA), g * f))
       for f in (0.5, 0.3, 0.2, 0.12)]
)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    items = list_split("test")[:limit]
    rows = []
    for n, (im, pts) in enumerate(items):
        img, gt = crop_to_annotation(cv2.imread(im), read_pts(pts))
        name = os.path.basename(im)
        for kind, level, fn in PERTURBATIONS:
            pimg, pgt = fn(img, gt)
            r = row_for(raw_detect(pimg), pgt, extra={"image": name, "perturbation": kind,
                                                     "level": str(level)})
            rows.append(r)
        if n % 20 == 0:
            print(f"{n}/{len(items)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES, "perturbation_rows.csv"), index=False)
    print("saved", len(df), "rows")
