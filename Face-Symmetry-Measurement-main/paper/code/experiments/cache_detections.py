"""
Run the detector once on every LFPW image (original, mirrored, baseline-aligned) and
cache the raw outputs, so all clean-image experiments use identical detections.

    python experiments/cache_detections.py --split train
    python experiments/cache_detections.py --split test
Writes results/raw/detections_<split>.pkl
"""
import argparse
import os
import pickle
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fsm.data import crop_to_annotation, list_split, read_pts  # noqa: E402
from fsm.pipeline import raw_detect  # noqa: E402
from fsm.mesh import ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["train", "test"])
args = ap.parse_args()

rows = []
items = list_split(args.split)
for k, (im, pts) in enumerate(items):
    img, gt = crop_to_annotation(cv2.imread(im), read_pts(pts))
    raw = raw_detect(img)
    raw["image"] = os.path.basename(im)
    raw["gt68"] = gt
    rows.append(raw)
    if k % 100 == 0:
        print(f"{args.split}: {k}/{len(items)}", flush=True)

out = os.path.join(ROOT, "results", "raw", f"detections_{args.split}.pkl")
with open(out, "wb") as f:
    pickle.dump(rows, f)
n_ok = sum(r["ok"] for r in rows)
print(f"saved {out}: {n_ok}/{len(rows)} faces detected")
