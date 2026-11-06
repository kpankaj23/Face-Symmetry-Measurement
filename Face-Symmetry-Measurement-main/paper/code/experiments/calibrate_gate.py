"""
Calibrate the data-dependent gate thresholds on the LFPW TRAINING split only.

Rules (written down before looking at test results):
  * a-priori limits from the measurement's assumptions (not tuned):
      |yaw|   caution 10 deg, reject 20 deg   (left-right comparison in 2D)
      |pitch| caution 15 deg, reject 25 deg
      IOD     caution 60 px,  reject 30 px    (landmark quantisation becomes ~1-3 % of IOD)
      exposure reject: mean luma < 35 or > 225
  * statistic limits from the distribution of *clean* training images:
      sharpness caution = 2.5th percentile, reject = 0.5th percentile of train - 0.3
      brightness caution = 2.5th / 97.5th percentiles
      clipped caution = 97.5th percentile, reject = max(0.6, caution + 0.15)
      mirror inconsistency caution = 90th percentile, reject = 99th percentile
      per-region mirror limit = 97.5th percentile over all regions
  * readable index S = 100 / (1 + alpha * A[%]) with alpha s.t. the train median maps to 50.
Writes configs/gate_thresholds.json
"""
import json
import os

import numpy as np
import pandas as pd

from common import REGIONS, TABLES, load_cache, row_for
from fsm.mesh import CONFIG_DIR
from fsm.quality import DEFAULT_GATE

rows = [row_for(r, r["gt68"], gate=DEFAULT_GATE, extra={"image": r["image"]})
        for r in load_cache("train")]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(TABLES, "train_clean_rows.csv"), index=False)
ok = df[df.ok]
q = lambda c, p: float(np.percentile(ok[c].dropna(), p))  # noqa: E731

g = {k: dict(v) for k, v in DEFAULT_GATE.items()}
g["sharpness"].update(caution=q("sig_sharpness", 2.5), reject=q("sig_sharpness", 0.5) - 0.3)
g["brightness_lo"]["caution"] = q("sig_brightness", 2.5)
g["brightness_hi"]["caution"] = q("sig_brightness", 97.5)
g["clipped"]["caution"] = q("sig_clipped", 97.5)
# the face crop includes some background, so clipping is common; keep reject above caution
g["clipped"]["reject"] = max(0.6, g["clipped"]["caution"] + 0.15)
g["mirror_incons"].update(caution=q("sig_mirror_incons", 90), reject=q("sig_mirror_incons", 99))
region_limit = float(np.percentile(np.concatenate(
    [ok[f"incons_{k}"].dropna().values for k in REGIONS]), 97.5))
alpha = 1.0 / float(np.median(ok["M3_score"].dropna()))     # A in % of IOD

cfg = {"thresholds": g, "region_mirror_limit": region_limit, "index_alpha_per_pct": alpha,
       "calibrated_on": f"LFPW trainset, {len(ok)} detected faces of {len(df)}",
       "rules": __doc__}
with open(os.path.join(CONFIG_DIR, "gate_thresholds.json"), "w") as f:
    json.dump(cfg, f, indent=1)
print(json.dumps({k: v for k, v in cfg.items() if k != "rules"}, indent=1))
