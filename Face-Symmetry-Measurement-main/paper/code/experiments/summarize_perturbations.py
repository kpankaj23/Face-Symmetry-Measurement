"""
Summarise the perturbation study (E4-E7) and build the ablation table (E9).

Metrics (all defined before running; tau fixed a priori at 1.0 % IOD, about twice the
clean-image mean absolute error):
  detect_rate       fraction of perturbed images where a face is found
  drift_<M>_sd      median |score(perturbed) - score(clean)| / SD of clean scores of M
                    (scale-free, so M0 (face-width units) and M1/M3 (IOD units) compare)
  mae_<M>           mean |A_det - A_ref| over images where a score is produced
  flag_<M>          fraction not accepted (caution or reject; failures count as reject)
  reject_<M>        fraction rejected
  silent_<M>        fraction of images where a score is REPORTED (status != reject) and
                    its error exceeds tau  -> the failure mode the gate must prevent
Writes results/tables/perturbation_summary.csv and ablation.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import TABLES

TAU = 1.0
clean = pd.read_csv(os.path.join(TABLES, "test_clean_rows.csv"))
pert = pd.read_csv(os.path.join(TABLES, "perturbation_rows.csv"))
sd = {k: clean[f"{k}_score"].std() for k in ("M0", "M0noalign", "M1", "M3")}
base = clean.set_index("image")
for k in ("M0", "M0noalign", "M1", "M3"):
    pert[f"clean_{k}"] = pert["image"].map(base[f"{k}_score"])
    pert[f"drift_{k}"] = (pert[f"{k}_score"] - pert[f"clean_{k}"]).abs() / sd[k]
for k in ("M1", "M3"):
    pert[f"err_{k}"] = (pert[f"{k}_score"] - pert["A_ref"]).abs()
for k in ("M2", "M3"):
    pert[f"{k}_status"] = pert[f"{k}_status"].fillna("reject")


def summarise(d):
    r = {"n": len(d), "detect_rate": d["ok"].mean()}
    for k in ("M0", "M0noalign", "M1", "M3"):
        r[f"drift_{k}_sd"] = d[f"drift_{k}"].median()
    r["mae_M1"] = d["err_M1"].mean()
    r["mae_M3"] = d["err_M3"].mean()
    for k in ("M2", "M3"):
        r[f"flag_{k}"] = (d[f"{k}_status"] != "accept").mean()
        r[f"reject_{k}"] = (d[f"{k}_status"] == "reject").mean()
    r["silent_M1"] = ((d["err_M1"] > TAU) & d["ok"]).mean()
    r["silent_M2"] = ((d["err_M1"] > TAU) & (d["M2_status"] != "reject")).mean()
    r["silent_M3"] = ((d["err_M3"] > TAU) & (d["M3_status"] != "reject")).mean()
    return pd.Series(r)


summ = pert.groupby(["perturbation", "level"], sort=False).apply(
    summarise, include_groups=False).reset_index()
summ.to_csv(os.path.join(TABLES, "perturbation_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(summ.round(3).to_string(index=False))

# ------------------------------------------------------------------ E9 ablation
clean = clean.copy()
clean["err_M1"] = (clean["M1_score"] - clean["A_ref"]).abs()
clean["err_M3"] = (clean["M3_score"] - clean["A_ref"]).abs()


def ablation_rows(d, setting):
    rows = []
    spec = [("M0", "M0", None, "baseline: KD-tree, vertical midline, align+redetect"),
            ("M1", "M1", None, "+ fixed pairs, fitted axis, 68-pt, IOD"),
            ("M2", "M1", "M2_status", "+ quality/pose gate"),
            ("M3", "M3", "M3_status", "+ mirror-averaged landmarks, mirror signal, region validity")]
    for name, scol, stcol, desc in spec:
        reported = d["ok"] if stcol is None else (d[stcol].fillna("reject") != "reject")
        accepted = d["ok"] if stcol is None else (d[stcol].fillna("reject") == "accept")
        sc = d[f"{scol}_score"]
        r = {"setting": setting, "variant": name, "description": desc, "n": len(d),
             "coverage_reported": reported.mean(), "coverage_accepted": accepted.mean()}
        dd = d[reported & sc.notna()]
        r["spearman_vs_ref"] = spearmanr(dd[f"{scol}_score"], dd["A_ref"]).statistic if len(dd) > 2 else np.nan
        r["MAE_reported"] = dd[f"err_{scol}"].mean() if scol != "M0" else np.nan
        r["silent_failure_rate"] = (((d[f"err_{scol}"] > TAU) & reported).mean()
                                    if scol != "M0" else np.nan)
        if "drift_M0" in d:
            r["median_drift_sd"] = d[f"drift_{scol}"].median()
        rows.append(r)
    return rows


rows = ablation_rows(clean, "clean test (224)")
for kind in ("brightness", "blur", "occlusion", "rotation", "downscale"):
    rows += ablation_rows(pert[pert.perturbation == kind], f"perturbed: {kind}")
rows += ablation_rows(pert, "perturbed: all (pooled)")
ab = pd.DataFrame(rows)
ab.to_csv(os.path.join(TABLES, "ablation.csv"), index=False)
print(ab.drop(columns="description").round(3).to_string(index=False))

# Region validity under occlusion: is the occluded region the one marked unreliable?
occ = pert[(pert.perturbation == "occlusion") & pert.ok]
target = {"one eye": "Eyes", "nose": "Nose", "mouth": "Mouth"}
rv = []
for lvl, reg in target.items():
    d = occ[occ.level == lvl]
    cl = clean.set_index("image")
    rv.append({"occluded": lvl, "n": len(d),
               "occluded_region_flagged": (~d[f"M3_{reg}_valid"].astype(bool)).mean(),
               "other_regions_flagged": np.mean([(~d[f"M3_{o}_valid"].astype(bool)).mean()
                                                 for o in target.values() if o != reg]),
               "clean_flag_rate_same_region": (~cl[f"M3_{reg}_valid"].astype(bool)).mean(),
               "region_err_M1": (d[f"M1_{reg}"] - d[f"ref_{reg}"]).abs().mean(),
               "region_err_clean": (cl[f"M1_{reg}"] - cl[f"ref_{reg}"]).abs().mean()})
rv = pd.DataFrame(rv)
rv.to_csv(os.path.join(TABLES, "region_validity_occlusion.csv"), index=False)
print(rv.round(3).to_string(index=False))
