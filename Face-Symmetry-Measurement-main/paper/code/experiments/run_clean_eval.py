"""
Clean-image evaluation on the LFPW TEST split (experiments E1, E2, E3, E8 and the
clean part of E9).

Reference: the same asymmetry measure computed on the human-annotated 68 landmarks
(A_ref). This is NOT a human symmetry rating; it is the geometric quantity we want to
measure, computed from expert landmarks instead of detector landmarks. Measurement
error = |A_detector - A_ref|.

Outputs (results/tables): test_clean_rows.csv, agreement_global.csv, agreement_regions.csv,
correspondence_ablation.csv, risk_coverage.csv, gate_operating_point.csv,
pose_strata.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import REGIONS, TABLES, agreement, load_cache, risk_coverage, row_for

df = pd.DataFrame([row_for(r, r["gt68"], extra={"image": r["image"]})
                   for r in load_cache("test")])
df.to_csv(os.path.join(TABLES, "test_clean_rows.csv"), index=False)
ok = df[df.ok].copy()
print(f"test images: {len(df)}, detected: {len(ok)}")

# ---------------- E1/E2: agreement of each measure with the reference ----------------
rows = []
for name, col in [("M0 baseline (KD-tree, 478 mesh, align+redetect)", "M0_score"),
                  ("M0 without alignment", "M0noalign_score"),
                  ("M1 fixed pairs + fitted axis (68)", "M1_score"),
                  ("M3 mirror-averaged landmarks (68)", "M3_score")]:
    a = agreement(ok, col)
    # MAE only meaningful for measures on the same scale as A_ref (68-layout, % IOD)
    if not col.startswith("M1") and not col.startswith("M3"):
        a["MAE"] = np.nan
    rows.append({"measure": name, **a})
ag = pd.DataFrame(rows)
ag.to_csv(os.path.join(TABLES, "agreement_global.csv"), index=False)
print(ag.round(3).to_string(index=False))

# ---------------- E3: regional agreement --------------------------------------------
rows = []
for k in REGIONS:
    for v in ("M1", "M3"):
        a = agreement(ok, f"{v}_{k}", ref=f"ref_{k}")
        rows.append({"region": k, "variant": v, **a,
                     "ref_mean": ok[f"ref_{k}"].mean(), "det_mean": ok[f"{v}_{k}"].mean()})
rg = pd.DataFrame(rows)
rg.to_csv(os.path.join(TABLES, "agreement_regions.csv"), index=False)
print(rg.round(3).to_string(index=False))

# ---------------- correspondence ablation (same detections, same midline) ------------
cr = pd.DataFrame([
    {"correspondence": "KD-tree reflected nearest neighbour", **agreement(ok, "M0_score")},
    {"correspondence": "fixed canonical mirror pairs", **agreement(ok, "M0fixed_score")},
])
cr["MAE"] = np.nan
cr["kdtree_wrong_partner_rate_mean"] = [ok["M0_cross_rate"].mean(), 0.0]
cr.to_csv(os.path.join(TABLES, "correspondence_ablation.csv"), index=False)
print(cr.round(3).to_string(index=False))

# ---------------- E9 (clean): gate as a selective-measurement mechanism ---------------
ok["err_M1"] = (ok["M1_score"] - ok["A_ref"]).abs()
ok["err_M3"] = (ok["M3_score"] - ok["A_ref"]).abs()
rng = np.random.default_rng(0)
curves, summ = [], []
signals = {
    "random": (rng.random(len(ok)), "err_M1"),
    "M2 gate risk (size/blur/exposure/pose)": (ok["M2_risk"].values, "err_M1"),
    "|yaw| only": (ok["sig_abs_yaw"].values, "err_M1"),
    "mirror inconsistency only": (ok["sig_mirror_incons"].values, "err_M1"),
    "M3 gate risk (M2 + mirror), M3 score": (ok["M3_risk"].values, "err_M3"),
}
for name, (risk, ecol) in signals.items():
    cov, rc, aurc = risk_coverage(ok[ecol].values, risk)
    curves.append(pd.DataFrame({"signal": name, "coverage": cov, "mean_abs_err": rc}))
    summ.append({"signal": name, "error_col": ecol, "AURC": aurc,
                 "err_at_100pct": rc[-1], "err_at_80pct": rc[int(0.8 * len(rc)) - 1],
                 "err_at_50pct": rc[int(0.5 * len(rc)) - 1],
                 "spearman_risk_vs_err": spearmanr(risk, ok[ecol]).statistic})
pd.concat(curves).to_csv(os.path.join(TABLES, "risk_coverage.csv"), index=False)
sm = pd.DataFrame(summ)
sm.to_csv(os.path.join(TABLES, "risk_coverage_summary.csv"), index=False)
print(sm.round(4).to_string(index=False))

# Operating point of the discrete gate
op = []
for v, ecol in (("M2", "err_M1"), ("M3", "err_M3")):
    for st in ("accept", "caution", "reject"):
        s = ok[ok[f"{v}_status"] == st]
        op.append({"variant": v, "status": st, "n": len(s), "pct": 100 * len(s) / len(ok),
                   "mean_abs_err": s[ecol].mean() if len(s) else np.nan,
                   "median_abs_err": s[ecol].median() if len(s) else np.nan,
                   "mean_nme_inner": s["nme_inner_pct"].mean() if len(s) else np.nan})
op = pd.DataFrame(op)
op.to_csv(os.path.join(TABLES, "gate_operating_point.csv"), index=False)
print(op.round(3).to_string(index=False))

# Which signal triggered each non-accept decision (M3 gate)
from fsm.quality import gate_decision, load_gate  # noqa: E402
gate = load_gate()
cnt = {}
for _, r in ok.iterrows():
    sig = {k[4:]: r[k] for k in ok.columns if k.startswith("sig_")}
    for why in gate_decision(sig, gate, use_mirror=True)[1]:
        key = (why["signal"], why["level"])
        cnt[key] = cnt.get(key, 0) + 1
rc = pd.DataFrame([{"signal": k[0], "level": k[1], "n_images": v} for k, v in cnt.items()])
rc = rc.sort_values(["level", "n_images"], ascending=[False, False])
rc.to_csv(os.path.join(TABLES, "gate_reasons.csv"), index=False)
print(rc.to_string(index=False))

# Landmark error (NME) as the target: can the mirror test detect bad landmarks
# without ground truth?
rows = []
for name, risk in [("random", rng.random(len(ok))), ("M2 gate risk", ok["M2_risk"].values),
                   ("mirror inconsistency", ok["sig_mirror_incons"].values),
                   ("M3 gate risk", ok["M3_risk"].values),
                   ("oracle (true NME)", ok["nme_inner_pct"].values)]:
    cov, rcurve, aurc = risk_coverage(ok["nme_inner_pct"].values, risk)
    rows.append({"signal": name, "AURC_nme": aurc, "nme_at_80pct": rcurve[int(0.8 * len(rcurve)) - 1],
                 "nme_at_50pct": rcurve[int(0.5 * len(rcurve)) - 1],
                 "spearman_vs_nme": spearmanr(risk, ok["nme_inner_pct"]).statistic})
    curves.append(pd.DataFrame({"signal": "NME|" + name, "coverage": cov, "mean_abs_err": rcurve}))
nm = pd.DataFrame(rows)
nm.to_csv(os.path.join(TABLES, "risk_coverage_nme.csv"), index=False)
pd.concat(curves).to_csv(os.path.join(TABLES, "risk_coverage.csv"), index=False)
print(nm.round(4).to_string(index=False))

# Pose confound: how strongly does each score follow |yaw|, before and after gating?
pc = []
for subset_name, sub in [("all", ok), ("M3 accept", ok[ok.M3_status == "accept"]),
                         ("M3 accept+caution", ok[ok.M3_status != "reject"])]:
    pc.append({"subset": subset_name, "n": len(sub),
               "rho(A_ref,|yaw|)": spearmanr(sub["A_ref"], sub["sig_abs_yaw"]).statistic,
               "rho(M3,|yaw|)": spearmanr(sub["M3_score"], sub["sig_abs_yaw"]).statistic,
               "rho(M0,|yaw|)": spearmanr(sub["M0_score"], sub["sig_abs_yaw"]).statistic,
               "A_ref_mean": sub["A_ref"].mean(), "M3_mean": sub["M3_score"].mean()})
pc = pd.DataFrame(pc)
pc.to_csv(os.path.join(TABLES, "pose_confound.csv"), index=False)
print(pc.round(3).to_string(index=False))

# Shrinkage: detector-based asymmetry vs annotation-based (paired, per region)
sh = []
from scipy.stats import wilcoxon  # noqa: E402
for k in ["global"] + REGIONS:
    a = ok["A_ref"] if k == "global" else ok[f"ref_{k}"]
    b = ok["M3_score"] if k == "global" else ok[f"M3_{k}"]
    sh.append({"region": k, "ref_mean": a.mean(), "detector_mean": b.mean(),
               "ratio": b.mean() / a.mean(), "wilcoxon_p": wilcoxon(a, b).pvalue})
sh = pd.DataFrame(sh)
sh.to_csv(os.path.join(TABLES, "shrinkage.csv"), index=False)
print(sh.round(4).to_string(index=False))

# ---------------- E8: pose stratification ------------------------------------------
ok["yaw_bin"] = pd.cut(ok["sig_abs_yaw"], [0, 5, 10, 20, 90], right=False,
                       labels=["0-5", "5-10", "10-20", ">=20"])
ps = ok.groupby("yaw_bin", observed=False).agg(
    n=("A_ref", "size"), A_ref_mean=("A_ref", "mean"), M1_mean=("M1_score", "mean"),
    err_M1=("err_M1", "mean"), err_M3=("err_M3", "mean"),
    nme_inner=("nme_inner_pct", "mean"), mirror_incons=("sig_mirror_incons", "mean"),
    M0_mean=("M0_score", "mean")).reset_index()
ps.to_csv(os.path.join(TABLES, "pose_strata.csv"), index=False)
print(ps.round(3).to_string(index=False))
print("corr(mirror_incons, nme):", spearmanr(ok["sig_mirror_incons"], ok["nme_inner_pct"]).statistic)
print("corr(mirror_incons, err_M1):", spearmanr(ok["sig_mirror_incons"], ok["err_M1"]).statistic)
print("mean runtime per image (3 detections) s:", ok["detect_s"].mean())
