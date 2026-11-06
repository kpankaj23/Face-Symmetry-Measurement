"""Shared helpers for the experiment scripts."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fsm import mesh as MS  # noqa: E402
from fsm.mesh import ROOT  # noqa: E402
from fsm.pipeline import asym68, measure_from_raw, signals_of, variant_outputs  # noqa: E402

TABLES = os.path.join(ROOT, "results", "tables")
FIGS = os.path.join(ROOT, "results", "figures")
RAW = os.path.join(ROOT, "results", "raw")
REGIONS = list(MS.IBUG68_REGIONS)


def load_cache(split):
    with open(os.path.join(RAW, f"detections_{split}.pkl"), "rb") as f:
        return pickle.load(f)


def reference(gt68):
    """Reference asymmetry from the human-annotated landmarks (same measure as M1)."""
    return asym68(gt68)


def row_for(raw, gt68, gate=None, extra=None):
    """
    One CSV row: reference asymmetry, every variant's output and all gate signals.
    Scores are stored in % of IOD (x100) for readability.
    """
    ref = reference(gt68)
    m = measure_from_raw(raw)
    v = variant_outputs(m, gate)
    r = {"ok": m["ok"], "failure": m.get("failure"),
         "A_ref": 100 * ref.global_score,
         **{f"ref_{k}": 100 * x for k, x in ref.region_scores.items()}}
    if extra:
        r.update(extra)
    for k in ("M0", "M1", "M2", "M3"):
        r[f"{k}_score"] = 100 * v[k]["score"] if np.isfinite(v[k]["score"]) else np.nan
        r[f"{k}_status"] = v[k]["status"]
        r[f"{k}_risk"] = v[k]["risk"]
    if m["ok"]:
        for k in REGIONS:
            r[f"M1_{k}"] = 100 * v["M1"]["regions"][k]
            if v["M3"]["regions"]:
                r[f"M3_{k}"] = 100 * v["M3"]["regions"][k]
                r[f"M3_{k}_valid"] = v["M3"]["region_valid"][k]
        s = signals_of(m, use_mirror=True)
        r.update({f"sig_{k}": val for k, val in s.items()})
        r["yaw"], r["pitch"], r["roll"] = (m["pose"][q] for q in ("yaw", "pitch", "roll"))
        G = gt68
        iod = np.linalg.norm(G[36] - G[45])
        r["nme_inner_pct"] = 100 * np.mean(np.linalg.norm(
            m["lm68"][MS.IBUG68_INNER] - G[MS.IBUG68_INNER], axis=1)) / iod
        if m.get("mirror"):
            r["M3_band"] = 100 * v["M3"]["stability_band"]
            for k in REGIONS:
                r[f"incons_{k}"] = m["mirror"]["region_incons"][k]
        b = m.get("baseline")
        if b:
            r["M0fixed_score"] = 100 * b["fixed_pair_score"]
            r["M0noalign_score"] = 100 * b["noalign_score"]
            r["M0_cross_rate"] = b["cross_structure_rate"]
        r["detect_s"] = m.get("detect_s")
    return r


def agreement(df, col, ref="A_ref"):
    d = df[[col, ref]].dropna()
    if len(d) < 3:
        return {"n": len(d), "spearman": np.nan, "pearson": np.nan, "MAE": np.nan}
    return {"n": len(d), "spearman": spearmanr(d[col], d[ref]).statistic,
            "pearson": pearsonr(d[col], d[ref]).statistic,
            "MAE": float(np.mean(np.abs(d[col] - d[ref])))}


def risk_coverage(err, risk):
    """Selective-prediction curve: mean error of the retained set when keeping the lowest-risk fraction."""
    o = np.argsort(risk, kind="stable")
    e = np.asarray(err)[o]
    n = len(e)
    cov = np.arange(1, n + 1) / n
    risk_curve = np.cumsum(e) / np.arange(1, n + 1)
    return cov, risk_curve, float(np.mean(risk_curve))   # AURC = mean over coverages
