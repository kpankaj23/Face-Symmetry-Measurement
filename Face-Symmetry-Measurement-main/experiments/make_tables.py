"""
Write the paper's LaTeX tables directly from the result CSVs (no hand-typed numbers).
    python experiments/make_tables.py      ->  paper/tables/*.tex  and paper/tables/numbers.tex
`numbers.tex` defines \\newcommand macros for every number quoted in the text.
"""
import json
import os

import numpy as np
import pandas as pd

from common import TABLES
from fsm.mesh import CONFIG_DIR, ROOT

OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
T = lambda n: pd.read_csv(os.path.join(TABLES, n))  # noqa: E731
f2 = lambda x: "--" if pd.isna(x) else f"{x:.2f}"  # noqa: E731
f3 = lambda x: "--" if pd.isna(x) else f"{x:.3f}"  # noqa: E731
pc = lambda x: "--" if pd.isna(x) else f"{100 * x:.1f}"  # noqa: E731
macros = {}


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)


# ---------------------------------------------------------------- agreement (Tab. 3)
ag = T("agreement_global.csv")
cr = T("correspondence_ablation.csv")
mp = T("mapping_nme.csv")
rows = [
    ("M0 baseline~\\cite{nashokan_facialasymmetry} (KD-tree, vert.\\ midline)", ag.iloc[0], "face width"),
    ("\\quad M0 w/o alignment", ag.iloc[1], "face width"),
    ("\\quad M0 + fixed pairs (no KD-tree)", cr.iloc[1], "face width"),
    ("M1: 68-pt pairs + fitted axis", ag.iloc[2], "IOD"),
    ("M3: M1 + mirror averaging", ag.iloc[3], "IOD"),
]
body = ("\\begin{tabular}{@{}lccc@{}}\n\\toprule\nMeasure ($n$=224) & $\\rho$ & $r$ & MAE\\\\\n\\midrule\n")
for name, r, _ in rows:
    body += f"{name} & {f3(r['spearman'])} & {f3(r['pearson'])} & {f2(r['MAE'])}\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("agreement.tex", body)
macros.update(rhoMzero=f3(ag.iloc[0].spearman), rhoMzeroNoAlign=f3(ag.iloc[1].spearman),
              rhoMzeroFixed=f3(cr.iloc[1].spearman), rhoMone=f3(ag.iloc[2].spearman),
              rhoMthree=f3(ag.iloc[3].spearman), maeMone=f2(ag.iloc[2].MAE),
              maeMthree=f2(ag.iloc[3].MAE),
              kdWrong=f"{100 * cr.iloc[0].kdtree_wrong_partner_rate_mean:.1f}")
g = mp.set_index(["mapping", "points"])
macros.update(nmeFitInner=f2(g.loc[("fitted (train)", "inner51"), "test_NME_mean_pct"]),
              nmeHandInner=f2(g.loc[("hand-made", "inner51"), "test_NME_mean_pct"]),
              nmeFitJaw=f2(g.loc[("fitted (train)", "jaw17"), "test_NME_mean_pct"]),
              nmeHandJaw=f2(g.loc[("hand-made", "jaw17"), "test_NME_mean_pct"]),
              nmeFitInnerMed=f2(g.loc[("fitted (train)", "inner51"), "test_NME_median_pct"]))

# ---------------------------------------------------------------- regions + shrinkage
rg = T("agreement_regions.csv")
sh = T("shrinkage.csv").set_index("region")
body = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Spearman $\\rho$} & "
        "\\multicolumn{2}{c}{MAE (\\% IOD)} & \\multicolumn{2}{c}{mean A (\\% IOD)}\\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nRegion & M1 & M3 & M1 & M3 & ref & M3\\\\\n\\midrule\n")
for k in ["Brows", "Eyes", "Nose", "Mouth", "Jaw"]:
    a = rg[(rg.region == k) & (rg.variant == "M1")].iloc[0]
    b = rg[(rg.region == k) & (rg.variant == "M3")].iloc[0]
    star = "" if sh.loc[k, "wilcoxon_p"] < 0.01 else "$^{\\dagger}$"
    body += (f"{k} & {f3(a.spearman)} & {f3(b.spearman)} & {f2(a.MAE)} & {f2(b.MAE)} & "
             f"{f2(sh.loc[k, 'ref_mean'])} & {f2(sh.loc[k, 'detector_mean'])}{star}\\\\\n")
body += (f"\\midrule\nGlobal (inner 51) & -- & -- & -- & -- & {f2(sh.loc['global', 'ref_mean'])} & "
         f"{f2(sh.loc['global', 'detector_mean'])}\\\\\n\\bottomrule\n\\end{{tabular}}\n")
write("regions.tex", body)
macros.update(shrinkGlobal=f"{100 * (1 - sh.loc['global', 'ratio']):.1f}",
              shrinkMouth=f"{100 * (1 - sh.loc['Mouth', 'ratio']):.1f}",
              shrinkBrows=f"{100 * (1 - sh.loc['Brows', 'ratio']):.1f}",
              pBrows=f"{sh.loc['Brows', 'wilcoxon_p']:.2f}")

# ---------------------------------------------------------------- ablation (Tab. 4)
ab = T("ablation.csv")
settings = ["clean test (224)", "perturbed: all (pooled)", "perturbed: blur",
            "perturbed: downscale", "perturbed: brightness", "perturbed: occlusion",
            "perturbed: rotation"]
nice = {"clean test (224)": "Clean test", "perturbed: all (pooled)": "Perturbed, pooled",
        "perturbed: blur": "\\quad blur", "perturbed: downscale": "\\quad downscale",
        "perturbed: brightness": "\\quad brightness", "perturbed: occlusion": "\\quad occlusion",
        "perturbed: rotation": "\\quad rotation"}
body = ("\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\nSetting & Var. & n & Reported & Accepted & "
        "$\\rho$ & Silent fail. & Drift\\\\\n & & & (\\%) & (\\%) & vs ref & (\\%) $\\downarrow$ & (SD) $\\downarrow$\\\\\n\\midrule\n")
for s in settings:
    d = ab[ab.setting == s]
    for i, (_, r) in enumerate(d.iterrows()):
        lab = nice[s] if i == 0 else ""
        n = str(int(r.n)) if i == 0 else ""
        drift = f3(r.get("median_drift_sd", np.nan)) if "median_drift_sd" in r else "--"
        if r.variant == "M2" and "median_drift_sd" in r and not pd.isna(r.median_drift_sd):
            drift = "="  # same score as M1 (only the gate differs)
        body += (f"{lab} & {r.variant} & {n} & {pc(r.coverage_reported)} & {pc(r.coverage_accepted)} & "
                 f"{f3(r.spearman_vs_ref)} & {pc(r.silent_failure_rate)} & {drift}\\\\\n")
    body += "\\midrule\n" if s != settings[-1] else ""
body += "\\bottomrule\n\\end{tabular}\n"
write("ablation.tex", body)
A = ab.set_index(["setting", "variant"])
for s, tag in [("clean test (224)", "Clean"), ("perturbed: all (pooled)", "Pool"),
               ("perturbed: blur", "Blur"), ("perturbed: downscale", "Down")]:
    for v in ("M1", "M2", "M3"):
        macros[f"silent{tag}{v.replace('M1','One').replace('M2','Two').replace('M3','Three')}"] = pc(A.loc[(s, v), "silent_failure_rate"])
    macros[f"cov{tag}Three"] = pc(A.loc[(s, "M3"), "coverage_reported"])
    macros[f"acc{tag}Three"] = pc(A.loc[(s, "M3"), "coverage_accepted"])
macros["driftPoolOne"] = f3(A.loc[("perturbed: all (pooled)", "M1"), "median_drift_sd"])
macros["driftPoolThree"] = f3(A.loc[("perturbed: all (pooled)", "M3"), "median_drift_sd"])
macros["driftPoolZero"] = f3(A.loc[("perturbed: all (pooled)", "M0"), "median_drift_sd"])
macros["driftReduction"] = f"{100 * (1 - A.loc[('perturbed: all (pooled)', 'M3'), 'median_drift_sd'] / A.loc[('perturbed: all (pooled)', 'M1'), 'median_drift_sd']):.0f}"

ps = T("perturbation_summary.csv")
rot = ps[ps.perturbation == "rotation"]
macros["rotNoAlignMax"] = f2(rot.drift_M0noalign_sd.max())
macros["rotMthreeMax"] = f3(rot.drift_M3_sd.max())
bl8 = ps[(ps.perturbation == "blur") & (ps.level == "8")].iloc[0]
macros["detBlurEight"] = pc(bl8.detect_rate)
macros["detMin"] = pc(ps.detect_rate.min())

# ---------------------------------------------------------------- pose (Tab. 5)
st = T("pose_strata.csv")
body = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n$|$yaw$|$ (deg) & n & A$_{ref}$ & M1 & "
        "$|$err$|$ M3 & NME\\\\\n\\midrule\n")
for _, r in st.iterrows():
    body += (f"{r.yaw_bin.replace('>=', '$\\geq$')} & {int(r.n)} & {f2(r.A_ref_mean)} & {f2(r.M1_mean)} & "
             f"{f2(r.err_M3)} & {f2(r.nme_inner)}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"
write("pose.tex", body)
pcf = T("pose_confound.csv").set_index("subset")
macros.update(rhoYawRef=f2(pcf.loc["all", "rho(A_ref,|yaw|)"]),
              rhoYawRefAcc=f2(pcf.loc["M3 accept", "rho(A_ref,|yaw|)"]),
              rhoYawMthree=f2(pcf.loc["all", "rho(M3,|yaw|)"]),
              yawLow=f2(st.iloc[0].A_ref_mean), yawHigh=f2(st.iloc[-1].A_ref_mean))

# ---------------------------------------------------------------- risk coverage + gate
nm = T("risk_coverage_nme.csv").set_index("signal")
rs = T("risk_coverage_summary.csv").set_index("signal")
op = T("gate_operating_point.csv")
o3 = op[op.variant == "M3"].set_index("status")
macros.update(
    rhoMirrorNme=f2(nm.loc["mirror inconsistency", "spearman_vs_nme"]),
    aurcRandom=f2(nm.loc["random", "AURC_nme"]), aurcMirror=f2(nm.loc["mirror inconsistency", "AURC_nme"]),
    aurcGateTwo=f2(nm.loc["M2 gate risk", "AURC_nme"]), aurcOracle=f2(nm.loc["oracle (true NME)", "AURC_nme"]),
    gapClosed=f"{100 * (nm.loc['random', 'AURC_nme'] - nm.loc['mirror inconsistency', 'AURC_nme']) / (nm.loc['random', 'AURC_nme'] - nm.loc['oracle (true NME)', 'AURC_nme']):.0f}",
    aurcAsymRandom=f3(rs.loc["random", "AURC"]),
    aurcAsymMirror=f3(rs.loc["mirror inconsistency only", "AURC"]),
    aurcAsymGate=f3(rs.loc["M2 gate risk (size/blur/exposure/pose)", "AURC"]),
    rhoMirrorErr=f2(rs.loc["mirror inconsistency only", "spearman_risk_vs_err"]),
    pAccept=f"{o3.loc['accept', 'pct']:.1f}", pCaution=f"{o3.loc['caution', 'pct']:.1f}",
    pReject=f"{o3.loc['reject', 'pct']:.1f}",
    nmeAccept=f2(o3.loc["accept", "mean_nme_inner"]), nmeCaution=f2(o3.loc["caution", "mean_nme_inner"]),
)
reasons = T("gate_reasons.csv")
body = "\\begin{tabular}{@{}lcc@{}}\n\\toprule\nSignal & caution & reject\\\\\n\\midrule\n"
nice_sig = {"abs_yaw": "$|$yaw$|$", "abs_pitch": "$|$pitch$|$", "iod_px": "face size (IOD)",
            "mirror_incons": "mirror inconsistency $\\delta$", "brightness_hi": "over-exposure",
            "brightness_lo": "under-exposure", "sharpness": "sharpness", "n_faces": "several faces",
            "clipped": "clipped pixels"}
for sig in reasons.signal.unique():
    c = reasons[(reasons.signal == sig) & (reasons.level == "caution")].n_images.sum()
    r = reasons[(reasons.signal == sig) & (reasons.level == "reject")].n_images.sum()
    body += f"{nice_sig.get(sig, sig)} & {c} & {r}\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("reasons.tex", body)

rv = T("region_validity_occlusion.csv").set_index("occluded")
macros.update(occEye=pc(rv.loc["one eye", "occluded_region_flagged"]),
              occNose=pc(rv.loc["nose", "occluded_region_flagged"]),
              occNoseOther=pc(rv.loc["nose", "other_regions_flagged"]),
              occMouth=pc(rv.loc["mouth", "occluded_region_flagged"]),
              occMouthOther=pc(rv.loc["mouth", "other_regions_flagged"]))

# ---------------------------------------------------------------- gate thresholds
cfg = json.load(open(os.path.join(CONFIG_DIR, "gate_thresholds.json")))
th = cfg["thresholds"]
body = "\\begin{tabular}{@{}llll@{}}\n\\toprule\nSignal & caution & reject & source\\\\\n\\midrule\n"
src = {"iod_px": "a priori", "abs_yaw": "a priori", "abs_pitch": "a priori"}
units = {"iod_px": " px", "abs_yaw": "$^\\circ$", "abs_pitch": "$^\\circ$"}
for k, r in th.items():
    sym = "$<$" if r["direction"] == "min" else "$>$"
    s_ = src.get(k, "train pct." if k not in ("brightness_lo", "brightness_hi", "clipped") else "train pct. / a priori")
    u = units.get(k, "")
    body += f"{nice_sig.get(k, k)} & {sym} {r['caution']:.3g}{u} & {sym} {r['reject']:.3g}{u} & {s_}\\\\\n"
body += (f"per-region $\\delta_r$ (validity) & \\multicolumn{{2}}{{l}}{{unreliable if $>$ "
         f"{cfg['region_mirror_limit']:.3g}}} & train pct.\\\\\n\\bottomrule\n\\end{{tabular}}\n")
body = body.replace("sharpness &", "sharpness ($\\log_{10}$ var. Lap.) &").replace("over-exposure &", "mean luma (high) &").replace("under-exposure &", "mean luma (low) &")
write("thresholds.tex", body)
macros["alphaIndex"] = f"{cfg['index_alpha_per_pct']:.3f}"
macros["trainMedian"] = f2(1 / cfg["index_alpha_per_pct"])

clean = T("test_clean_rows.csv")
macros["runtime"] = f"{1000 * clean.detect_s.mean():.0f}"
macros["nTrainOk"] = str(int(T("train_clean_rows.csv").ok.sum()))

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write("% Auto-generated by experiments/make_tables.py -- do not edit by hand\n")
    for k, v in macros.items():
        assert k.isalpha(), k
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and tables to {OUT}")
for k, v in macros.items():
    print(f"  {k} = {v}")

# ---------------------------------------------------------------- code appendix
# LaTeX `listings` cannot typeset some Unicode symbols, so ASCII copies of the source files
# are written to paper/code/ for the appendix. The repository files remain the originals.
import glob  # noqa: E402

ASCII = {"✅": "[OK]", "⚠": "[!]", "️": "", "⛔": "[X]", "σ": "sigma",
         "…": "...", "±": "+/-", "−": "-", "α": "alpha", "·": "*",
         "°": " deg", "ρ": "rho"}
code_dir = os.path.join(ROOT, "paper", "code")
srcs = (glob.glob(os.path.join(ROOT, "src", "fsm", "*.py")) + [os.path.join(ROOT, "app.py")]
        + glob.glob(os.path.join(ROOT, "experiments", "*.py"))
        + glob.glob(os.path.join(ROOT, "tests", "*.py")))
for path in srcs:
    rel = os.path.relpath(path, ROOT)
    txt = open(path, encoding="utf-8").read()
    for k, v in ASCII.items():
        txt = txt.replace(k, v)
    assert all(ord(ch) < 128 for ch in txt), f"non-ASCII left in {rel}"
    dst = os.path.join(code_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(txt)
print("ASCII code copies written to", code_dir)

# ---------------------------------------------------------------- figures for the paper
# Copy the generated figures next to the LaTeX source so paper/ is self-contained
# (e.g. it can be uploaded to Overleaf as-is).
import shutil  # noqa: E402

fig_dst = os.path.join(ROOT, "paper", "figures")
os.makedirs(fig_dst, exist_ok=True)
for pdf in glob.glob(os.path.join(ROOT, "results", "figures", "*.pdf")):
    shutil.copy(pdf, fig_dst)
print("figures copied to", fig_dst)
