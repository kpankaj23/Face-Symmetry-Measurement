"""
Generate every paper figure from the saved CSVs / cached detections (no manual edits).
    python experiments/make_figures.py
Writes results/figures/*.pdf and *.png
"""
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGS, TABLES, load_cache  # noqa: E402
from fsm import mesh as MS, viz  # noqa: E402
from fsm.data import crop_to_annotation, list_split, read_pts  # noqa: E402
from fsm.pipeline import PERM478, measure_from_raw, variant_outputs, raw_detect  # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode) + status palette
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"accept": "#0ca30c", "caution": "#fab219", "reject": "#d03b3b"}
STATUS_MARK = {"accept": "o", "caution": "^", "reject": "X"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": int(os.environ.get("FIG_DPI", 220)),
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


test_rows = pd.read_csv(os.path.join(TABLES, "test_clean_rows.csv"))
cache = {r["image"]: r for r in load_cache("test")}
ok = test_rows[test_rows.ok].copy()


def canvas(img):
    """
    Privacy rule for figures: dataset photographs (which include identifiable people,
    some of them public figures) are never reproduced in the paper. Geometry is drawn on
    a blank canvas of the same size instead.
    """
    return np.full_like(img, 248)


DARK = (60, 60, 60)
BLUE = (214, 120, 42)    # C[0] in BGR


def load_test_image(name):
    d = dict(list_split("test"))
    path = [p for p in d if os.path.basename(p) == name][0]
    return crop_to_annotation(cv2.imread(path), read_pts(d[path]))


# ------------------------------------------------------------------ Fig: pipeline strip
def fig_pipeline(name):
    img, gt = load_test_image(name)
    m = measure_from_raw(cache[name])
    v = variant_outputs(m)
    P = m["mirror"]["lm68_avg"]
    res = m["mirror"]["avg"]
    bg = canvas(img)
    U = m["mirror"]["lm478_unflipped"][:468]
    both = viz.draw_mesh(viz.draw_mesh(bg, U, BLUE, radius=3), m["lm478"][:468], DARK, radius=1)
    panels = [
        ("(a) 478-pt mesh on I", viz.draw_mesh(bg, m["lm478"][:468], DARK)),
        ("(b) I + unflip(flip(I))", both),
        ("(c) mapped 68 points", viz.draw_regions(bg, P)),
        ("(d) fitted axis + pairs", viz.draw_axis(viz.draw_pairs(bg, P, color=BLUE), res, P)),
        ("(e) mirror residuals", viz.draw_mirrored(bg, res, P)),
        ("(f) per-point heatmap", viz.draw_heatmap(bg, res, P, valid=v["M3"]["region_valid"])),
    ]
    fig, ax = plt.subplots(1, 6, figsize=(7.0, 1.6))
    for a, (t, im) in zip(ax, panels):
        a.imshow(rgb(im))
        a.set_title(t, fontsize=7)
        a.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "pipeline_examples")
    return v


# ------------------------------------------------------------------ Fig: KD-tree vs fixed
def fig_correspondence(name):
    img, _ = load_test_image(name)
    m = measure_from_raw(cache[name])
    b = m["baseline"]
    La = b["lm478_aligned"]
    aligned = cv2.warpAffine(canvas(img), cv2.estimateAffinePartial2D(
        np.float32([m["lm478"][33], m["lm478"][263]]),
        np.float32([[600 * .33, 600 * .33], [600 * .67, 600 * .33]]))[0], (600, 600))
    # Reflected left points vs their KD-tree partners (draw left point -> partner)
    kd = viz.draw_kdtree_matches(aligned, La, b["pairs"], PERM478)
    fixed_pairs = np.stack([b["pairs"][:, 0], PERM478[b["pairs"][:, 0]]], 1)
    fx = viz.draw_kdtree_matches(aligned, La, fixed_pairs, PERM478)
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for a, im, t in zip(ax, [kd, fx], [f"KD-tree matching\n{100*b['cross_structure_rate']:.0f}% wrong partner (red)",
                                       "fixed canonical pairs\n0% wrong partner"]):
        a.imshow(rgb(im[60:560, 60:540]))
        a.set_title(t, fontsize=7)
        a.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "correspondence")


# ------------------------------------------------------------------ Fig: agreement scatter
def fig_agreement():
    ag = pd.read_csv(os.path.join(TABLES, "agreement_global.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.75))
    for a, col, lab, c in [(ax[0], "M0_score", "M0 baseline score (% face width)", C[1]),
                           (ax[1], "M3_score", "M3 score (% IOD)", C[0])]:
        a.scatter(ok["A_ref"], ok[col], s=9, color=c, edgecolor="white", linewidth=0.4)
        rho = ag.loc[ag.measure.str.startswith(col[:2]) & ~ag.measure.str.contains("without"),
                     "spearman"].iloc[0]
        a.set_title(f"Spearman rho = {rho:.2f}", fontsize=7.5)
        a.set_xlabel("A$_{ref}$ (% IOD)")
        a.set_ylabel(lab, fontsize=7)
        if col == "M3_score":
            lim = [0, ok[["A_ref", col]].max().max() * 1.05]
            a.plot(lim, lim, color=MUTED, lw=1, ls="--")
            a.text(lim[1] * 0.97, lim[1] * 0.05, "dashed: y = x", color=MUTED, ha="right", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    save(fig, "agreement")


# ------------------------------------------------------------------ Fig: pose confound
def fig_pose():
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for st in ("accept", "caution", "reject"):
        s = ok[ok.M3_status == st]
        ax.scatter(s["sig_abs_yaw"], s["A_ref"], s=12, marker=STATUS_MARK[st], color=STATUS[st],
                   edgecolor="white", linewidth=0.4, label=f"{st} (n={len(s)})")
    for x in (10, 20):
        ax.axvline(x, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("|yaw| estimated by MediaPipe (deg)")
    ax.set_ylabel("A$_{ref}$ from annotations (% IOD)")
    ax.legend(loc="upper left", fontsize=6.5, title="M3 gate decision", title_fontsize=6.5)
    save(fig, "pose_confound")


# ------------------------------------------------------------------ Fig: risk-coverage
def fig_risk_coverage():
    rc = pd.read_csv(os.path.join(TABLES, "risk_coverage.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.8))
    sets = [("NME", [("NME|random", "random", C[7]), ("NME|M2 gate risk", "M2 gate", C[1]),
                     ("NME|mirror inconsistency", "mirror test", C[0]),
                     ("NME|oracle (true NME)", "oracle", MUTED)],
             "landmark NME (% IOD)"),
            ("asym", [("random", "random", C[7]), ("M2 gate risk (size/blur/exposure/pose)", "M2 gate", C[1]),
                      ("mirror inconsistency only", "mirror test", C[0])],
             "|A$_{det}$ - A$_{ref}$| (% IOD)")]
    for a, (_, curves, ylab) in zip(ax, sets):
        for key, lab, c in curves:
            s = rc[rc.signal == key]
            s = s[s.coverage >= 0.2]
            a.plot(100 * s.coverage, s.mean_abs_err, color=c, label=lab,
                   ls="--" if lab == "oracle" else "-", lw=1.6)
        a.set_xlabel("coverage (% images kept)")
        a.set_ylabel("mean " + ylab, fontsize=7)
    ax[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout(pad=0.3)
    save(fig, "risk_coverage")


# ------------------------------------------------------------------ Fig: robustness
def fig_robustness():
    p = os.path.join(TABLES, "perturbation_summary.csv")
    if not os.path.exists(p):
        print("skip robustness (run summarize_perturbations.py first)")
        return
    s = pd.read_csv(p)
    kinds = ["brightness", "blur", "rotation", "downscale", "occlusion"]
    xl = {"brightness": "intensity factor", "blur": "sigma (% IOD)", "rotation": "angle (deg)",
          "downscale": "scale factor", "occlusion": "occluded part"}
    fig, ax = plt.subplots(2, 5, figsize=(7.0, 3.0))
    for j, k in enumerate(kinds):
        d = s[s.perturbation == k].copy()
        if k == "occlusion":
            x = np.arange(len(d))
        else:
            d["lv"] = d.level.astype(float)
            d = d.sort_values("lv")
            x = d.lv.values
        for col, lab, c in [("drift_M0_sd", "M0 baseline", C[1]), ("drift_M1_sd", "M1 fixed+axis", C[2]),
                            ("drift_M3_sd", "M3 mirror-avg", C[0])]:
            ax[0, j].plot(x, d[col], marker="o", ms=3, color=c, label=lab, lw=1.5)
        if k == "rotation":
            ax[0, j].plot(x, d["drift_M0noalign_sd"], marker="s", ms=3, color=C[1], ls="--",
                          lw=1.2, label="M0 w/o alignment")
            ax[0, j].set_yscale("log")
        for col, lab, c in [("flag_M2", "M2 gate", C[1]), ("flag_M3", "M3 gate", C[0])]:
            ax[1, j].plot(x, 100 * d[col], marker="o", ms=3, color=c, label=lab, lw=1.5)
        ax[0, j].set_title(k, fontsize=8)
        ax[1, j].set_xlabel(xl[k], fontsize=7)
        if k == "occlusion":
            for a in ax[:, j]:
                a.set_xticks(x)
                a.set_xticklabels(d.level, fontsize=6.5)
        if k == "downscale":
            for a in ax[:, j]:
                a.invert_xaxis()
    for j in range(5):
        ax[1, j].set_ylim(0, 105)
        if j:
            ax[1, j].set_yticklabels([])
    ax[0, 0].set_ylabel("median score drift\n(SD units)", fontsize=7)
    ax[1, 0].set_ylabel("% flagged\n(caution/reject)", fontsize=7)
    h1, l1 = ax[0, 2].get_legend_handles_labels()
    h2, l2 = ax[1, 0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
    fig.legend(h1 + h2, ["drift: " + t for t in l1] + ["flagged: " + t for t in l2],
               loc="upper center", ncol=6, fontsize=6.2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "robustness")


# ------------------------------------------------------------------ Fig: gate examples
def fig_gate_examples(names_by_status):
    fig, ax = plt.subplots(1, len(names_by_status), figsize=(3.4, 1.55))
    for a, (st, name) in zip(ax, names_by_status):
        img, _ = load_test_image(name)
        m = measure_from_raw(cache[name])
        v = variant_outputs(m)["M3"]
        P = m["mirror"]["lm68_avg"]
        im = viz.draw_axis(viz.draw_mesh(canvas(img), P, DARK, radius=3), m["mirror"]["avg"], P)
        im = viz.status_banner(im, v["status"])
        a.imshow(rgb(im))
        why = ", ".join(sorted({{"abs_yaw": "|yaw|", "abs_pitch": "|pitch|"}.get(r["signal"], r["signal"].replace("_", " ")) for r in v["reasons"]
                                if r["level"] == v["status"]})) or "all checks pass"
        a.set_title(f"A = {100*v['score']:.1f}% IOD\n{why}", fontsize=6)
        a.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "gate_examples")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    # Example selection rule (reproducible): the accepted image with median score for the
    # pipeline figure; for the gate figure the median-yaw image of each M3 decision class.
    acc = ok[ok.M3_status == "accept"].sort_values("M3_score")
    ex = acc.iloc[len(acc) // 2]["image"]
    fig_pipeline(ex)
    fig_correspondence(ex)
    fig_agreement()
    fig_pose()
    fig_risk_coverage()
    fig_robustness()
    picks = []
    for st in ("accept", "caution", "reject"):
        s = ok[ok.M3_status == st].sort_values("sig_abs_yaw")
        picks.append((st, s.iloc[len(s) // 2]["image"]))
    fig_gate_examples(picks)
    print("examples:", ex, picks)
