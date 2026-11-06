# Demo script and viva preparation (Nov 7–8)

## Before the session (10 minutes, on the demo laptop)

```bash
git clone https://github.com/pruthav23/Face-Symmetry-Measurement
cd Face-Symmetry-Measurement
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q tests                                       # should print: 15 passed
streamlit run app.py                                  # opens http://localhost:8501
```

* Everything runs offline on CPU once installed (the model file is in `models/`).
* Prepare **3 photos of a team member who has agreed to be shown**:
  1. frontal, neutral, even light (should be ACCEPT);
  2. head clearly turned (~25–30°) (should be REJECT on |yaw|);
  3. optional: slightly turned (~12°) (should be CAUTION).
* Keep `results/figures/robustness.png` and `paper/main.pdf` open in another window.

## Live demo (about 4 minutes)

| Step | Action | What to say |
|---|---|---|
| 1 | Upload the frontal photo | “Green ACCEPT. A is the mean distance of the inner-face landmarks from the nearest perfectly mirror-symmetric shape, as % of the eye-corner distance. The ± band is the mirror test: how much the score changes when we analyse the flipped image.” |
| 2 | Open the **Overlays** tab | “Mesh, the *fitted* symmetry axis with fixed bilateral pairs, the residual between each point and its partner's reflection, and the per-point heatmap.” |
| 3 | **Regions** tab | “Five regions; a region is greyed out if its landmarks disagree between the two passes. The jaw is excluded from A because it is a pose-dependent silhouette.” |
| 4 | Sidebar: **blur σ → 6–8** | “The gate moves to caution, then reject, and gives the reason: sharpness. The score is withheld.” Reset blur. |
| 5 | Sidebar: **rotation → 25°** | “The score barely changes, because the axis is fitted, not assumed vertical. A vertical-midline method without alignment drifts by up to 3.7 SD (robustness plot).” Reset. |
| 6 | Sidebar: **brightness → 0.3** | “Under-exposure: caution or reject, with the reason.” Reset. |
| 7 | Upload the **turned-head** photo | “REJECT on |yaw|. Note the grey caption: the baseline would still have reported a number. That is exactly the failure we prevent; on LFPW, yaw correlates 0.91 with asymmetry even from expert landmarks.” |
| 8 | **Export** tab → Download JSON | “Every result is exportable; the experiment scripts produce all tables and plots from such rows.” Show the robustness figure. |

Fallbacks: if the webcam is blocked, use upload. If the app fails, show `pytest` passing and the figures in `results/figures/`.

## Who presents what (suggested; adjust to actual roles)

| Slides | Presenter |
|---|---|
| 1–5 Problem, motivation, related work, contributions | Pankaj Kumar |
| 6–9 Method: architecture, measurement, mirror test, gate | Prutha Vinay, Ashish Porwal |
| 10–14 Data, protocol, results, ablation | Ankit Yadav, Ashish Porwal |
| 15 Failure cases | Pankaj Kumar |
| 16 Live demo | Mahi Mittal |
| 17 Conclusion and questions | Prutha Vinay |

## Likely viva questions and short answers

**What exactly is novel?** Not landmarks, regions or the UI; those exist. (1) Fixed anatomical pairs from MediaPipe's exactly symmetric canonical mesh plus a *fitted* symmetry axis: the KD-tree matching in prior tools picks the wrong partner 58.3% of the time, and fixing that raises agreement with expert landmarks from 0.823 to 0.931–0.947. (2) The mirror test: flip test-time augmentation exists in pose estimation for accuracy; we use the fact that flipping cannot change asymmetry, so disagreement is a label-free reliability signal (ρ = 0.59 with true landmark error), and averaging cancels left/right detector bias (−29% drift). (3) A gate calibrated on train only and evaluated as selective measurement (silent failures 8.1% → 3.6%). (4) A protocol that evaluates without symmetry labels.

**How do you evaluate without ground-truth symmetry?** We compute the identical measure from expert 68-point annotations (A_ref) and measure agreement (Spearman, MAE). We also use controlled perturbations where the face is unchanged, so any score change is error. A_ref is a reference, not truth: it contains annotator noise.

**Why fit the axis instead of using the eye-corner midline?** A vertical line through two landmarks turns head roll, or an error in those two points, into apparent asymmetry. The least-squares reflection axis is rotation-, translation- and scale-invariant (unit-tested) and needs no re-detection.

**Why convert 478 points to 68?** Expert annotations exist for 68 points (300-W), so we can validate. The 478→68 map is fitted on the train split and constrained to be mirror-consistent, so the mapping cannot create asymmetry. It improves NME from 4.23% to 3.95% (inner points).

**Why normalise by the eye distance (IOD)?** It is the standard 300-W normaliser, stable and pose-robust compared with face width, which depends on the jaw silhouette.

**How were thresholds chosen? Did you tune on test?** No. Pose and size limits are a priori (yaw 10°/20°, pitch 15°/25°, IOD 60/30 px). Image statistics use percentiles of the 810 training faces. The test split is used only for evaluation.

**Why MediaPipe?** Pretrained, CPU real time (46 ms per image for three passes), fixed mesh topology (needed for fixed pairs), and a head-pose matrix. Its drawbacks: no per-landmark confidence (hence the mirror test) and a symmetric shape prior (hence the 12% shrinkage we report).

**Why not train a network?** There are no symmetry labels to train on, and explicit geometry is interpretable and testable. The project's question is measurement reliability, not landmark accuracy.

**Why LFPW and not WFLW?** WFLW's images are on Google Drive, which our compute environment could not reach. LFPW has 300-W expert landmarks and a clean train/test split; the code reads any 68-point `.pts` dataset.

**Can MediaPipe have been trained on LFPW?** Its training data is not public, so we do not claim an independent landmark benchmark. We use NME only to compare mappings and gate signals.

**What fails?** An occluded eye: the detector invents it consistently in both passes (flagged only 0.9%). The detector shrinks asymmetry (−12% overall, −27% mouth). On clean photos the gate does not predict small errors. In-the-wild coverage is low (42% accepted).

**Is this a medical or beauty score?** No. It is a 2D geometric indicator. The app says so, and the symmetry index S is only a monotone re-expression of A, not a validated scale.

**Did you copy code?** No. The audited repositories have no licence, so the baseline was re-implemented from its documented method. MediaPipe's model and canonical mesh are Apache-2.0.
