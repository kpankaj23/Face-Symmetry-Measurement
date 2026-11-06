"""
app.py -- Streamlit demo of the quality-aware facial symmetry measurement system.

    streamlit run app.py

The UI only *calls* the tested engine in src/fsm; no geometry is implemented here.
"""
import io
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fsm import viz  # noqa: E402
from fsm.mesh import CONFIG_DIR, IBUG68_REGIONS  # noqa: E402
from fsm.pipeline import measure, signals_of, variant_outputs  # noqa: E402
from fsm.quality import EXPLAIN, load_gate  # noqa: E402

st.set_page_config(page_title="Face Symmetry Measurement", layout="wide")

with open(os.path.join(CONFIG_DIR, "gate_thresholds.json")) as f:
    CFG = json.load(f)
ALPHA = CFG.get("index_alpha_per_pct", 0.5)
STATUS_TEXT = {
    "accept": ("[OK] ACCEPT", "Image meets the measurement conditions. The score below is reported."),
    "caution": ("[!] CAUTION", "The score is shown, but at least one condition is outside the "
                "calibrated range. Interpret with care."),
    "reject": ("[X] REJECT", "The image does not meet the measurement conditions, so no score "
               "is reported. Fix the issues listed and try again."),
}


# ------------------------------------------------------------------ helpers
def to_bgr(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def limit_size(img, max_side=1280):
    """Large phone photos are downscaled for speed (does not change the normalised score)."""
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def perturb(img, brightness, blur, angle, occlude):
    """Optional controlled perturbations so the demo can show the gate reacting live."""
    out = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if angle != 0:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if occlude != "none":
        h, w = out.shape[:2]
        box = {"left half of image": (0, 0, w // 2, h),
               "upper-left quarter": (0, 0, w // 2, h // 2),
               "lower third": (0, 2 * h // 3, w, h)}[occlude]
        cv2.rectangle(out, box[:2], box[2:], (128, 128, 128), -1)
    return out


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def index_of(A_pct):
    return 100.0 / (1.0 + ALPHA * A_pct)


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Input")
source = st.sidebar.radio("Image source", ["Upload a photo", "Webcam snapshot"])
st.sidebar.header("Stress-test the input (demo)")
st.sidebar.caption("Apply a controlled change and watch the quality gate respond.")
b = st.sidebar.slider("Brightness factor", 0.2, 2.5, 1.0, 0.05)
bl = st.sidebar.slider("Gaussian blur sigma (px)", 0.0, 15.0, 0.0, 0.5)
ang = st.sidebar.slider("In-plane rotation (deg)", -40, 40, 0, 1)
occ = st.sidebar.selectbox("Occlusion", ["none", "upper-left quarter", "left half of image",
                                         "lower third"])
show_base = st.sidebar.checkbox("Compare with baseline (M0)", value=True)

# ------------------------------------------------------------------ header
st.title("Face Symmetry Measurement")
st.markdown(
    "Quality-aware, region-based **2D geometric** facial asymmetry. The system checks whether "
    "the photo is suitable *before* reporting a number, and verifies its own landmarks with a "
    "mirror test.  \n"
    "<small>This is an image-based geometric indicator only: not a medical assessment and not "
    "a measure of attractiveness. Photos are processed in memory and not stored.</small>",
    unsafe_allow_html=True)

file_bytes = None
if source == "Upload a photo":
    up = st.file_uploader("Upload one clear, frontal face photo (JPG/PNG)",
                          type=["jpg", "jpeg", "png", "bmp", "webp"])
    file_bytes = up.getvalue() if up else None
else:
    cam = st.camera_input("Look straight at the camera, neutral expression")
    file_bytes = cam.getvalue() if cam else None

if file_bytes is None:
    st.info("Tips for a valid measurement: face the camera, keep the head level, even "
            "lighting, neutral expression, nothing covering the face.")
    st.stop()

img0 = to_bgr(file_bytes)
if img0 is None:
    st.error("Could not read this file as an image.")
    st.stop()
img = perturb(limit_size(img0), b, bl, ang, occ)

with st.spinner("Detecting landmarks (original + mirrored image)..."):
    m = measure(img, run_baseline=show_base, run_mirror=True)

if not m["ok"]:
    st.error(f"[X] REJECT: {m['failure']}. Please upload a photo with one clearly visible face.")
    st.image(rgb(img), caption="input", width=360)
    st.stop()

v = variant_outputs(m)
out = v["M3"]
status = out["status"]
label, expl = STATUS_TEXT[status]

# ------------------------------------------------------------------ verdict + numbers
c1, c2 = st.columns([1.1, 1])
with c1:
    {"accept": st.success, "caution": st.warning, "reject": st.error}[status](f"**{label}**: {expl}")
    if out.get("reasons"):
        for r in out["reasons"]:
            st.markdown(f"- **{r['level']}**: {r['why']} "
                        f"(`{r['signal']}` = {r['value']:.3g})")
with c2:
    A = 100 * out["score"] if np.isfinite(out["score"]) else np.nan
    band = 100 * out.get("stability_band", np.nan)
    show_score = status != "reject" or st.checkbox("Show the (unreliable) score anyway")
    k1, k2, k3 = st.columns(3)
    if show_score and np.isfinite(A):
        k1.metric("Asymmetry A (% IOD)", f"{A:.2f}", help="Mean distance of the inner-face "
                  "landmarks from the closest perfectly mirror-symmetric shape, as a percentage "
                  "of the distance between the outer eye corners. Lower = more symmetric.")
        k2.metric("Mirror stability +/- (% IOD)", f"{band:.2f}", help="|A(original) - A(mirrored "
                  "image)|. A perfectly consistent detector gives 0.")
        k3.metric("Symmetry index (0-100)", f"{index_of(A):.0f}", help="S = 100/(1+alpha*A); alpha set so "
                  "the median LFPW-train face scores 50. A readable re-expression of A, not a "
                  "validated clinical or aesthetic score.")
    else:
        k1.metric("Asymmetry A", "withheld")
    if show_base and v["M0"]["score"] == v["M0"]["score"]:
        st.caption(f"Baseline M0 (KD-tree, vertical midline): {100 * v['M0']['score']:.2f} % of "
                   "face width, reported without any quality check.")

# ------------------------------------------------------------------ visualisations
mir = m["mirror"]
res = mir["avg"] if mir else m["single"]
P = mir["lm68_avg"] if mir else m["lm68"]
valid = out.get("region_valid", {})
tabs = st.tabs(["Overlays", "Regions", "Quality signals", "Export"])


def zoom(im, pts=m["lm478"][:468], margin=0.25):
    """Crop a drawn overlay to the face so the landmarks are visible."""
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    h, w = im.shape[:2]
    return im[int(max(0, y0 - my)):int(min(h, y1 + my)), int(max(0, x0 - mx)):int(min(w, x1 + mx))]


with tabs[0]:
    a, b2, c, d = st.columns(4)
    a.image(rgb(zoom(viz.draw_mesh(img, m["lm478"][:468]))), caption="478-point mesh",
            use_container_width=True)
    b2.image(rgb(zoom(viz.draw_axis(viz.draw_pairs(img, P), res, P))),
             caption="fitted symmetry axis + bilateral pairs", use_container_width=True)
    c.image(rgb(zoom(viz.draw_mirrored(img, res, P))),
            caption="green: landmark, red: reflection of its partner", use_container_width=True)
    d.image(rgb(zoom(viz.draw_heatmap(img, res, P, valid=valid))),
            caption="per-point asymmetry (bright = larger; grey = unreliable region)",
            use_container_width=True)

with tabs[1]:
    rows = []
    for k in IBUG68_REGIONS:
        ok_r = valid.get(k, True)
        rows.append({"region": k, "asymmetry (% IOD)": round(100 * res.region_scores[k], 2),
                     "mirror inconsistency (% IOD)":
                         round(100 * mir["region_incons"][k], 2) if mir else None,
                     "status": "reliable" if ok_r else "unreliable (detector inconsistent)"})
    rdf = pd.DataFrame(rows)
    st.dataframe(rdf, hide_index=True, use_container_width=True)
    inner = rdf[(rdf.region != "Jaw") & (rdf.status == "reliable")]
    if len(inner):
        top = inner.sort_values("asymmetry (% IOD)", ascending=False).iloc[0]
        st.markdown(f"Largest measured inner-face asymmetry: **{top['region']}** "
                    f"({top['asymmetry (% IOD)']:.2f} % IOD).")
    st.bar_chart(rdf.set_index("region")["asymmetry (% IOD)"], horizontal=True)
    st.caption("The jaw contour is a silhouette whose landmark positions shift with head pose; "
               "its values are larger and less reliable, so it is excluded from the global score.")

with tabs[2]:
    gate = load_gate()
    sig = signals_of(m, use_mirror=True)
    sig["brightness_lo"] = sig["brightness_hi"] = sig["brightness"]
    rows = []
    for name, rule in gate.items():
        rows.append({"signal": name, "value": round(float(sig[name]), 4),
                     "caution if": f"{'<' if rule['direction'] == 'min' else '>'} {rule['caution']:.3g}",
                     "reject if": f"{'<' if rule['direction'] == 'min' else '>'} {rule['reject']:.3g}",
                     "meaning": EXPLAIN.get(name, "")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    p = m["pose"]
    st.caption(f"Pose estimate: yaw {p['yaw']:.1f} deg, pitch {p['pitch']:.1f} deg, roll {p['roll']:.1f} deg "
               "(roll is handled by the fitted symmetry axis and is not gated).")

with tabs[3]:
    record = {
        "status": status, "reasons": out.get("reasons", []),
        "asymmetry_pct_iod": None if status == "reject" else A,
        "mirror_stability_pct_iod": band, "symmetry_index": None if status == "reject" else index_of(A),
        "regions_pct_iod": {k: 100 * x for k, x in res.region_scores.items()},
        "region_valid": valid, "pose_deg": m["pose"], "quality": m["quality"],
        "iod_px": m["iod_px"], "baseline_M0_pct_face_width":
            100 * v["M0"]["score"] if show_base else None,
        "landmarks68_xy": P.round(2).tolist(),
    }
    st.download_button("Download JSON", json.dumps(record, indent=1, default=float),
                       "symmetry_result.json", "application/json")
    flat = {k: val for k, val in record.items() if not isinstance(val, (dict, list))}
    flat.update({f"region_{k}": x for k, x in record["regions_pct_iod"].items()})
    buf = io.StringIO()
    pd.DataFrame([flat]).to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(), "symmetry_result.csv", "text/csv")
