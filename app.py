"""
Microplastic Detector — Complete Fixed Version
"""

import os
import io
import time
import math
import tempfile
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import streamlit as st

YOLO = None
YOLO_IMPORT_ERROR = ""
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    YOLO_IMPORT_ERROR = str(e)

st.set_page_config(page_title="Microplastic Detector", layout="wide", page_icon="🧫")

APP_SUBTITLE = "📸 Upload images, detect microplastics using your YOLOv8 model, and get easy-to-understand visualizations."

def color_map_for_values(n: int, cmap_name: str = "viridis") -> List[str]:
    cmap = cm.get_cmap(cmap_name)
    return [cm.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def generate_feedback(df: pd.DataFrame, filename: str, conf_threshold: float) -> Dict[str, Any]:
    if df.empty:
        return {
            "severity": "safe",
            "level_label": "✅ CLEAR",
            "level_color": "#22c55e",
            "level_bg": "rgba(34,197,94,0.12)",
            "level_border": "#22c55e",
            "summary": "No microplastics detected in this sample.",
            "detail": "The analysis found no microplastic particles above the configured confidence threshold.",
            "warnings": [],
            "health_info": [],
            "recommendations": [
                "✔ Continue monitoring with periodic sample checks.",
                "✔ Consider lowering the confidence threshold if trace plastics are suspected.",
                "✔ Verify sample preparation methods for accuracy.",
            ],
            "count": 0,
            "avg_conf": 0.0,
        }

    count = len(df)
    avg_conf = float(df["confidence"].mean())
    max_conf = float(df["confidence"].max())
    unique_types = df["class"].nunique() if "class" in df.columns else 0

    if count <= 3:
        severity, level_label = "low", "🟡 LOW RISK"
        level_color, level_bg, level_border = "#f59e0b", "rgba(245,158,11,0.12)", "#f59e0b"
    elif count <= 10:
        severity, level_label = "moderate", "🟠 MODERATE RISK"
        level_color, level_bg, level_border = "#f97316", "rgba(249,115,22,0.12)", "#f97316"
    elif count <= 25:
        severity, level_label = "high", "🔴 HIGH RISK"
        level_color, level_bg, level_border = "#ef4444", "rgba(239,68,68,0.14)", "#ef4444"
    else:
        severity, level_label = "critical", "🚨 CRITICAL CONTAMINATION"
        level_color, level_bg, level_border = "#dc2626", "rgba(220,38,38,0.18)", "#dc2626"

    warnings = []
    if count > 0:
        warnings.append(f"⚠️ <b>{count} microplastic particle(s)</b> detected in <i>{filename}</i>.")
    if avg_conf >= 0.80:
        warnings.append(f"⚠️ High detection confidence (<b>{avg_conf:.1%}</b>) — detections are likely accurate.")
    if max_conf >= 0.90:
        warnings.append(f"⚠️ At least one particle detected with very high confidence (<b>{max_conf:.1%}</b>).")
    if unique_types > 1:
        warnings.append(f"⚠️ <b>{unique_types} distinct microplastic types</b> identified — mixed contamination.")
    if count > 20:
        warnings.append("⚠️ Extremely high particle density — immediate remediation strongly advised.")

    health_map = {
        "low": [
            "🧬 Trace microplastics present at low concentrations.",
            "🫁 Low-level exposure risk; continued monitoring recommended.",
            "💧 Water sources showing minimal contamination should still be filtered.",
        ],
        "moderate": [
            "🧬 Moderate concentrations detected — potential health concern.",
            "🫁 Microplastics linked to inflammation and oxidative stress.",
            "💧 This sample should not be consumed without proper filtration.",
            "🐟 Aquatic life may be ingesting microplastics, entering the food chain.",
        ],
        "high": [
            "🧬 High microplastic load — significant contamination confirmed.",
            "🫁 High exposure linked to respiratory, endocrine, and cardiovascular concerns.",
            "💧 Immediate use of advanced purification (reverse osmosis) recommended.",
            "🐟 Ecosystem impact likely — aquatic species face serious risk.",
            "🏭 Identify and address upstream pollution sources immediately.",
        ],
        "critical": [
            "🚨 CRITICAL: Extremely high microplastic contamination detected.",
            "🧬 Direct contact with this sample poses serious health risks.",
            "🫁 Long-term exposure associated with organ damage and carcinogenic risk.",
            "💧 This sample is UNSAFE — do not consume without industrial-grade treatment.",
            "🐟 Complete ecological damage likely — report to environmental authorities.",
            "🏭 Emergency remediation protocols should be activated immediately.",
        ],
    }

    rec_map = {
        "low": [
            "📋 Document findings and schedule follow-up sampling within 30 days.",
            "🔬 Consider expanding sampling area for broader contamination mapping.",
            "🧪 Cross-validate results using spectroscopy or chemical analysis.",
        ],
        "moderate": [
            "📋 File a formal contamination report with your environmental team.",
            "🔬 Increase sampling frequency — collect from multiple zones.",
            "🧪 Use spectroscopic analysis (FTIR/Raman) to identify polymer types.",
            "🌊 Trace contamination to upstream or nearby industrial/waste sources.",
            "🛡️ Deploy filtration barriers if this is a water supply sample.",
        ],
        "high": [
            "🚨 Escalate findings to environmental regulatory authorities immediately.",
            "🔬 Conduct emergency multi-point sampling across the affected area.",
            "🧪 Perform full polymer composition analysis and size classification.",
            "🌊 Issue advisories to communities relying on this water source.",
            "🛡️ Implement emergency containment and filtration measures.",
        ],
        "critical": [
            "🚨 IMMEDIATE ACTION REQUIRED — notify authorities NOW.",
            "🚨 Restrict all access to the affected area or water source.",
            "🔬 Deploy multi-agency environmental response team.",
            "🧪 Conduct urgent toxicological analysis of samples.",
            "🌊 Issue public health advisory for affected region.",
            "🛡️ Activate industrial-grade remediation and containment protocols.",
        ],
    }

    return {
        "severity": severity,
        "level_label": level_label,
        "level_color": level_color,
        "level_bg": level_bg,
        "level_border": level_border,
        "summary": f"{count} microplastic particle(s) detected with average confidence of {avg_conf:.1%}.",
        "detail": f"Analysis identified {count} particle(s) across {unique_types} type(s). Highest confidence: {max_conf:.1%}.",
        "warnings": warnings,
        "health_info": health_map.get(severity, []),
        "recommendations": rec_map.get(severity, []),
        "count": count,
        "avg_conf": avg_conf,
    }


def render_feedback_panel(fb: Dict[str, Any], card_bg: str, text_color: str, primary_color: str):
    st.markdown(f"""
    <div style="background:{fb['level_bg']}; border:2px solid {fb['level_border']};
         border-radius:12px; padding:20px 28px; margin:12px 0;
         display:flex; align-items:center; gap:16px;
         box-shadow:0 0 20px {fb['level_border']}40;">
        <div style="font-size:36px;">{'🔬' if fb['severity'] == 'safe' else '⚠️'}</div>
        <div>
            <div style="font-size:22px; font-weight:800; color:{fb['level_color']}; letter-spacing:1px;">
                {fb['level_label']}
            </div>
            <div style="font-size:14px; color:{text_color}; margin-top:4px; opacity:0.9;">{fb['summary']}</div>
            <div style="font-size:13px; color:{text_color}; margin-top:2px; opacity:0.7;">{fb['detail']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if fb["severity"] == "safe":
        return

    fb_col1, fb_col2, fb_col3 = st.columns(3)
    with fb_col1:
        html = "".join(f"<div style='padding:8px 0; border-bottom:1px solid {fb['level_border']}30; font-size:13px; color:{text_color}; line-height:1.5;'>{w}</div>" for w in fb["warnings"])
        st.markdown(f"""<div style="background:{card_bg}; border:1px solid {fb['level_border']}60;
            border-top:3px solid {fb['level_border']}; border-radius:10px; padding:18px; min-height:220px;">
            <div style="font-size:15px; font-weight:700; color:{fb['level_color']}; margin-bottom:12px;">⚠️ Detection Warnings</div>{html}</div>""", unsafe_allow_html=True)
    with fb_col2:
        html = "".join(f"<div style='padding:8px 0; border-bottom:1px solid #ef444430; font-size:13px; color:{text_color}; line-height:1.5;'>{h}</div>" for h in fb["health_info"])
        st.markdown(f"""<div style="background:{card_bg}; border:1px solid #ef444460;
            border-top:3px solid #ef4444; border-radius:10px; padding:18px; min-height:220px;">
            <div style="font-size:15px; font-weight:700; color:#ef4444; margin-bottom:12px;">🏥 Health & Environmental Impact</div>{html}</div>""", unsafe_allow_html=True)
    with fb_col3:
        html = "".join(f"<div style='padding:8px 0; border-bottom:1px solid {primary_color}30; font-size:13px; color:{text_color}; line-height:1.5;'>{r}</div>" for r in fb["recommendations"])
        st.markdown(f"""<div style="background:{card_bg}; border:1px solid {primary_color}60;
            border-top:3px solid {primary_color}; border-radius:10px; padding:18px; min-height:220px;">
            <div style="font-size:15px; font-weight:700; color:{primary_color}; margin-bottom:12px;">📋 Recommended Actions</div>{html}</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── SESSION STATE ──────────────────────────────────────────────────────────────
for key, val in [("theme", "dark"), ("model_path", ""), ("uploaded_images", []), ("results", []), ("model_loaded_for", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")
theme_choice = st.sidebar.radio("Theme", ["🌑 Dark", "🌕 Light"], index=0)
st.session_state.theme = "dark" if theme_choice.startswith("🌑") else "light"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Setup")

uploaded_model_file = st.sidebar.file_uploader("Upload model file (.pt)", type=["pt"], help="Upload your YOLOv8 best.pt weights file directly")
if uploaded_model_file is not None:
    tmp_model_path = os.path.join(tempfile.gettempdir(), "uploaded_model.pt")
    with open(tmp_model_path, "wb") as f:
        f.write(uploaded_model_file.read())
    st.session_state.model_path = tmp_model_path
    st.sidebar.success("✅ Model file uploaded!")

st.sidebar.markdown("**Or paste absolute path:**")
manual_path = st.sidebar.text_input("Model path", value=st.session_state.model_path, placeholder="e.g. /home/user/best.pt")
if manual_path.strip():
    st.session_state.model_path = manual_path.strip()

if st.session_state.model_path:
    st.sidebar.info("📂 Path found ✅" if os.path.exists(st.session_state.model_path) else "⚠️ Path not found")

st.sidebar.markdown("---")
device_choice = st.sidebar.selectbox("Device", ["CPU", "GPU"], index=0)
device = "cpu" if device_choice == "CPU" else 0
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
max_display    = st.sidebar.slider("Max detections in plot", 5, 200, 60)
box_thickness  = st.sidebar.slider("Box thickness (px)", 1, 8, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:**")
st.sidebar.markdown("- Use clear close-up images of samples.")
st.sidebar.markdown("- Increase confidence threshold to reduce false positives.")
st.sidebar.markdown("- Use GPU if available for faster inference.")
st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Severity Levels:**")
st.sidebar.markdown("🟢 **CLEAR** — No detections")
st.sidebar.markdown("🟡 **LOW RISK** — 1–3 particles")
st.sidebar.markdown("🟠 **MODERATE** — Up to 10 particles")
st.sidebar.markdown("🔴 **HIGH RISK** — 11–25 particles")
st.sidebar.markdown("🚨 **CRITICAL** — 25+ particles")

# ── THEME COLORS ───────────────────────────────────────────────────────────────
if st.session_state.theme == "dark":
    PAGE_BG        = "linear-gradient(135deg, #0f1724 0%, #2b2143 40%, #6e5b7b 100%)"
    CARD_BG        = "rgba(255,255,255,0.04)"
    TEXT_COLOR     = "#e6eef6"
    PRIMARY_COLOR  = "#00f0ea"
    SECONDARY_COLOR= "#ff33cc"
    HIST_COLOR     = "#ff33cc"
    PIE_CMAP       = "viridis"
    CONF_CMAP      = "plasma"
    MT_FIG_FACE    = "#0f1724"
    WIDGET_BG      = "#1e1e2f"
    WIDGET_TEXT    = "#f5f5f5"
    WIDGET_ACCENT  = "#00f0ea"
else:
    PAGE_BG        = "linear-gradient(135deg, #e6f7ff 0%, #c7f0ff 50%, #a0e8ff 100%)"
    CARD_BG        = "rgba(255,255,255,0.85)"
    TEXT_COLOR     = "#07263b"
    PRIMARY_COLOR  = "#0288d1"
    SECONDARY_COLOR= "#ff7ab6"
    HIST_COLOR     = "#0077be"
    PIE_CMAP       = "Spectral"
    CONF_CMAP      = "coolwarm"
    MT_FIG_FACE    = "#e6f7ff"
    WIDGET_BG      = "#ffffff"
    WIDGET_TEXT    = "#07263b"
    WIDGET_ACCENT  = "#0288d1"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background: {PAGE_BG}; color: {TEXT_COLOR}; }}

  /* ── FILE UPLOADER — white box, black border, black text ── */
  [data-testid="stFileUploaderDropzone"] {{
      background: #ffffff !important;
      border: 2px solid #000000 !important;
      border-radius: 10px !important;
  }}
  [data-testid="stFileUploader"] button {{
      background: #ffffff !important;
      color: #000000 !important;
      border: 2px solid #000000 !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
  }}
  [data-testid="stFileUploader"] small,
  [data-testid="stFileUploader"] span,
  [data-testid="stFileUploader"] p {{
      color: #000000 !important;
      font-weight: 600 !important;
  }}

  /* ── Regular buttons ── */
  .stButton>button, .stDownloadButton>button {{
      background-color: {WIDGET_BG};
      color: {WIDGET_TEXT};
      border: 1px solid {WIDGET_ACCENT};
      border-radius: 8px;
      padding: 0.4em 1em;
      font-weight: 600;
      transition: all 0.3s ease;
  }}
  .stButton>button:hover, .stDownloadButton>button:hover {{
      background-color: {WIDGET_ACCENT};
      color: #000;
      transform: translateY(-2px);
  }}

  @keyframes pulse-border {{
    0%   {{ box-shadow: 0 0 0 0   rgba(220,38,38,0.5); }}
    70%  {{ box-shadow: 0 0 0 10px rgba(220,38,38,0);  }}
    100% {{ box-shadow: 0 0 0 0   rgba(220,38,38,0);  }}
  }}
  .critical-pulse {{ animation: pulse-border 2s infinite; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": MT_FIG_FACE, "axes.facecolor": MT_FIG_FACE,
    "axes.edgecolor": TEXT_COLOR,    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,       "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": "#2b2143" if st.session_state.theme == "dark" else "#c7f0ff",
})

# ── MODEL LOADING ──────────────────────────────────────────────────────────────
@st.cache_resource
def _cached_load_model(path: str):
    if YOLO is None:
        raise ImportError(f"ultralytics not available: {YOLO_IMPORT_ERROR}")
    return YOLO(path)

model        = None
model_status = "not_configured"
if not st.session_state.model_path:
    model_status = "not_configured"
elif not os.path.exists(st.session_state.model_path):
    model_status = "path_missing"
else:
    try:
        model        = _cached_load_model(st.session_state.model_path)
        model_status = "loaded"
    except Exception as e:
        model_status = f"load_error: {e}"

# ── INFERENCE ──────────────────────────────────────────────────────────────────
def run_yolo_inference(image, model, conf_threshold=0.25, device="cpu", box_thickness=2):
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = np.array(image)
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

    annotated_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    detections: List[Dict[str, Any]] = []

    try:
        if hasattr(model, "predict"):
            results = model.predict(source=img_rgb, conf=conf_threshold, device=device)
            boxes   = getattr(results[0], "boxes", None)
            if boxes is not None and hasattr(boxes, "xyxy"):
                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls   = boxes.cls.cpu().numpy().astype(int)
                names = getattr(model, "names", {}) or {}
                for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, cls):
                    xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)
                    detections.append({
                        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                        "confidence": float(conf), "class": int(c),
                        "name": names.get(int(c), str(int(c))),
                        "width": max(0, xmax - xmin), "height": max(0, ymax - ymin),
                    })
    except Exception as e:
        st.warning(f"Inference error: {e}")
        return Image.fromarray(img_rgb), pd.DataFrame()

    for d in detections:
        cv2.rectangle(annotated_bgr, (d["xmin"], d["ymin"]), (d["xmax"], d["ymax"]), (0, 255, 0), box_thickness)
        cv2.putText(annotated_bgr, f'{d["name"]} {d["confidence"]:.2f}',
                    (d["xmin"], max(d["ymin"] - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)), pd.DataFrame(detections)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; padding:20px; background:{CARD_BG}; border-radius:12px;
     box-shadow:0 4px 16px rgba(0,240,234,0.2);">
  <div style="font-size:40px; color:{PRIMARY_COLOR}; font-weight:bold; margin-bottom:10px;">🔬 MicroDetect</div>
  <div style="font-size:16px; color:{TEXT_COLOR};">{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── MODEL STATUS ───────────────────────────────────────────────────────────────
if model_status == "not_configured":
    st.warning("⚠️ **No model configured.** Please upload your `.pt` file or paste the model path in the sidebar.")
elif model_status == "path_missing":
    st.error("❌ **Model file not found.** Please check the path or upload the file in the sidebar.")
elif model_status.startswith("load_error"):
    st.error(f"❌ **Model failed to load.** {model_status.replace('load_error: ', '')}")
else:
    st.success("✅ Model loaded and ready!")

# ── FILE UPLOADER ──────────────────────────────────────────────────────────────
st.markdown(f"<p style='color:{TEXT_COLOR}; font-size:15px; font-weight:600; margin-bottom:4px;'>📤 Upload sample images for microplastic analysis</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Max 200MB per file • JPG, JPEG, PNG",
    label_visibility="collapsed",
)

if uploaded_files:
    st.session_state.uploaded_images = []
    for f in uploaded_files:
        if f.size > 200 * 1024 * 1024:
            st.error(f"File {f.name} exceeds 200MB limit.")
        else:
            try:
                st.session_state.uploaded_images.append((f.name, Image.open(f).convert("RGB")))
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")

# ── IMAGE PREVIEW ──────────────────────────────────────────────────────────────
if st.session_state.uploaded_images:
    st.markdown(f"""
    <div style='background:{CARD_BG}; padding:16px; border-radius:8px; margin:12px 0;
         border-left:4px solid {PRIMARY_COLOR};'>
        <b>📸 Uploaded Images ({len(st.session_state.uploaded_images)})</b>
    </div>
    """, unsafe_allow_html=True)
    cols_per_row = min(4, len(st.session_state.uploaded_images))
    for i in range(0, len(st.session_state.uploaded_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(st.session_state.uploaded_images):
                name, img = st.session_state.uploaded_images[idx]
                with cols[j]:
                    st.image(img, caption=name, use_container_width=True)

# ── DETECT BUTTON ──────────────────────────────────────────────────────────────
if st.session_state.uploaded_images:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        detect_button = st.button("🚀 Detect Microplastics", use_container_width=True, type="primary")

    if detect_button:
        if model is None:
            st.error("❌ Model not loaded. Please upload your model (.pt) file or fix the model path in the sidebar.")
        else:
            st.session_state.results = []
            progress_bar = st.progress(0)
            status_text  = st.empty()
            total        = len(st.session_state.uploaded_images)

            for idx, (name, img) in enumerate(st.session_state.uploaded_images):
                status_text.text(f"Processing {idx + 1}/{total}: {name}...")
                t0 = time.time()
                try:
                    annotated_img, df = run_yolo_inference(img, model, conf_threshold, device, box_thickness)
                except Exception as e:
                    st.error(f"Error processing {name}: {e}")
                    annotated_img, df = img, pd.DataFrame()

                st.session_state.results.append({
                    "filename": name, "original": img,
                    "annotated": annotated_img, "df": df,
                    "elapsed": time.time() - t0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                progress_bar.progress((idx + 1) / total)

            status_text.text("✅ Done!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ Successfully processed {total} image(s)!")

# ── RESULTS ────────────────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{CARD_BG}; padding:20px; border-radius:12px;
         box-shadow:0 4px 16px rgba(0,240,234,0.2);'>
        <h2 style='color:{PRIMARY_COLOR}; margin:0;'>🧪 Detection Results</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All Results"):
        st.session_state.uploaded_images = []
        st.session_state.results = []
        st.rerun()

    for result in st.session_state.results:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:{CARD_BG}; padding:16px; border-radius:8px;
             border-left:4px solid {SECONDARY_COLOR};'>
            <h3 style='margin:0; color:{TEXT_COLOR};'>🖼 {result['filename']}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(f"<div style='text-align:center; font-weight:bold; color:{TEXT_COLOR}; margin-bottom:8px;'>Original</div>", unsafe_allow_html=True)
            st.image(result["original"], use_container_width=True)
        with ic2:
            st.markdown(f"<div style='text-align:center; font-weight:bold; color:{TEXT_COLOR}; margin-bottom:8px;'>Annotated (YOLO Detection)</div>", unsafe_allow_html=True)
            st.image(result["annotated"], use_container_width=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        df = result["df"]

        # ── KPI VALUES ────────────────────────────────────────────
        total_particles = len(df)

        if not df.empty:
            img_w, img_h      = result["original"].size
            img_area          = img_w * img_h
            box_area          = int((df["width"] * df["height"]).sum())
            density_pct       = (box_area / img_area * 100) if img_area > 0 else 0.0
            detection_density = f"{density_pct:.2f}%"
            avg_conf_val      = f"{float(df['confidence'].mean()):.2f}"
        else:
            detection_density = "0.00%"
            avg_conf_val      = "—"

        # ── KPI CARDS ─────────────────────────────────────────────
        kc1, kc2, kc3, kc4 = st.columns(4)
        kpi_data = [
            (kc1, "🎯", "Total Particles Detected", str(total_particles), PRIMARY_COLOR),
            (kc2, "📐", "Detection Density",         detection_density,    "#a78bfa"),
            (kc3, "📊", "Avg Confidence Score",      avg_conf_val,         "#34d399"),
            (kc4, "⚡", "Processing Time",            f"{result['elapsed']:.2f}s", "#fb923c"),
        ]
        for col, icon, label, value, accent in kpi_data:
            with col:
                st.markdown(f"""
                <div style="background:{CARD_BG}; border:1px solid {accent}55;
                     border-top:3px solid {accent}; border-radius:10px;
                     padding:20px 14px; text-align:center;
                     box-shadow:0 2px 10px {accent}22;">
                    <div style="font-size:22px; margin-bottom:6px;">{icon}</div>
                    <div style="font-size:11px; color:{TEXT_COLOR}; opacity:0.75;
                         font-weight:600; letter-spacing:0.5px;
                         text-transform:uppercase; margin-bottom:10px;">{label}</div>
                    <div style="font-size:26px; font-weight:800; color:{accent};">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── FEEDBACK PANEL ────────────────────────────────────────
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:{CARD_BG}; padding:14px 18px; border-radius:8px; margin-bottom:8px;
             border-left:4px solid #f59e0b;'>
            <b style='color:{TEXT_COLOR}; font-size:16px;'>⚠️ Analysis Feedback & Warnings</b>
            <span style='font-size:13px; color:{TEXT_COLOR}; opacity:0.7; margin-left:12px;'>
                AI-generated safety assessment based on detection results
            </span>
        </div>
        """, unsafe_allow_html=True)
        render_feedback_panel(generate_feedback(df, result["filename"], conf_threshold), CARD_BG, TEXT_COLOR, PRIMARY_COLOR)

        # ── DETECTIONS TABLE ──────────────────────────────────────
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:{CARD_BG}; padding:12px; border-radius:8px; margin-bottom:12px;'><b>📋 Detections Table</b></div>", unsafe_allow_html=True)
        if df.empty:
            st.info("No microplastics detected. Try lowering the confidence threshold.")
        else:
            st.dataframe(df.reset_index(drop=True), use_container_width=True, height=200)

        # ── CHARTS ───────────────────────────────────────────────
        vc1, vc2, vc3 = st.columns(3)

        with vc1:
            with st.expander("🍩 Type Distribution"):
                fig, ax = plt.subplots(figsize=(5, 4))
                if df.empty:
                    ax.text(0.5, 0.5, "No detections", ha="center", va="center", color=TEXT_COLOR, fontsize=14)
                else:
                    counts = df["class"].value_counts()
                    labels = counts.index.tolist()
                    values = counts.values.tolist()
                    colors = color_map_for_values(len(labels), PIE_CMAP)
                    wedges, _, _ = ax.pie(
                        values, labels=labels,
                        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else None,
                        pctdistance=0.75, startangle=90, colors=colors,
                        wedgeprops=dict(width=0.45, edgecolor=MT_FIG_FACE)
                    )
                    ax.legend(wedges, [f"{l}: {v}" for l, v in zip(labels, values)],
                              bbox_to_anchor=(1.02, 0.6), loc="center left", frameon=False)
                ax.set_aspect("equal")
                st.pyplot(fig); plt.close(fig)
                st.caption("Distribution of detected microplastic types")

        with vc2:
            with st.expander("📈 Confidence Levels"):
                fig2, ax2 = plt.subplots(figsize=(5, max(3, 0.25 * min(len(df), max_display) + 1.2)))
                if df.empty:
                    ax2.text(0.5, 0.5, "No detections", ha="center", va="center", color=TEXT_COLOR, fontsize=14)
                else:
                    df_plot = df.sort_values("confidence", ascending=False).head(max_display).reset_index(drop=True)
                    colors  = color_map_for_values(len(df_plot), CONF_CMAP)[::-1]
                    y_pos   = np.arange(len(df_plot))
                    ax2.barh(y_pos, df_plot["confidence"], color=colors, edgecolor="#0b0b0b", height=0.6)
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels([f"{i+1}: {c}" for i, c in enumerate(df_plot["class"].tolist())], fontsize=9)
                    ax2.invert_yaxis(); ax2.set_xlim(0, 1.02)
                    ax2.set_xlabel("Confidence (0–1)")
                    ax2.set_title(f"Top {len(df_plot)} Detections")
                    for i, v in enumerate(df_plot["confidence"]):
                        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", color=TEXT_COLOR, fontsize=9)
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
                st.caption("Confidence scores for each detection")

        with vc3:
            with st.expander("📉 Size Distribution"):
                fig3, ax3 = plt.subplots(figsize=(5, 3))
                if df.empty:
                    ax3.text(0.5, 0.5, "No detections", ha="center", va="center", color=TEXT_COLOR, fontsize=14)
                else:
                    widths = df["width"]
                    bins   = min(12, max(4, int(math.sqrt(len(widths)) * 2)))
                    _, _, patches = ax3.hist(widths, bins=bins, color=HIST_COLOR, edgecolor=MT_FIG_FACE, alpha=0.95)
                    ax3.set_xlabel("Width (pixels)"); ax3.set_ylabel("Frequency")
                    ax3.set_title("Object Width Distribution")
                    for rect in patches:
                        h = rect.get_height()
                        if h > 0:
                            ax3.text(rect.get_x() + rect.get_width() / 2.0, h + 0.05,
                                     f"{int(h)}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
                plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)
                st.caption("Size distribution of detected objects")

        # ── DOWNLOADS ─────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️ Download Annotated Image",
                data=pil_to_bytes(result["annotated"]),
                file_name=f"annotated_{result['filename']}", mime="image/png",
                use_container_width=True)
        with dl2:
            if not df.empty:
                st.download_button("⬇️ Download Detections CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"detections_{result['filename'].rsplit('.', 1)[0]}.csv",
                    mime="text/csv", use_container_width=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("---")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; padding:24px; background:{CARD_BG}; border-radius:12px;
     box-shadow:0 4px 16px rgba(0,240,234,0.2);">
  <div style="font-size:28px; color:{PRIMARY_COLOR}; font-weight:bold; margin-bottom:12px;">
    ֎ Advanced AI-powered Microplastic Detection
  </div>
  <div style="font-size:16px; color:{TEXT_COLOR}; line-height:1.6;">
    For environmental research, monitoring, and marine protection.
  </div>
  <hr style="margin:20px 0; border:none; border-top:1px solid rgba(200,200,200,0.2);" />
  <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:12px; font-size:14px; margin:16px 0;">
    <span style="padding:8px 16px; border-radius:20px; background:{PRIMARY_COLOR}20; border:1px solid {PRIMARY_COLOR}; color:{TEXT_COLOR};">🧠 YOLOv8 Deep Learning</span>
    <span style="padding:8px 16px; border-radius:20px; background:{PRIMARY_COLOR}20; border:1px solid {PRIMARY_COLOR}; color:{TEXT_COLOR};">🌐 Computer Vision</span>
    <span style="padding:8px 16px; border-radius:20px; background:{PRIMARY_COLOR}20; border:1px solid {PRIMARY_COLOR}; color:{TEXT_COLOR};">🌍 Environmental Analysis</span>
    <span style="padding:8px 16px; border-radius:20px; background:{PRIMARY_COLOR}20; border:1px solid {PRIMARY_COLOR}; color:{TEXT_COLOR};">📊 Research</span>
    <span style="padding:8px 16px; border-radius:20px; background:{PRIMARY_COLOR}20; border:1px solid {PRIMARY_COLOR}; color:{TEXT_COLOR};">🌱 Environmental Impact</span>
    <span style="padding:8px 16px; border-radius:20px; background:{PRIMARY_COLOR}20; border:1px solid {PRIMARY_COLOR}; color:{TEXT_COLOR};">🐟 Marine Biology</span>
    <span style="padding:8px 16px; border-radius:20px; background:{PRIMARY_COLOR}20; border:1px solid {PRIMARY_COLOR}; color:{TEXT_COLOR};">🚨 Pollution Monitoring</span>
  </div>
  <div style="margin-top:16px; font-size:13px; color:{TEXT_COLOR}; opacity:0.7;">
    ©2025 Microplastic Detection AI. Advancing environmental science through artificial intelligence.
  </div>
</div>
""", unsafe_allow_html=True)