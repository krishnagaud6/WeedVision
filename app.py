"""WeedVision - AI weed detection for field imagery.
 
Runs the trained YOLO model stored in ``best.pt`` (same folder as this file)
and presents results as a dashboard, detection workspace, session analytics
and an About page.
 
Every number shown in the UI is derived from real model output or from the
loaded model's own metadata. Nothing is simulated.
"""
 
from __future__ import annotations
 
import base64
import hashlib
import html
import io
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
 
import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
 
# Never let Ultralytics pip-install packages at runtime (e.g. when a corrupt
# upload is opened). All dependencies come from requirements.txt.
os.environ["YOLO_AUTOINSTALL"] = "False"
 
# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"
TRAINING_CSV = APP_DIR / "results.csv"  # optional: Ultralytics training log
MAX_SIDE = 2560  # uploaded images are downscaled beyond this (memory safety)
MAX_HISTORY = 200
APP_VERSION = "1.0"
 
st.set_page_config(
    page_title="WeedVision | AI weed detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
NAV_LABELS = {
    "dashboard": ":material/space_dashboard:  Dashboard",
    "detect": ":material/photo_camera:  Detect weeds",
    "analytics": ":material/insights:  Analytics",
    "about": ":material/info:  About",
}
 
# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def _furrow_svg() -> str:
    """Perspective crop-row lines used as the hero artwork."""
    lines = []
    vx, vy = 520, -60
    for i in range(-9, 10):
        bx = 520 + i * 95
        cx = vx + (bx - vx) * 0.42 + i * 4
        lines.append(
            f'<path d="M{vx + i * 3},{vy} Q{cx},180 {bx},420" '
            'fill="none" stroke="#7FBF8E" stroke-width="1.4"/>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 420" '
        'preserveAspectRatio="xMidYMid slice">' + "".join(lines) + "</svg>"
    )
    return "url(\"data:image/svg+xml;utf8," + quote(svg) + "\")"
 
 
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --forest:#DDF5E6; --forest-2:#173B2A; --emerald:#20D889; --emerald-d:#35E39A;
  --moss:#8EB99B; --charcoal:#F1F7F2; --ink-2:#B5C7BA; --ivory:#09140E;
  --card:#12231A; --line:#294536; --amber:#E6BD63; --red:#FF8178;
  --serif:'Fraunces','Iowan Old Style',Georgia,serif;
  --sans:'Manrope','Segoe UI',system-ui,-apple-system,sans-serif;
}
html, body, [class*="st-"], .stApp { font-family:var(--sans); }
.stApp { background:radial-gradient(900px 500px at 85% -10%,rgba(28,93,59,.24),transparent 65%),var(--ivory); color:var(--charcoal); }
[data-testid="stMainBlockContainer"], .block-container { padding-top:1.2rem; padding-bottom:3.5rem; max-width:1380px; }
#MainMenu, footer, [data-testid="stDecoration"], .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }

/* Top navigation */
[data-testid="stHorizontalBlock"]:has(.wv-brand) { align-items:center; gap:.45rem; padding:.65rem .8rem; margin:.1rem 0 1.35rem;
  background:rgba(16,35,25,.96); border:1px solid #294536; border-radius:18px;
  box-shadow:0 12px 32px rgba(0,0,0,.22); }
.wv-brand { display:flex; align-items:center; gap:.65rem; min-height:42px; }
.wv-brand-mark { width:38px; height:38px; flex:none; border-radius:12px; background:linear-gradient(145deg,#2BE69A,#087D4B);
  display:grid; place-items:center; box-shadow:0 5px 18px rgba(32,216,137,.22); }
.wv-brand-name { font-family:var(--serif); font-size:1.25rem; font-weight:600; color:#F5FFF8; line-height:1.05; }
.wv-brand-sub { font-size:.68rem; color:#A7C5B1; margin-top:3px; white-space:nowrap; }
.wv-status { display:flex; justify-content:flex-end; align-items:center; gap:.45rem; color:#C8D9CE; font-size:.76rem; font-weight:700; white-space:nowrap; }
.wv-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--emerald); box-shadow:0 0 0 4px rgba(32,216,137,.13); }
.wv-dot.off { background:var(--red); box-shadow:0 0 0 4px rgba(255,129,120,.13); }
.wv-topnav-rule { height:1px; background:linear-gradient(90deg,transparent,#31513E,transparent); margin:-.6rem 0 1.5rem; }

/* Global typography */
h1,h2,h3 { font-family:var(--serif); color:var(--charcoal); letter-spacing:-.01em; }
p, li, label, .stMarkdown, .stCaption { color:var(--charcoal); }
.wv-page-title { font-family:var(--serif); font-size:2.2rem; font-weight:600; margin:0; color:#F1F8F3; line-height:1.15; }
.wv-page-sub { color:var(--ink-2); font-size:1rem; margin:.45rem 0 1.6rem; max-width:68ch; line-height:1.6; }
.wv-h3 { font-family:var(--serif); font-size:1.12rem; font-weight:600; color:#EAF6EE; margin:.2rem 0 .8rem; }
.wv-h2 { font-family:var(--serif); font-size:1.35rem; font-weight:600; color:#EAF6EE; margin:2rem 0 .8rem; }

/* Hero and model card */
.wv-hero { position:relative; overflow:hidden; border:1px solid #315541; border-radius:22px; padding:2.5rem 2.6rem;
  background:radial-gradient(1000px 380px at 85% -10%,#24583B 0%,transparent 62%),linear-gradient(135deg,#102D20,#0B1B12);
  color:#fff; display:grid; grid-template-columns:1.35fr 1fr; gap:2rem; align-items:center;
  box-shadow:0 20px 50px rgba(0,0,0,.28); }
.wv-hero::before { content:""; position:absolute; inset:0; background-image:__FURROW__; background-size:cover; opacity:.13;
  -webkit-mask-image:linear-gradient(90deg,transparent 10%,#000 80%); mask-image:linear-gradient(90deg,transparent 10%,#000 80%); }
.wv-hero > * { position:relative; }
.wv-hero h1 { font-family:var(--serif); color:#fff; font-size:2.65rem; font-weight:600; line-height:1.1; margin:0 0 1rem; }
.wv-hero p { color:#C7DACB; font-size:1.02rem; line-height:1.65; max-width:54ch; margin:0; }
.wv-model-card { background:rgba(255,255,255,.065); border:1px solid rgba(255,255,255,.15); border-radius:16px; padding:1.25rem 1.35rem; backdrop-filter:blur(6px); }
.wv-model-card .t { font-weight:800; font-size:.95rem; margin-bottom:.8rem; color:#fff; }
.wv-kv { display:flex; justify-content:space-between; gap:1rem; padding:.55rem 0; border-top:1px solid rgba(255,255,255,.12); font-size:.88rem; color:#C7DACB; }
.wv-kv b { color:#fff; font-weight:700; text-align:right; overflow-wrap:anywhere; }

/* Cards and panels */
.wv-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:1rem; margin:1.4rem 0 .4rem; }
.wv-stat { background:linear-gradient(145deg,#172B20,#112219); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.25rem; box-shadow:0 8px 22px rgba(0,0,0,.13); }
.wv-stat.accent { background:linear-gradient(145deg,#153C2A,#10291D); border-color:#28734E; }
.wv-stat-label { color:#B7C9BD; font-size:.82rem; font-weight:700; }
.wv-stat-value { font-family:var(--serif); font-size:2.1rem; font-weight:600; color:#F2FFF6; line-height:1.2; margin-top:.2rem; }
.wv-stat-note { color:#8FA697; font-size:.76rem; margin-top:.2rem; }
.wv-panel { background:linear-gradient(145deg,#14271C,#102017); border:1px solid var(--line); border-radius:18px; padding:1.35rem 1.45rem; box-shadow:0 10px 26px rgba(0,0,0,.12); }
.wv-panel h3 { font-family:var(--serif); font-size:1.12rem; font-weight:600; margin:0 0 .7rem; color:#EAF6EE; }
.wv-panel p,.wv-panel li { color:#B7C9BD; font-size:.93rem; line-height:1.65; }
.wv-chip { display:inline-block; padding:.3rem .75rem; margin:0 .4rem .5rem 0; border-radius:999px; background:#173C2A;
  color:#8FF0B9; font-weight:700; font-size:.8rem; border:1px solid #2D6847; }
.wv-steps { list-style:none; counter-reset:s; padding:0; margin:0; }
.wv-steps li { counter-increment:s; display:flex; gap:.9rem; padding:.7rem 0; border-top:1px solid #294536; color:#B7C9BD; }
.wv-steps li:first-child { border-top:0; }
.wv-steps li::before { content:counter(s); flex:none; width:28px; height:28px; border-radius:50%; background:#1D7048; color:#fff; font-weight:800; font-size:.82rem; display:grid; place-items:center; }
.wv-steps b { color:#F0F8F2; }
.wv-empty { border:1.5px dashed #42644F; border-radius:16px; padding:2.3rem 1.4rem; text-align:center; color:#B4C6B9; background:rgba(20,39,28,.55); }
.wv-empty b { display:block; font-family:var(--serif); font-size:1.15rem; color:#F1F8F3; margin-bottom:.3rem; }
.wv-band { display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.78rem; font-weight:700; }
.wv-note { border-left:3px solid var(--amber); background:#302A18; padding:.8rem 1rem; border-radius:0 10px 10px 0; color:#F2DDA6; font-size:.9rem; line-height:1.55; }

/* Buttons: force readable foregrounds on every Streamlit button variant. */
.stButton > button, .stDownloadButton > button, .stLinkButton > a,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  color:#F7FFF9 !important; -webkit-text-fill-color:#F7FFF9 !important;
  background:#1B3426 !important; border:1px solid #42634D !important; border-radius:11px !important;
  font-family:var(--sans) !important; font-weight:800 !important; min-height:42px;
  box-shadow:0 4px 12px rgba(0,0,0,.18); transition:background .15s ease,border-color .15s ease,transform .15s ease,box-shadow .15s ease;
}
.stButton > button *, .stDownloadButton > button *, .stLinkButton > a *,
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-secondary"] * {
  color:#F7FFF9 !important; -webkit-text-fill-color:#F7FFF9 !important; opacity:1 !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  color:#06180E !important; -webkit-text-fill-color:#06180E !important;
  background:linear-gradient(180deg,#46E9A0,#1DBB73) !important; border:1px solid #4BE8A2 !important;
  box-shadow:0 7px 18px rgba(22,190,112,.22);
}
.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] *, [data-testid="stBaseButton-primary"] * {
  color:#06180E !important; -webkit-text-fill-color:#06180E !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
  transform:translateY(-1px); border-color:#65E6A6 !important; box-shadow:0 8px 20px rgba(0,0,0,.25);
}
.stButton > button:disabled, .stDownloadButton > button:disabled,
[data-testid="stBaseButton-primary"]:disabled, [data-testid="stBaseButton-secondary"]:disabled {
  background:#26352C !important; color:#AAB9AE !important; -webkit-text-fill-color:#AAB9AE !important;
  border-color:#3A4B3F !important; opacity:1 !important; cursor:not-allowed;
}
.stButton > button:disabled *, .stDownloadButton > button:disabled * { color:#AAB9AE !important; -webkit-text-fill-color:#AAB9AE !important; }
button:focus-visible,[role="radio"]:focus-visible { outline:3px solid rgba(32,216,137,.7) !important; outline-offset:3px; }

/* Inputs, tabs, tables and chart containers */
.stApp label,.stApp [data-testid="stWidgetLabel"] p { color:#DCE9DF !important; }
.stTextInput input,.stNumberInput input,.stTextArea textarea,[data-baseweb="select"] > div {
  background:#14271C !important; color:#F1F8F3 !important; border-color:#3A5945 !important;
}
.stSlider [data-baseweb="slider"] div { color:#DCE9DF; }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #426B50; background:#14271C; border-radius:16px; padding:1.3rem; }
[data-testid="stFileUploaderDropzone"] * { color:#E5F2E8 !important; }
[data-testid="stFileUploaderDropzone"] button { background:#1D7048 !important; color:#fff !important; border-color:#328A5E !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--emerald); background:#183324; }
[data-testid="stImage"] img { border-radius:14px; border:1px solid #35513F; }
[data-testid="stDataFrame"] { border:1px solid #294536; border-radius:14px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid #294536; }
.stTabs [data-baseweb="tab"] { color:#B7C9BD !important; font-weight:700; padding:.55rem 1rem; }
.stTabs [aria-selected="true"] { color:#62E6A5 !important; }
[data-testid="stExpander"] { background:#12231A; border:1px solid #294536; border-radius:14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#E7F2EA !important; }
[data-testid="stAlert"] { border-radius:12px; }

@media (max-width:900px) {
  .wv-hero { grid-template-columns:1fr; padding:1.7rem 1.4rem; }
  .wv-hero h1 { font-size:2rem; }
  .wv-page-title { font-size:1.8rem; }
  .wv-brand-sub { display:none; }
}
@media (max-width:600px) {
  [data-testid="stMainBlockContainer"],.block-container { padding-left:1rem; padding-right:1rem; }
  .wv-hero { border-radius:16px; }
  .wv-stat-value { font-size:1.7rem; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
</style>
""".replace("__FURROW__", _furrow_svg())
def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
 
 
# --------------------------------------------------------------------------
# Model loading and inference
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading WeedVision model...")
def load_model():
    """Load best.pt once per server process and collect real metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH.name} was not found next to app.py")

    from ultralytics import YOLO  # imported lazily so errors can be shown in the UI
    model = YOLO(str(MODEL_PATH))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    info = {
        "file": MODEL_PATH.name,
        "size_mb": MODEL_PATH.stat().st_size / 1e6,
        "task": str(getattr(model, "task", "detect")),
        "classes": [str(names[k]) for k in sorted(names)],
        "params_m": None,
    }
    try:
        info["params_m"] = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:  # metadata only; never block the app on this
        pass
    return model, info
 
 
@st.cache_resource
def inference_lock() -> threading.Lock:
    """The shared model is not guaranteed thread-safe; serialise predictions."""
    return threading.Lock()
 
 
def read_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img
 
 
def coverage_percent(res, width: int, height: int) -> float:
    """Share of the image covered by detections (union, so overlaps count once).
 
    Uses segmentation polygons when the model provides them, otherwise boxes.
    """
    scale = 512 / max(width, height)
    gw, gh = max(1, int(width * scale)), max(1, int(height * scale))
    grid = np.zeros((gh, gw), np.uint8)
    if getattr(res, "masks", None) is not None and len(res.masks) > 0:
        for poly in res.masks.xy:
            if len(poly) >= 3:
                cv2.fillPoly(grid, [(poly * scale).astype(np.int32)], 1)
    elif res.boxes is not None and len(res.boxes) > 0:
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            grid[int(y1 * scale): int(np.ceil(y2 * scale)), int(x1 * scale): int(np.ceil(x2 * scale))] = 1
    return float(grid.mean() * 100)
 
 
def band(conf: float) -> str:
    return "High" if conf >= 0.75 else "Medium" if conf >= 0.5 else "Low"
 
 
def run_inference(model, image: Image.Image, conf: float, iou: float, imgsz: int) -> dict:
    t0 = time.perf_counter()
    with inference_lock():
        res = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    total_ms = (time.perf_counter() - t0) * 1000
 
    # Ultralytics plot() returns BGR; convert for display.
    annotated = np.ascontiguousarray(res.plot()[:, :, ::-1])
    w, h = image.size
    names = res.names
    rows = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-confs)
        for rank, i in enumerate(order, start=1):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            rows.append({
                "ID": rank,
                "Class": str(names[int(clss[i])]),
                "Confidence": float(confs[i]),
                "Review band": band(float(confs[i])),
                "X1": round(x1), "Y1": round(y1), "X2": round(x2), "Y2": round(y2),
                "Box area (% of image)": round((x2 - x1) * (y2 - y1) / (w * h) * 100, 2),
            })
    det = pd.DataFrame(rows)
    speed = getattr(res, "speed", {}) or {}
    return {
        "annotated": annotated,
        "detections": det,
        "width": w,
        "height": h,
        "coverage": coverage_percent(res, w, h),
        "inference_ms": float(speed.get("inference", total_ms)),
        "total_ms": total_ms,
        "conf": conf, "iou": iou, "imgsz": imgsz,
    }
 
 
# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("nav", "dashboard")
    st.session_state.setdefault("runs", [])  # one record per analysed image
    st.session_state.setdefault("dets", [])  # one record per detection
    st.session_state.setdefault("last", None)
    st.session_state.setdefault("run_seq", 0)
 
 
def goto(page: str) -> None:
    st.session_state["nav"] = page
 
 
def record_run(filename: str, file_id: str, out: dict) -> None:
    st.session_state["run_seq"] += 1
    rid = st.session_state["run_seq"]
    det: pd.DataFrame = out["detections"]
    counts = det["Class"].value_counts().to_dict() if not det.empty else {}
    st.session_state["runs"].append({
        "Run": rid,
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Image": filename,
        "Detections": int(len(det)),
        "Avg confidence": float(det["Confidence"].mean()) if not det.empty else None,
        "Coverage (%)": round(out["coverage"], 2),
        "Inference (ms)": round(out["inference_ms"], 1),
        "Threshold": out["conf"],
        "Classes": ", ".join(f"{k} x{v}" for k, v in counts.items()) or "none",
    })
    for _, r in det.iterrows():
        st.session_state["dets"].append({
            "Run": rid, "Image": filename, "Class": r["Class"],
            "Confidence": r["Confidence"], "Box area (% of image)": r["Box area (% of image)"],
        })
    st.session_state["runs"] = st.session_state["runs"][-MAX_HISTORY:]
    # Keep detections only for runs still present in the session history.
    retained_run_ids = {run["Run"] for run in st.session_state["runs"]}
    st.session_state["dets"] = [
        detection
        for detection in st.session_state["dets"]
        if detection["Run"] in retained_run_ids
    ]
 
 
# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def class_chips(classes: list[str], limit: int = 16) -> str:
    shown = "".join(f'<span class="wv-chip">{esc(c)}</span>' for c in classes[:limit])
    if len(classes) > limit:
        shown += f'<span class="wv-chip">+{len(classes) - limit} more</span>'
    return shown
 
 
def esc(value) -> str:
    return html.escape(str(value))
 
 
def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<h1 class="wv-page-title">{esc(title)}</h1><p class="wv-page-sub">{esc(subtitle)}</p>',
        unsafe_allow_html=True,
    )
 
 
def h2(text: str) -> None:
    st.markdown(f'<div class="wv-h2">{esc(text)}</div>', unsafe_allow_html=True)
 
 
def stat(label: str, value: str, note: str = "", accent: bool = False) -> str:
    cls = "wv-stat accent" if accent else "wv-stat"
    return (f'<div class="{cls}"><div class="wv-stat-label">{esc(label)}</div>'
            f'<div class="wv-stat-value">{esc(value)}</div>'
            f'<div class="wv-stat-note">{esc(note)}</div></div>')
 
 
def stat_grid(cards: list[str]) -> None:
    st.markdown('<div class="wv-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
 
 
def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="wv-empty"><b>{esc(title)}</b>{esc(body)}</div>', unsafe_allow_html=True)
 
 
def style_chart(chart: alt.Chart, height: int = 280) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0, fill="#14271C")
        .configure_axis(labelColor="#C3D4C8", titleColor="#DDE9E0", gridColor="#294536",
                        domainColor="#42634D", labelFont="Manrope", titleFont="Manrope")
        .configure_legend(labelColor="#DDE9E0", titleColor="#DDE9E0", labelFont="Manrope", titleFont="Manrope")
        .configure(background="#14271C")
    )


def show_chart(chart: alt.Chart) -> None:
    """Render charts across Streamlit versions with different width APIs."""
    try:
        st.altair_chart(chart, use_container_width=True)
    except TypeError:
        st.altair_chart(chart)
 
 
EMERALD_RANGE = ["#35E39A", "#8BE8B2", "#4DBB7D", "#E6BD63", "#2D8F5B", "#A6DDB8", "#D4F1DD", "#FF8178"]
 
 
def df_json_safe(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))
 
 
# --------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------
def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
 
 
def build_html_report(filename: str, out: dict, info: dict) -> str:
    det: pd.DataFrame = out["detections"]
    buf = io.BytesIO()
    Image.fromarray(out["annotated"]).save(buf, format="JPEG", quality=88, optimize=True)
    img64 = base64.b64encode(buf.getvalue()).decode()
    when = datetime.now().strftime("%d %b %Y, %H:%M")
    rows = "".join(
        f"<tr><td>{r.ID}</td><td>{esc(r.Class)}</td><td>{r.Confidence:.1%}</td><td>{r._4}</td>"
        f"<td>({r.X1}, {r.Y1}) to ({r.X2}, {r.Y2})</td></tr>"
        for r in det.itertuples()
    ) or '<tr><td colspan="5">No detections above the confidence threshold.</td></tr>'
    avg = f"{det['Confidence'].mean():.1%}" if not det.empty else "n/a"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WeedVision report - {esc(filename)}</title>
<style>
body{{font-family:'Segoe UI',system-ui,sans-serif;color:#1E2421;background:#F5F7F3;margin:0;padding:2rem}}
.wrap{{max-width:960px;margin:auto;background:#fff;border:1px solid #DDE5DD;border-radius:18px;overflow:hidden}}
header{{background:#0E2B1E;color:#fff;padding:1.6rem 2rem}} header h1{{margin:0;font-size:1.6rem}}
header p{{margin:.3rem 0 0;color:#B7CDBD}} main{{padding:1.6rem 2rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:1rem 0 1.4rem}}
.s{{border:1px solid #DDE5DD;border-radius:12px;padding:.8rem 1rem}} .s b{{display:block;font-size:1.5rem;color:#0E2B1E}}
.s span{{color:#4A544E;font-size:.82rem}} img{{width:100%;border-radius:12px;border:1px solid #DDE5DD}}
table{{width:100%;border-collapse:collapse;margin-top:1rem;font-size:.92rem}}
th,td{{text-align:left;padding:.55rem .6rem;border-bottom:1px solid #E6ECE6}} th{{color:#4A544E}}
footer{{padding:1rem 2rem 1.6rem;color:#6B766F;font-size:.8rem}}
</style></head><body><div class="wrap">
<header><h1>WeedVision detection report</h1><p>{esc(filename)} &middot; generated {when}</p></header>
<main>
<div class="grid">
<div class="s"><b>{len(det)}</b><span>Detections</span></div>
<div class="s"><b>{avg}</b><span>Average confidence</span></div>
<div class="s"><b>{out['coverage']:.1f}%</b><span>Image area covered</span></div>
<div class="s"><b>{out['inference_ms']:.0f} ms</b><span>Inference time</span></div>
</div>
<img alt="Annotated detection result" src="data:image/jpeg;base64,{img64}">
<h3>Detections</h3>
<table><tr><th>#</th><th>Class</th><th>Confidence</th><th>Review band</th><th>Bounding box (px)</th></tr>{rows}</table>
<h3>Run settings</h3>
<p>Confidence threshold {out['conf']:.2f} &middot; IoU {out['iou']:.2f} &middot; inference size {out['imgsz']} px &middot;
image {out['width']} x {out['height']} px &middot; weights {esc(info['file'])} ({esc(info['task'])},
{len(info['classes'])} classes)</p>
</main>
<footer>Coverage is the union of detected regions as a share of the image. Review bands (High at least 75%, Medium 50-75%,
Low under 50%) are triage aids, not accuracy guarantees. Results depend on the model&rsquo;s training classes and image quality.</footer>
</div></body></html>"""
 
 
# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def top_navigation(info: dict | None, error: str | None) -> None:
    """Responsive top navigation with explicit high-contrast button styles."""
    nav_cols = st.columns([2.0, 1.0, 1.2, 1.0, .85, 1.35], gap="small")
    with nav_cols[0]:
        st.markdown(
            '<div class="wv-brand"><div class="wv-brand-mark">'
            '<svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10z"/>'
            '<path d="M2 21c0-3 1.9-5.5 6-7"/></svg></div>'
            '<div><div class="wv-brand-name">WeedVision</div><div class="wv-brand-sub">AI weed detection</div></div></div>',
            unsafe_allow_html=True,
        )
    current = st.session_state.get("nav", "dashboard")
    items = [("dashboard", "Dashboard"), ("detect", "Detect weeds"), ("analytics", "Analytics"), ("about", "About")]
    for col, (key, label) in zip(nav_cols[1:5], items):
        with col:
            st.button(label, key=f"topnav_{key}", type="primary" if current == key else "secondary",
                      on_click=goto, args=(key,), width="stretch")
    with nav_cols[5]:
        if info:
            st.markdown('<div class="wv-status"><span class="wv-dot"></span>MODEL READY</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="wv-status"><span class="wv-dot off"></span>MODEL UNAVAILABLE</div>', unsafe_allow_html=True)


def page_dashboard(info: dict) -> None:
    params = f'{info["params_m"]:.1f} M' if info["params_m"] else "n/a"
    st.markdown(
        '<div class="wv-hero"><div>'
        '<h1>See the weeds in your fields before they cost you yield.</h1>'
        '<p>Upload a photo of a crop row or a field. WeedVision&rsquo;s trained YOLO model marks every weed it finds, '
        'scores each one and packages the result into a report you can share.</p></div>'
        '<div class="wv-model-card"><div class="t"><span class="wv-dot"></span>Model ready</div>'
        f'<div class="wv-kv"><span>Weights</span><b>{esc(info["file"])} ({info["size_mb"]:.1f} MB)</b></div>'
        f'<div class="wv-kv"><span>Task</span><b>{esc(info["task"].capitalize())}</b></div>'
        f'<div class="wv-kv"><span>Classes</span><b>{len(info["classes"])}</b></div>'
        f'<div class="wv-kv"><span>Parameters</span><b>{params}</b></div></div></div>',
        unsafe_allow_html=True,
    )
 
    runs = pd.DataFrame(st.session_state["runs"])
    if runs.empty:
        cards = [
            stat("Images analysed", "0", "This session", True),
            stat("Detections", "0", "This session"),
            stat("Average confidence", "n/a", "Appears after your first analysis"),
            stat("Average inference time", "n/a", "Appears after your first analysis"),
        ]
    else:
        confs = pd.DataFrame(st.session_state["dets"])
        avg_conf = f'{confs["Confidence"].mean():.1%}' if not confs.empty else "n/a"
        cards = [
            stat("Images analysed", f"{len(runs)}", "This session", True),
            stat("Detections", f'{int(runs["Detections"].sum())}', "Across all images"),
            stat("Average confidence", avg_conf, "Across all detections"),
            stat("Average inference time", f'{runs["Inference (ms)"].mean():.0f} ms', "Per image, on this server"),
        ]
    stat_grid(cards)
 
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        chips = class_chips(info["classes"])
        st.markdown(
            '<div class="wv-panel"><h3>What the model detects</h3>' + chips +
            '<h3 style="margin-top:1.1rem">How it works</h3><ol class="wv-steps">'
            '<li><div><b>Upload an image.</b> JPG, PNG or WebP, or take a photo with your camera.</div></li>'
            '<li><div><b>Choose a confidence threshold.</b> Higher values show fewer, more certain detections.</div></li>'
            '<li><div><b>Review and export.</b> Inspect the annotated image, then download the image, data or report.</div></li>'
            '</ol></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.button("Analyse an image", type="primary", on_click=goto, args=("detect",), width="stretch")
    with right:
        st.markdown('<div class="wv-h3">Recent analyses</div>', unsafe_allow_html=True)
        if runs.empty:
            empty_state("No analyses yet", "Run your first detection and it will be listed here.")
        else:
            recent = runs.sort_values("Run", ascending=False).head(6)[
                ["Time", "Image", "Detections", "Avg confidence", "Coverage (%)"]]
            st.dataframe(
                recent, hide_index=True, width="stretch",
                column_config={
                    "Avg confidence": st.column_config.NumberColumn(format="percent"),
                    "Coverage (%)": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
 
 
def detection_table(det: pd.DataFrame) -> None:
    view = det.copy()
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="percent"),
            "Box area (% of image)": st.column_config.NumberColumn("Box area (% of image)", format="%.2f"),
        },
    )
 
 
def render_result(filename: str, original: Image.Image, out: dict, info: dict) -> None:
    det: pd.DataFrame = out["detections"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Original**")
        st.image(original, width="stretch")
    with c2:
        st.markdown("**WeedVision result**")
        st.image(out["annotated"], width="stretch")
 
    avg = f'{det["Confidence"].mean():.1%}' if not det.empty else "n/a"
    top = f'{det["Confidence"].max():.1%}' if not det.empty else "n/a"
    stat_grid([
        stat("Detections", str(len(det)), f'Threshold {out["conf"]:.2f}', True),
        stat("Average confidence", avg, "Across detections"),
        stat("Highest confidence", top, "Strongest detection"),
        stat("Image area covered", f'{out["coverage"]:.1f}%', "Union of detected regions"),
        stat("Inference time", f'{out["inference_ms"]:.0f} ms', f'{out["width"]} x {out["height"]} px'),
    ])
 
    if det.empty:
        st.info("No objects were detected above this confidence threshold. Try lowering the threshold, "
                "or use a sharper, closer image.")
    else:
        h2("Detections")
        t, ch = st.columns([1.6, 1], gap="large")
        with t:
            detection_table(det)
        with ch:
            counts = det["Class"].value_counts().rename_axis("Class").reset_index(name="Count")
            chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=6, size=22).encode(
                y=alt.Y("Class:N", sort="-x", title=None),
                x=alt.X("Count:Q", title="Detections", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Class:N", scale=alt.Scale(range=EMERALD_RANGE), legend=None),
                tooltip=["Class", "Count"],
            )
            show_chart(style_chart(chart, 220))
 
    h2("Export")
    stem = Path(filename).stem or "image"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "image": filename, "generated": datetime.now().isoformat(timespec="seconds"),
        "model": {"weights": info["file"], "task": info["task"], "classes": info["classes"]},
        "settings": {"confidence": out["conf"], "iou": out["iou"], "imgsz": out["imgsz"]},
        "summary": {
            "detections": int(len(det)),
            "average_confidence": None if det.empty else float(det["Confidence"].mean()),
            "image_area_covered_percent": round(out["coverage"], 2),
            "inference_ms": round(out["inference_ms"], 1),
        },
        "detections": df_json_safe(det),
    }
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Annotated image (PNG)", png_bytes(out["annotated"]), f"weedvision_{stem}_{stamp}.png",
                       "image/png", width="stretch", type="primary")
    d2.download_button("Report (HTML)", build_html_report(filename, out, info), f"weedvision_report_{stem}_{stamp}.html",
                       "text/html", width="stretch")
    d3.download_button("Detections (CSV)", det.to_csv(index=False), f"weedvision_{stem}_{stamp}.csv",
                       "text/csv", width="stretch", disabled=det.empty)
    d4.download_button("Data (JSON)", json.dumps(payload, indent=2), f"weedvision_{stem}_{stamp}.json",
                       "application/json", width="stretch")
 
 
def page_detect(model, info: dict) -> None:
    page_header("Detect weeds", "Add an image, choose how strict the model should be, then run the analysis.")
 
    tab_up, tab_cam = st.tabs(["Upload a file", "Use the camera"])
    with tab_up:
        uploaded = st.file_uploader("Choose a plant or field image", type=["jpg", "jpeg", "png", "webp"],
                                    label_visibility="collapsed", key="uploader")
    with tab_cam:
        shot = st.camera_input("Take a photo", label_visibility="collapsed", key="camera")
 
    source = uploaded or shot
    # Use a regular section instead of Streamlit's expander. This avoids the
    # overlapping/duplicated expander title while preserving all controls.
    st.markdown('<div class="wv-h3" style="margin:1rem 0 .75rem">Detection settings</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        s1, s2, s3 = st.columns(3, gap="large")
        conf = s1.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                         help="Detections scoring below this are hidden. Raise it to reduce false alarms.")
        iou = s2.slider("Overlap filter (IoU)", 0.10, 0.90, 0.70, 0.05,
                        help="Lower values merge overlapping boxes more aggressively.")
        imgsz = s3.select_slider("Inference size (px)", options=[320, 480, 640, 800, 960, 1280], value=640,
                                 help="Larger sizes can find smaller weeds but run slower.")
 
    if source is None:
        empty_state("Add an image to begin", "Drop a JPG, PNG or WebP above, or switch to the camera tab.")
        return
 
    data = source.getvalue()
    file_id = hashlib.md5(data).hexdigest()[:12]
    filename = getattr(source, "name", None) or "camera-photo.jpg"
    try:
        image = read_image(data)
    except Exception:
        st.error("This file could not be read as an image. Please upload a valid JPG, PNG or WebP.")
        return
 
    if st.button("Analyse image", type="primary"):
        try:
            with st.spinner("Running WeedVision..."):
                out = run_inference(model, image, conf, iou, imgsz)
        except Exception as exc:
            st.error("Detection failed. Try a smaller inference size or a different image.")
            st.exception(exc)
            return
        record_run(filename, file_id, out)
        st.session_state["last"] = {"file_id": file_id, "filename": filename, "out": out}
 
    last = st.session_state.get("last")
    if last and last["file_id"] == file_id:
        render_result(last["filename"], image, last["out"], info)
    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Input image**")
            st.image(image, width="stretch")
            st.caption(f"{filename} - {image.size[0]} x {image.size[1]} px")
        with right:
            st.markdown("**WeedVision result**")
            empty_state("Ready when you are", "Select Analyse image to see the annotated result here.")
 
 
def page_analytics(info: dict) -> None:
    page_header("Analytics", "Trends across every image analysed in this session. Figures come straight from model output.")
    runs = pd.DataFrame(st.session_state["runs"])
    dets = pd.DataFrame(st.session_state["dets"])
    if runs.empty:
        empty_state("No data yet", "Analyse an image on the Detect page and your analytics will build up here.")
        st.write("")
        st.button("Go to Detect weeds", type="primary", on_click=goto, args=("detect",))
        return
 
    avg_conf = f'{dets["Confidence"].mean():.1%}' if not dets.empty else "n/a"
    stat_grid([
        stat("Images analysed", str(len(runs)), "This session", True),
        stat("Total detections", str(int(runs["Detections"].sum())), "All classes"),
        stat("Average confidence", avg_conf, "All detections"),
        stat("Average coverage", f'{runs["Coverage (%)"].mean():.1f}%', "Image area per photo"),
        stat("Median inference", f'{runs["Inference (ms)"].median():.0f} ms', "Per image"),
    ])
 
    h2("Detection mix")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Detections by class**")
        if dets.empty:
            empty_state("No detections recorded", "All analysed images had no detections above their thresholds.")
        else:
            counts = dets["Class"].value_counts().rename_axis("Class").reset_index(name="Count")
            chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=6, size=48).encode(
                x=alt.X("Class:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title="Detections", axis=alt.Axis(tickMinStep=1)),
                color=alt.Color("Class:N", scale=alt.Scale(range=EMERALD_RANGE), legend=None),
                tooltip=["Class", "Count"])
            show_chart(style_chart(chart))
    with b:
        st.markdown("**Confidence distribution**")
        if dets.empty:
            empty_state("No detections recorded", "A confidence histogram appears once something is detected.")
        else:
            chart = alt.Chart(dets).mark_bar(color="#12B76A", cornerRadiusEnd=4).encode(
                x=alt.X("Confidence:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="Confidence",
                        axis=alt.Axis(format="%")),
                y=alt.Y("count():Q", title="Detections", axis=alt.Axis(tickMinStep=1)),
                tooltip=[alt.Tooltip("count():Q", title="Detections")])
            show_chart(style_chart(chart))
 
    h2("Image by image")
    c, d = st.columns(2, gap="large")
    runs_plot = runs.assign(Label=runs["Run"].astype(str))
    with c:
        st.markdown("**Detections per image**")
        chart = alt.Chart(runs_plot).mark_bar(color="#0E2B1E", cornerRadiusEnd=6, size=36).encode(
            x=alt.X("Label:N", sort=None, title="Run", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Detections:Q", title="Detections", axis=alt.Axis(tickMinStep=1)),
            tooltip=["Run", "Image", "Detections"])
        show_chart(style_chart(chart))
    with d:
        st.markdown("**Image area covered (%)**")
        chart = alt.Chart(runs_plot).mark_line(color="#12B76A", point=alt.OverlayMarkDef(color="#0B8F52", size=70),
                                               strokeWidth=3).encode(
            x=alt.X("Label:N", sort=None, title="Run", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Coverage (%):Q", title="Coverage (%)"),
            tooltip=["Run", "Image", "Coverage (%)"])
        show_chart(style_chart(chart))
 
    if not dets.empty and dets["Class"].nunique() > 1:
        st.markdown("**Confidence by class**")
        chart = alt.Chart(dets).mark_boxplot(extent="min-max", size=36).encode(
            x=alt.X("Class:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Confidence:Q", title="Confidence", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Class:N", scale=alt.Scale(range=EMERALD_RANGE), legend=None))
        show_chart(style_chart(chart))
 
    h2("Session log")
    log = runs.sort_values("Run", ascending=False)
    st.dataframe(
        log, hide_index=True, width="stretch",
        column_config={
            "Avg confidence": st.column_config.NumberColumn(format="percent"),
            "Threshold": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    x1, x2, x3, _ = st.columns([1, 1, 1, 2])
    x1.download_button("Session log (CSV)", log.to_csv(index=False), "weedvision_session_log.csv", "text/csv",
                       width="stretch", type="primary")
    x2.download_button("All detections (CSV)", dets.to_csv(index=False), "weedvision_all_detections.csv", "text/csv",
                       width="stretch", disabled=dets.empty)
    if x3.button("Clear session data", width="stretch"):
        st.session_state["runs"], st.session_state["dets"], st.session_state["last"] = [], [], None
        st.rerun()
    st.caption("Analytics are kept in memory for this browser session only. Nothing is stored on the server.")
 
 
def find_training_csv() -> Path | None:
    """Find results.csv beside app.py or inside this project's Ultralytics runs folder."""
    candidates = []
    if TRAINING_CSV.is_file():
        candidates.append(TRAINING_CSV)
    runs_dir = APP_DIR / "runs"
    if runs_dir.is_dir():
        candidates.extend(
            path for path in runs_dir.rglob("results.csv")
            if path.is_file() and path not in candidates
        )
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def training_metrics() -> dict | None:
    """Read the final valid metrics row from the latest Ultralytics training log."""
    csv_path = find_training_csv()
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [str(column).strip() for column in df.columns]
        if df.empty:
            return None
        wanted = {
            "metrics/precision(B)": "Precision",
            "metrics/recall(B)": "Recall",
            "metrics/mAP50(B)": "mAP@50",
            "metrics/mAP50-95(B)": "mAP@50-95",
        }
        metric_columns = [column for column in wanted if column in df.columns]
        if not metric_columns:
            return None
        valid_rows = df.dropna(subset=metric_columns, how="all")
        if valid_rows.empty:
            return None
        last = valid_rows.iloc[-1]
        found = {
            label: float(last[column])
            for column, label in wanted.items()
            if column in df.columns and pd.notna(last[column])
        }
        if not found:
            return None
        found["Epochs"] = (
            int(float(last["epoch"])) + 1
            if "epoch" in df.columns and pd.notna(last.get("epoch"))
            else len(df)
        )
        return found
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None
 
 
def page_about(info: dict) -> None:
    page_header("About WeedVision", "An AI assistant that finds weeds in crop imagery so they can be managed earlier.")
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        st.markdown(
            '<div class="wv-panel"><h3>The problem</h3>'
            '<p>Weeds compete with crops for light, water and nutrients. Spotting them early normally means walking '
            'the field and judging by eye, which is slow and varies from person to person.</p>'
            '<h3 style="margin-top:1.1rem">What WeedVision does</h3>'
            '<p>WeedVision runs a custom-trained YOLO object detection model over a photo and outlines every weed it '
            'recognises. Each detection carries a confidence score and a position, so a person can quickly review the '
            'result, export it and decide what to do next.</p>'
            '<h3 style="margin-top:1.1rem">How the pipeline runs</h3><ol class="wv-steps">'
            '<li><div><b>Prepare.</b> The image is orientation-corrected and converted to RGB.</div></li>'
            '<li><div><b>Detect.</b> The model predicts boxes and classes at the chosen inference size.</div></li>'
            '<li><div><b>Filter.</b> Detections below your confidence threshold are dropped, and overlapping boxes are merged.</div></li>'
            '<li><div><b>Report.</b> Results are drawn on the image and summarised as statistics, charts and downloadable files.</div></li>'
            '</ol></div>',
            unsafe_allow_html=True,
        )
    with b:
        chips = class_chips(info["classes"])
        params = f'{info["params_m"]:.1f} M' if info["params_m"] else "n/a"
        st.markdown(
            '<div class="wv-panel"><h3>Model card</h3>'
            f'<div class="wv-kv" style="color:#B7C9BD;border-color:#294536"><span>Weights file</span><b style="color:#F1F8F3">{esc(info["file"])}</b></div>'
            f'<div class="wv-kv" style="color:#B7C9BD;border-color:#294536"><span>File size</span><b style="color:#F1F8F3">{info["size_mb"]:.1f} MB</b></div>'
            f'<div class="wv-kv" style="color:#B7C9BD;border-color:#294536"><span>Task</span><b style="color:#F1F8F3">{esc(info["task"].capitalize())}</b></div>'
            f'<div class="wv-kv" style="color:#B7C9BD;border-color:#294536"><span>Parameters</span><b style="color:#F1F8F3">{params}</b></div>'
            f'<div style="margin-top:1rem"><p style="margin-bottom:.5rem"><b>Classes</b></p>{chips}</div></div>',
            unsafe_allow_html=True,
        )
        metrics = training_metrics()
        if metrics:
            st.write("")
            cards = [stat(k, f"{v:.1%}" if k != "Epochs" else str(v), "Final training epoch") for k, v in metrics.items()]
            st.markdown('<div class="wv-h3">Training results</div>', unsafe_allow_html=True)
            stat_grid(cards)
        else:
            st.write("")
            if find_training_csv() is None:
                st.info(
                    "Training metrics are unavailable because no Ultralytics results.csv "
                    "was found. Keep the training output in this project's runs folder "
                    "(for example, runs/detect/train/results.csv), or place results.csv "
                    "beside app.py. Detection with best.pt will continue to work."
                )
            else:
                st.warning(
                    "A results.csv file was found, but it does not contain readable "
                    "precision, recall or mAP columns. Check that it came from an "
                    "Ultralytics detection training run."
                )
 
    h2("Using the results responsibly")
    st.markdown(
        '<div class="wv-panel"><ul>'
        '<li>Predictions are limited to the classes the model was trained on, and quality depends on lighting, focus and distance.</li>'
        '<li>Confidence is the model&rsquo;s own score, not a measured accuracy. Review low-confidence detections by eye.</li>'
        '<li>&ldquo;Image area covered&rdquo; describes the photo, not the whole field, and is not a calibrated infestation measure.</li>'
        '<li>Images are processed in memory for your session and are not saved by this application.</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )
 
 
def page_model_error(message: str) -> None:
    page_header("Model could not be loaded", "WeedVision needs its trained weights before it can analyse images.")
    st.markdown(
        '<div class="wv-note"><b>What to check</b><ul style="margin:.5rem 0 0">'
        f'<li><code>best.pt</code> must sit in the same folder as <code>app.py</code> (looked in {esc(APP_DIR)}).</li>'
        '<li>If the file is over 100 MB, track it with Git LFS before pushing to GitHub.</li>'
        '<li>Confirm <code>requirements.txt</code> installed without errors in the deployment logs.</li></ul></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Technical details"):
        st.code(message)
 
 
# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_state()
 
    model = info = None
    error = None
    try:
        model, info = load_model()
    except Exception as exc:  # shown in the UI; the app stays up
        error = f"{type(exc).__name__}: {exc}"
 
    top_navigation(info, error)
    if error:
        page_model_error(error)
        return
 
    page = st.session_state["nav"]
    if page == "dashboard":
        page_dashboard(info)
    elif page == "detect":
        page_detect(model, info)
    elif page == "analytics":
        page_analytics(info)
    else:
        page_about(info)
 
 
main()
 
