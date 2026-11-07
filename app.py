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
    initial_sidebar_state="auto",
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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Manrope:wght@400;500;600;700&display=swap');
 
:root{
  --forest:#0E2B1E; --forest-2:#143A29; --emerald:#12B76A; --emerald-d:#0B8F52;
  --moss:#5F8F6B; --charcoal:#1E2421; --ink-2:#4A544E; --ivory:#F5F7F3;
  --card:#FFFFFF; --line:#DDE5DD; --amber:#D9A441; --red:#C2493D;
  --serif:'Fraunces', 'Iowan Old Style', Georgia, serif;
  --sans:'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
html, body, [class*="st-"], .stApp { font-family: var(--sans); }
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *, span[translate="no"][role="img"] {
  font-family: "Material Symbols Rounded", "Material Icons" !important; }
.stApp { background: var(--ivory); color: var(--charcoal); }
 
/* chrome */
#MainMenu, footer, [data-testid="stDecoration"], .stDeployButton,
[data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px;
}
 
/* sidebar */
[data-testid="stSidebar"] { background: var(--forest); border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] * { color: #E6EFE8; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .62rem .85rem; border-radius: 10px; width: 100%;
  transition: background .15s ease; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.07); }
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) { display:none; }
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(18,183,106,.18); box-shadow: inset 3px 0 0 var(--emerald);
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#fff; font-weight:700; }
[data-testid="stSidebar"] [role="radiogroup"] p { font-size: .95rem; font-weight: 500; }
.wv-brand { display:flex; align-items:center; gap:.7rem; padding:.4rem .3rem 1.4rem; }
.wv-brand-mark { width:40px; height:40px; border-radius:12px; background:linear-gradient(145deg,#1FCB7C,#0B8F52);
  display:grid; place-items:center; box-shadow:0 6px 18px rgba(18,183,106,.35); }
.wv-brand-name { font-family:var(--serif); font-size:1.4rem; font-weight:600; color:#fff; line-height:1.05; }
.wv-brand-sub { font-size:.74rem; color:#9DB8A5; margin-top:2px; }
.wv-side-status { margin-top:1.6rem; padding:.9rem 1rem; border-radius:12px;
  background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); font-size:.82rem; line-height:1.5; }
.wv-side-status b { color:#fff; }
.wv-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--emerald);
  margin-right:.5rem; box-shadow:0 0 0 4px rgba(18,183,106,.2); }
.wv-dot.off { background:var(--red); box-shadow:0 0 0 4px rgba(194,73,61,.2); }
.wv-side-foot { margin-top:1rem; font-size:.72rem; color:#7E9A87; }
 
/* type */
h1, h2, h3 { font-family: var(--serif); color: var(--charcoal); letter-spacing: -.01em; }
.wv-page-title { font-family:var(--serif); font-size:2.15rem; font-weight:600; margin:0; color:var(--forest); line-height:1.15; }
.wv-page-sub { color:var(--ink-2); font-size:1.02rem; margin:.45rem 0 1.6rem; max-width:62ch; line-height:1.55; }
.wv-h3 { font-family:var(--serif); font-size:1.12rem; font-weight:600; color:var(--forest); margin:.2rem 0 .8rem; }
.wv-h2 { font-family:var(--serif); font-size:1.35rem; font-weight:600; color:var(--forest); margin:2rem 0 .8rem; }
 
/* hero */
.wv-hero { position:relative; overflow:hidden; border-radius:22px; padding:2.6rem 2.8rem;
  background: radial-gradient(1200px 400px at 85% -10%, #1B5239 0%, transparent 60%), var(--forest);
  color:#fff; display:grid; grid-template-columns: 1.35fr 1fr; gap:2rem; align-items:center;
  box-shadow:0 18px 40px rgba(14,43,30,.22); }
.wv-hero::before { content:""; position:absolute; inset:0; background-image:__FURROW__; background-size:cover;
  opacity:.16; -webkit-mask-image:linear-gradient(90deg,transparent 15%,#000 75%); mask-image:linear-gradient(90deg,transparent 15%,#000 75%); }
.wv-hero > * { position:relative; }
.wv-hero h1 { font-family:var(--serif); color:#fff; font-size:2.7rem; font-weight:600; line-height:1.1; margin:0 0 1rem; }
.wv-hero p { color:#C7DACB; font-size:1.05rem; line-height:1.6; max-width:52ch; margin:0; }
.wv-model-card { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14);
  border-radius:16px; padding:1.3rem 1.4rem; backdrop-filter: blur(6px); }
.wv-model-card .t { font-weight:700; font-size:.95rem; margin-bottom:.8rem; color:#fff; }
.wv-kv { display:flex; justify-content:space-between; gap:1rem; padding:.5rem 0; border-top:1px solid rgba(255,255,255,.1);
  font-size:.9rem; color:#C7DACB; }
.wv-kv b { color:#fff; font-weight:600; text-align:right; }
 
/* stat cards */
.wv-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:1rem; margin:1.4rem 0 .4rem; }
.wv-stat { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.25rem; }
.wv-stat.accent { background:linear-gradient(160deg,#fff,#EAF7F0); border-color:#BFE6D0; }
.wv-stat-label { color:var(--ink-2); font-size:.84rem; font-weight:600; }
.wv-stat-value { font-family:var(--serif); font-size:2.15rem; font-weight:600; color:var(--forest); line-height:1.2; margin-top:.2rem; }
.wv-stat-note { color:#7A857E; font-size:.78rem; margin-top:.15rem; }
 
/* panels */
.wv-panel { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:1.4rem 1.5rem; }
.wv-panel h3 { font-family:var(--serif); font-size:1.12rem; font-weight:600; margin:0 0 .7rem; color:var(--forest); }
.wv-panel p, .wv-panel li { color:var(--ink-2); font-size:.95rem; line-height:1.6; }
.wv-chip { display:inline-block; padding:.3rem .75rem; margin:0 .4rem .5rem 0; border-radius:999px;
  background:#E8F5EE; color:var(--emerald-d); font-weight:600; font-size:.85rem; border:1px solid #C6E8D6; }
.wv-steps { list-style:none; counter-reset:s; padding:0; margin:0; }
.wv-steps li { counter-increment:s; display:flex; gap:.9rem; padding:.7rem 0; border-top:1px solid var(--line); }
.wv-steps li:first-child { border-top:0; }
.wv-steps li::before { content:counter(s); flex:none; width:28px; height:28px; border-radius:50%;
  background:var(--forest); color:#fff; font-weight:700; font-size:.82rem; display:grid; place-items:center; }
.wv-steps b { color:var(--charcoal); }
.wv-empty { border:1.5px dashed #B8CDBE; border-radius:16px; padding:2.4rem 1.5rem; text-align:center;
  color:var(--ink-2); background:rgba(255,255,255,.55); }
.wv-empty b { display:block; font-family:var(--serif); font-size:1.15rem; color:var(--forest); margin-bottom:.3rem; }
.wv-band { display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.78rem; font-weight:700; }
.wv-note { border-left:3px solid var(--amber); background:#FBF5E6; padding:.8rem 1rem; border-radius:0 10px 10px 0;
  color:#5E4A16; font-size:.9rem; line-height:1.55; }
 
/* widgets */
.stButton > button, .stDownloadButton > button { border-radius:12px; font-weight:700; padding:.6rem 1.2rem;
  border:1px solid var(--line); transition: transform .12s ease, box-shadow .12s ease; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background:linear-gradient(180deg,#14C373,#0B9A57); border:0; color:#fff; box-shadow:0 6px 16px rgba(18,183,106,.28); }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #9DB8A5; background:#fff; border-radius:16px; padding:1.6rem; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--emerald); background:#F4FBF7; }
[data-testid="stImage"] img { border-radius:14px; border:1px solid var(--line); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight:600; padding:.55rem 1rem; }
.stTabs [aria-selected="true"] { color:var(--emerald-d); }
button:focus-visible, [role="radio"]:focus-visible { outline:3px solid rgba(18,183,106,.55) !important; outline-offset:2px; }
 
@media (max-width: 900px){
  .wv-hero { grid-template-columns:1fr; padding:1.8rem 1.5rem; }
  .wv-hero h1 { font-size:2rem; }
  .wv-page-title { font-size:1.7rem; }
}
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; } }
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
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#4A544E", titleColor="#4A544E", gridColor="#E6ECE6",
                        domainColor="#C9D5CB", labelFont="Manrope", titleFont="Manrope")
        .configure_legend(labelFont="Manrope", titleFont="Manrope")
    )
 
 
EMERALD_RANGE = ["#12B76A", "#0E2B1E", "#5F8F6B", "#D9A441", "#2F6F4F", "#9BCBA8", "#1E2421", "#C2493D"]
 
 
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
def sidebar(info: dict | None, error: str | None) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="wv-brand"><div class="wv-brand-mark">'
            '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10z"/>'
            '<path d="M2 21c0-3 1.9-5.5 6-7"/></svg></div>'
            '<div><div class="wv-brand-name">WeedVision</div>'
            '<div class="wv-brand-sub">AI weed detection for crops</div></div></div>',
            unsafe_allow_html=True,
        )
        st.radio("Navigation", list(NAV_LABELS), format_func=NAV_LABELS.get, key="nav",
                 label_visibility="collapsed")
        if info:
            status = (f'<span class="wv-dot"></span><b>Model ready</b><br>{esc(info["file"])} &middot; '
                      f'{len(info["classes"])} class{"es" if len(info["classes"]) != 1 else ""}')
        else:
            status = '<span class="wv-dot off"></span><b>Model unavailable</b><br>See the Dashboard for details'
        st.markdown(f'<div class="wv-side-status">{status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="wv-side-foot">WeedVision v{APP_VERSION}</div>', unsafe_allow_html=True)
 
 
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
        st.button("Analyse an image", type="primary", on_click=goto, args=("detect",), use_container_width=True)
    with right:
        st.markdown('<div class="wv-h3">Recent analyses</div>', unsafe_allow_html=True)
        if runs.empty:
            empty_state("No analyses yet", "Run your first detection and it will be listed here.")
        else:
            recent = runs.sort_values("Run", ascending=False).head(6)[
                ["Time", "Image", "Detections", "Avg confidence", "Coverage (%)"]]
            st.dataframe(
                recent, hide_index=True, use_container_width=True,
                column_config={
                    "Avg confidence": st.column_config.NumberColumn(format="percent"),
                    "Coverage (%)": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
 
 
def detection_table(det: pd.DataFrame) -> None:
    view = det.copy()
    st.dataframe(
        view, hide_index=True, use_container_width=True,
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
        st.image(original, use_container_width=True)
    with c2:
        st.markdown("**WeedVision result**")
        st.image(out["annotated"], use_container_width=True)
 
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
            st.altair_chart(style_chart(chart, 220), use_container_width=True)
 
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
                       "image/png", use_container_width=True, type="primary")
    d2.download_button("Report (HTML)", build_html_report(filename, out, info), f"weedvision_report_{stem}_{stamp}.html",
                       "text/html", use_container_width=True)
    d3.download_button("Detections (CSV)", det.to_csv(index=False), f"weedvision_{stem}_{stamp}.csv",
                       "text/csv", use_container_width=True, disabled=det.empty)
    d4.download_button("Data (JSON)", json.dumps(payload, indent=2), f"weedvision_{stem}_{stamp}.json",
                       "application/json", use_container_width=True)
 
 
def page_detect(model, info: dict) -> None:
    page_header("Detect weeds", "Add an image, choose how strict the model should be, then run the analysis.")
 
    tab_up, tab_cam = st.tabs(["Upload a file", "Use the camera"])
    with tab_up:
        uploaded = st.file_uploader("Choose a plant or field image", type=["jpg", "jpeg", "png", "webp"],
                                    label_visibility="collapsed", key="uploader")
    with tab_cam:
        shot = st.camera_input("Take a photo", label_visibility="collapsed", key="camera")
 
    source = uploaded or shot
    with st.expander("Detection settings", expanded=True):
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
            st.image(image, use_container_width=True)
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
            st.altair_chart(style_chart(chart), use_container_width=True)
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
            st.altair_chart(style_chart(chart), use_container_width=True)
 
    h2("Image by image")
    c, d = st.columns(2, gap="large")
    runs_plot = runs.assign(Label=runs["Run"].astype(str))
    with c:
        st.markdown("**Detections per image**")
        chart = alt.Chart(runs_plot).mark_bar(color="#0E2B1E", cornerRadiusEnd=6, size=36).encode(
            x=alt.X("Label:N", sort=None, title="Run", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Detections:Q", title="Detections", axis=alt.Axis(tickMinStep=1)),
            tooltip=["Run", "Image", "Detections"])
        st.altair_chart(style_chart(chart), use_container_width=True)
    with d:
        st.markdown("**Image area covered (%)**")
        chart = alt.Chart(runs_plot).mark_line(color="#12B76A", point=alt.OverlayMarkDef(color="#0B8F52", size=70),
                                               strokeWidth=3).encode(
            x=alt.X("Label:N", sort=None, title="Run", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Coverage (%):Q", title="Coverage (%)"),
            tooltip=["Run", "Image", "Coverage (%)"])
        st.altair_chart(style_chart(chart), use_container_width=True)
 
    if not dets.empty and dets["Class"].nunique() > 1:
        st.markdown("**Confidence by class**")
        chart = alt.Chart(dets).mark_boxplot(extent="min-max", size=36).encode(
            x=alt.X("Class:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Confidence:Q", title="Confidence", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Class:N", scale=alt.Scale(range=EMERALD_RANGE), legend=None))
        st.altair_chart(style_chart(chart), use_container_width=True)
 
    h2("Session log")
    log = runs.sort_values("Run", ascending=False)
    st.dataframe(
        log, hide_index=True, use_container_width=True,
        column_config={
            "Avg confidence": st.column_config.NumberColumn(format="percent"),
            "Threshold": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    x1, x2, x3, _ = st.columns([1, 1, 1, 2])
    x1.download_button("Session log (CSV)", log.to_csv(index=False), "weedvision_session_log.csv", "text/csv",
                       use_container_width=True, type="primary")
    x2.download_button("All detections (CSV)", dets.to_csv(index=False), "weedvision_all_detections.csv", "text/csv",
                       use_container_width=True, disabled=dets.empty)
    if x3.button("Clear session data", use_container_width=True):
        st.session_state["runs"], st.session_state["dets"], st.session_state["last"] = [], [], None
        st.rerun()
    st.caption("Analytics are kept in memory for this browser session only. Nothing is stored on the server.")
 
 
def training_metrics() -> dict | None:
    """Read final-epoch metrics from an Ultralytics results.csv, if one is shipped."""
    if not TRAINING_CSV.exists():
        return None
    try:
        df = pd.read_csv(TRAINING_CSV)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        wanted = {
            "metrics/precision(B)": "Precision", "metrics/recall(B)": "Recall",
            "metrics/mAP50(B)": "mAP@50", "metrics/mAP50-95(B)": "mAP@50-95",
        }
        found = {label: float(last[col]) for col, label in wanted.items() if col in df.columns}
        if not found:
            return None
        found["Epochs"] = int(last["epoch"]) + 1 if "epoch" in df.columns else len(df)
        return found
    except Exception:
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
            f'<div class="wv-kv" style="color:#4A544E;border-color:#E6ECE6"><span>Weights file</span><b style="color:#1E2421">{esc(info["file"])}</b></div>'
            f'<div class="wv-kv" style="color:#4A544E;border-color:#E6ECE6"><span>File size</span><b style="color:#1E2421">{info["size_mb"]:.1f} MB</b></div>'
            f'<div class="wv-kv" style="color:#4A544E;border-color:#E6ECE6"><span>Task</span><b style="color:#1E2421">{esc(info["task"].capitalize())}</b></div>'
            f'<div class="wv-kv" style="color:#4A544E;border-color:#E6ECE6"><span>Parameters</span><b style="color:#1E2421">{params}</b></div>'
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
            st.markdown(
                '<div class="wv-note"><b>Training metrics are not bundled.</b> Add the <code>results.csv</code> from your '
                'Ultralytics training run next to <code>app.py</code> and precision, recall and mAP will appear here automatically.</div>',
                unsafe_allow_html=True,
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
 
    sidebar(info, error)
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
 
