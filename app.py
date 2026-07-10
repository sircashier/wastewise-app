import os
# TF and torch each bundle their own OpenMP runtime; running both in one process
# (YOLO + ResNet) segfaults on Linux unless threading is capped BEFORE either imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from ultralytics import YOLO
import json

st.set_page_config(
    page_title="WasteWise | Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  /* ── Field Manual design tokens ── */
  :root {
    --paper: #f2efe6; --paper-card: #fbfaf5;
    --ink: #20241d; --ink-soft: #4a4f42; --ink-mute: #5c6152;
    --line: #cfcbbd; --accent: #3f7d3f; --num: #9a9a8a;
    --mono: 'IBM Plex Mono', monospace;
    --serif: 'Fraunces', serif;
    --ease: cubic-bezier(.22,.9,.35,1);
  }

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
  .stApp { background: var(--paper); }
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { display: none !important; }
  [data-testid="stSidebar"] { display: none; }
  body { overflow-x: hidden; }
  h1, h2, h3, h4 { font-family: var(--serif); color: var(--ink); }
  /* Style-only and dead-wrapper markdown blocks still occupy flex-gap slots —
     remove them from the flow (their <style> tags keep applying regardless). */
  [data-testid="stElementContainer"]:has(style),
  [data-testid="stElementContainer"]:has(.page-content:empty) { display: none !important; }
  /* Constrain content to a readable centered column */
  .block-container { padding: 0 1.7rem 4rem !important; max-width: 920px !important; margin: 0 auto !important; }

  @keyframes riseIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes fillIn { from { transform: scaleX(0); } to { transform: scaleX(1); } }

  /* Cap displayed photos so tall portrait shots don't stretch the page */
  [data-testid="stImage"] img { max-height: 52vh; object-fit: contain; border: 1px solid var(--line); border-radius: 2px; }
  /* Hide the zero-height autoscroll helper iframe's container */
  [data-testid="stElementContainer"]:has(iframe[height="0"]) { display: none !important; }

  /* ── Hero: editorial headline + mono meta block ── */
  .fm-hero {
    display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 20px;
    padding: 44px 0 30px; border-bottom: 1px solid var(--line);
    animation: riseIn .4s var(--ease) both;
  }
  .fm-eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }
  /* !important needed: Streamlit's own stMarkdown h2 rules outrank the class */
  .fm-title { font-family: var(--serif) !important; font-weight: 900 !important; font-size: 52px !important; line-height: .98 !important; letter-spacing: -.02em !important; color: var(--ink) !important; margin: 0 !important; padding: 0 !important; }
  .fm-title em { font-style: italic; font-weight: 400 !important; color: var(--accent); }
  .fm-title a { display: none !important; }
  .fm-meta { text-align: right; font-family: var(--mono); font-size: 10.5px; line-height: 1.7; color: var(--ink-mute); text-transform: uppercase; }
  .fm-meta b { color: var(--ink); }
  .fm-hero.compact { padding: 16px 0 12px; }
  .fm-hero.compact .fm-title { font-size: 24px !important; }
  .fm-hero.compact .fm-eyebrow { margin-bottom: 4px; }
  .fm-hero.compact .fm-meta { display: none; }

  /* ── Uploader: dashed spec sheet ── */
  .upload-label { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-mute); margin: 1.4rem 0 0.5rem; }
  [data-testid="stFileUploader"] {
    border: 1.5px dashed #7d8471 !important; border-radius: 0 !important;
    background: var(--paper-card) !important; padding: 1rem !important;
    transition: border-color .2s var(--ease);
  }
  [data-testid="stFileUploader"]:hover { border-color: var(--ink) !important; }
  [data-testid="stFileUploader"] button {
    cursor: pointer !important; background: var(--ink) !important; color: var(--paper) !important;
    font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: .12em !important;
    text-transform: uppercase !important; border: none !important; border-radius: 0 !important;
    padding: 12px 22px !important;
  }
  [data-testid="stFileUploader"] button:hover { background: var(--accent) !important; }

  [data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 0 !important; background: var(--paper-card); overflow: hidden; }
  [data-testid="stExpander"] summary { cursor: pointer; font-family: var(--mono); transition: background .18s var(--ease); }
  [data-testid="stExpander"] summary:hover { background: var(--paper); }

  /* ── Result cards: quiet spec cards, one green ── */
  .result-card {
    background: var(--paper-card); border: 1px solid var(--line); border-left: 3px solid var(--accent);
    border-radius: 0; padding: 1.5rem 1.5rem 1.4rem; margin-top: 1rem; text-align: left;
    animation: riseIn .32s var(--ease) both;
  }
  .result-card .label-eyebrow { font-family: var(--mono); font-size: 10.5px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-mute); margin-bottom: 0.5rem; }
  .result-card .result-icon { font-size: 1.6rem; margin-bottom: 0.3rem; line-height: 1.1; }
  .result-card .result-category { font-family: var(--serif); font-weight: 600; font-size: 1.55rem; color: var(--ink); margin-bottom: 0.4rem; line-height: 1.1; }
  .result-card .result-instruction { font-size: 0.86rem; color: var(--ink-soft); line-height: 1.6; }
  .obj-chip { display: inline-block; background: transparent; border: 1px solid var(--line); border-radius: 0; padding: 0.2rem 0.7rem; font-family: var(--mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-mute); margin-bottom: 0.7rem; }

  .conf-wrap { max-width: 260px; margin: 0.3rem 0 0.9rem; }
  .conf-track { height: 5px; background: rgba(32,36,29,.12); overflow: hidden; }
  .conf-fill { height: 100%; background: var(--accent); transform-origin: left; animation: fillIn .6s .15s var(--ease) both; }
  .conf-label { font-family: var(--mono); font-size: 10.5px; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-mute); margin-top: 0.4rem; }

  /* One green: category cards share the same quiet spec look */
  .card-none { border: 1.5px dashed #7d8471; border-left: 1.5px dashed #7d8471; }

  .fallback-note { background: var(--paper-card); border: 1px solid var(--line); border-left: 3px solid var(--ink); border-radius: 0; padding: 0.75rem 1rem; font-size: 0.82rem; color: var(--ink-soft); margin-top: 0.8rem; line-height: 1.55; animation: riseIn .3s var(--ease) both; }

  /* ── Numbered category reference table (signature element) ── */
  .fm-cats { margin-top: 2.2rem; }
  .fm-cats-title { font-family: var(--mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-mute); padding-bottom: 10px; border-bottom: 1.5px solid var(--ink); }
  .fm-cat { display: grid; grid-template-columns: 38px 190px 1fr; gap: 16px; padding: 15px 4px; border-bottom: 1px solid var(--line); align-items: baseline; }
  .fm-cat .num { font-family: var(--mono); font-size: 12px; color: var(--num); }
  .fm-cat .nm { font-family: var(--serif); font-weight: 600; font-size: 16px; color: var(--ink); }
  .fm-cat .nm span { display: block; font-family: var(--mono); font-size: 9.5px; font-weight: 400; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin-top: 2px; }
  .fm-cat .dsc { font-size: 13px; color: var(--ink-soft); line-height: 1.5; }

  .empty-state { text-align: center; border: 1.5px dashed var(--line); border-radius: 0; padding: 2.2rem 1rem; color: var(--ink-mute); background: var(--paper-card); font-size: 0.88rem; }
  .empty-state .es-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }

  /* ── Steps / team / stats / page chrome ── */
  .step-card { background: var(--paper-card); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 0; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
  .step-number { font-family: var(--mono); font-size: 10.5px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.2rem; }
  .step-title { font-family: var(--serif); font-weight: 600; font-size: 1.15rem; color: var(--ink); margin-bottom: 0.4rem; }
  .step-body { font-size: 0.87rem; color: var(--ink-soft); line-height: 1.6; }

  .team-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 2rem; }
  .team-card { background: var(--paper-card); border: 1px solid var(--line); border-radius: 0; padding: 1.25rem 1.4rem; }
  .team-role { font-family: var(--mono); font-size: 10px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.3rem; }
  .team-name { font-family: var(--serif); font-weight: 600; font-size: 1.1rem; color: var(--ink); margin-bottom: 0.2rem; }
  .team-desc { font-size: 0.78rem; color: var(--ink-mute); line-height: 1.5; }

  .stat-card { background: var(--paper-card); border: 1px solid var(--line); border-radius: 0; padding: 1.05rem 1.2rem; text-align: center; animation: riseIn .32s var(--ease) both; }
  .stat-number { font-family: var(--serif); font-weight: 900; font-size: 1.8rem; color: var(--ink); }
  .stat-label { font-family: var(--mono); font-size: 10px; color: var(--ink-mute); text-transform: uppercase; letter-spacing: .1em; }

  .page-title { font-family: var(--serif); font-weight: 900; font-size: 2.1rem; letter-spacing: -.01em; color: var(--ink); margin: 2.5rem 0 0.2rem; animation: riseIn .35s var(--ease) both; }
  .page-sub { font-family: var(--mono); font-size: 10.5px; color: var(--ink-mute); text-transform: uppercase; letter-spacing: .12em; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--line); }

  .app-footer { text-align: center; font-family: var(--mono); font-size: 10px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-mute); margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--line); }
  .tech-pill { display: inline-block; background: var(--paper-card); border: 1px solid var(--line); color: var(--ink-soft); border-radius: 0; padding: 0.28rem 0.8rem; font-family: var(--mono); font-size: 10.5px; letter-spacing: .04em; margin: 0.2rem; }
  .sdg-badge { display: inline-block; background: var(--ink); color: var(--paper); border-radius: 0; padding: 0.32rem 0.95rem; font-family: var(--mono); font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; margin-right: 0.5rem; margin-top: 0.5rem; }

  .stButton button, [data-testid="stDownloadButton"] button {
    cursor: pointer !important; font-family: var(--mono) !important; font-size: 11px !important;
    letter-spacing: .1em !important; text-transform: uppercase !important; border-radius: 0 !important;
    transition: background .18s var(--ease), color .18s var(--ease) !important;
  }
  .stButton button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

  @media (max-width: 680px) {
    .team-grid { grid-template-columns: 1fr; }
    .fm-hero { grid-template-columns: 1fr; padding-top: 28px; }
    .fm-meta { text-align: left; }
    .fm-title { font-size: 34px !important; }
    .fm-cat { grid-template-columns: 30px 1fr; }
    .fm-cat .dsc { grid-column: 2; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
  }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ── Category info (crops dataset uses 'Residual'; keep long key as alias) ──
_NONBIO = ("🗑️", "card-nonbio", "Residual / Non-Biodegradable",
           "Throw this in the non-biodegradable bin. This waste does not decompose.")
CATEGORY_INFO = {
    "Biodegradable":                ("🌱", "card-bio",        "Biodegradable",  "Throw this in the biodegradable bin. This waste will be composted or naturally decomposed."),
    "Recyclable":                   ("♻️", "card-recyclable",  "Recyclable",     "Throw this in the recyclable bin. This waste can be processed and reused."),
    "Residual":                     _NONBIO,
    "Residual - Non-Biodegradable": _NONBIO,
    "E-Waste":                      ("⚡", "card-ewaste",      "E-Waste",        "Bring this to a designated e-waste drop-off point. Do NOT throw in regular bins."),
    "Infectious Waste":             ("🧪", "card-infectious",  "Infectious Waste", "This requires special handling. Place in a sealed bag and bring to the proper infectious waste disposal facility."),
}

# Display-only reference table (numbers/Tagalog are presentation; CATEGORY_INFO keys unchanged)
WASTE_GUIDE = [
    ("01", "Biodegradable",                "Nabubulok",       "Food scraps and leftovers, leaves and plant matter."),
    ("02", "Recyclable",                   "Mababawi",        "Plastic bottles and containers, glass bottles and jars, metal and aluminum cans, paper, cardboard."),
    ("03", "Residual / Non-Biodegradable", "Hindi Nabubulok", "Cigarette butts, snack wrappers and sachets, sanitary products and diapers, styrofoam, cloth and textile waste."),
    ("04", "E-Waste",                      "Elektroniko",     "Phones, TVs and monitors, cables and wires, circuit boards."),
    ("05", "Infectious Waste",             "Nakakahawa",      "Face masks, medical gloves, gauze and bandages, syringes."),
]

# ── What category each YOLO object class implies (from yolo/category_map.json) ──
#    Used only to detect stage disagreement — ResNet50 always makes the final call.
OBJ_IMPLIES = {
    "food": "Biodegradable", "vegetation": "Biodegradable",
    "cords wires": "E-Waste", "ewaste misc": "E-Waste", "pcie board": "E-Waste",
    "phone": "E-Waste", "tv": "E-Waste",
    "gauze": "Infectious Waste", "glove": "Infectious Waste",
    "mask": "Infectious Waste", "syringe": "Infectious Waste",
    "cardboard": "Recyclable", "glass": "Recyclable", "metal can": "Recyclable",
    "paper": "Recyclable", "plastic": "Recyclable",
    "cigarette butt": "Residual", "multilayer plastic": "Residual",
    "sanitary product": "Residual", "styrofoam": "Residual", "textile": "Residual",
}

# ── Load the two-stage pipeline ──
#    Stage 1: YOLO11 finds the waste object and crops it (removes the background).
#    Stage 2: ResNet50 classifies the crop into an RA 9003 category.
DETECT_CONF = 0.25   # low threshold: better to find something and let ResNet decide
MAX_OBJECTS = 3      # classify at most this many detected objects per photo
LOW_CONF = 0.60      # below this, tell the user the model is unsure
UNRECOGNIZED_CONF = 0.60  # fallback below this = treat as "no supported item recognized"

@st.cache_resource
def load_models():
    localizer = YOLO("capson_yolo11s_best.pt")
    classifier = tf.keras.models.load_model("waste_classifier_crops.keras", compile=False)
    with open("labels.json") as f:
        labels = json.load(f)
    return localizer, classifier, labels

localizer, classifier, CLASSES = load_models()

def classify_crop(crop: Image.Image):
    # This model does NOT have preprocessing baked in — apply it here.
    arr = np.array(crop.resize((224, 224), Image.BILINEAR)).astype("float32")
    arr = preprocess_input(np.expand_dims(arr, axis=0))
    preds = classifier.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx])

def analyze_image(img: Image.Image):
    """Returns (annotated_image_or_None, [results]).
    Each result: {crop, category, confidence, object_name, fallback}"""
    # imgsz=1024 + test-time augmentation: better recall on hard objects
    # (transparent bottles, thin items) at the cost of ~2s per photo on CPU
    res = localizer.predict(img, conf=DETECT_CONF, imgsz=1024, augment=True, verbose=False)[0]

    if len(res.boxes) == 0:
        # Fallback: no object localized — classify the whole photo (old behavior)
        category, conf = classify_crop(img)
        return None, [{"crop": img, "category": category, "confidence": conf,
                       "object_name": None, "fallback": True}]

    annotated = Image.fromarray(res.plot()[..., ::-1])  # BGR -> RGB
    order = res.boxes.conf.argsort(descending=True)[:MAX_OBJECTS]
    results = []
    for j in order:
        box = res.boxes[int(j)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img.crop((x1, y1, x2, y2))
        category, conf = classify_crop(crop)
        results.append({
            "crop": crop, "category": category, "confidence": conf,
            "object_name": res.names[int(box.cls)].replace("_", " "),
            "fallback": False,
        })
    return annotated, results

# ══════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════
st.markdown("""
<style>
  [data-testid="stHorizontalBlock"]:first-of-type {
    background: var(--paper) !important;
    padding: 0.9rem 2.1rem 0.55rem !important;
    align-items: center !important;
    border-bottom: 1.5px solid var(--ink);
    /* full-bleed the navbar out of the centered content column
       (min-width beats flex-shrink; width alone gets shrunk back) */
    width: 100vw !important;
    min-width: 100vw !important;
    position: relative;
    left: 50%;
    transform: translateX(-50%);
    margin: 0 !important;
  }
  [data-testid="stHorizontalBlock"]:first-of-type button { white-space: nowrap !important; }
  @media (max-width: 780px) {
    [data-testid="stHorizontalBlock"]:first-of-type { padding: 0.5rem 0.8rem 0.4rem !important; }
    [data-testid="stHorizontalBlock"]:first-of-type button { font-size: 9.5px !important; padding: 0.3rem 0.3rem !important; }
  }
  [data-testid="stHorizontalBlock"]:first-of-type > div:first-child p {
    font-family: 'Fraunces', serif !important;
    font-size: 1.25rem !important;
    color: var(--ink) !important;
    font-weight: 900 !important;
    letter-spacing: -.01em;
    margin: 0 !important;
    padding: 0 !important;
  }
  [data-testid="stHorizontalBlock"]:first-of-type button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--ink-mute) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    width: 100% !important;
    box-shadow: none !important;
    cursor: pointer !important;
    transition: color .18s cubic-bezier(.22,.9,.35,1) !important;
  }
  [data-testid="stHorizontalBlock"]:first-of-type button:hover {
    background: transparent !important;
    color: var(--ink) !important;
  }
  [data-testid="stHorizontalBlock"]:first-of-type button:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
  }
</style>
""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4, nav5 = st.columns([3, 1, 1.3, 1, 1.2])
with nav1:
    st.markdown("**WasteWise**")
with nav2:
    if st.button("Home"):
        st.session_state.page = "Home"
        st.rerun()
with nav3:
    if st.button("How It Works"):
        st.session_state.page = "How It Works"
        st.rerun()
with nav4:
    if st.button("History"):
        st.session_state.page = "History"
        st.rerun()
with nav5:
    if st.button("About Us"):
        st.session_state.page = "About"
        st.rerun()

page = st.session_state.page

# Highlight the active page in the navbar (nav columns: 2=Home, 3=How It Works, 4=History, 5=About)
_nav_col = {"Home": 2, "How It Works": 3, "History": 4, "About": 5}[page]
st.markdown(f"""
<style>
  [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child({_nav_col}) button {{
    color: #20241d !important;
    border-bottom: 2px solid #3f7d3f !important;
    font-weight: 500 !important;
  }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════
if page == "Home":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    # Shrink the hero once a photo is loaded so results sit higher on the page
    _has_upload = st.session_state.get("upload") is not None
    _hero_cls = "fm-hero compact" if _has_upload else "fm-hero"
    st.markdown(f"""
    <div class="{_hero_cls}">
      <div>
        <div class="fm-eyebrow">Smart Waste Sorting · RA 9003</div>
        <h2 class="fm-title">Know where<br>it <em>belongs.</em></h2>
      </div>
      <div class="fm-meta">Margus Solutions<br>Philippines · 2026<br><b>5 categories</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-label">Upload a waste image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed", key="upload")

    with st.expander("📸 Photo tips for best results"):
        st.markdown("""
- **Get close** — the item should fill a good part of the photo.
- **1–3 items max**, spaced apart. Piles of mixed trash confuse the detector.
- **Show the whole item** — not inside a bag or mostly hidden.
- **Any background is fine** — table, grass, floor; the detector crops it away.
- Decent light and focus help; studio conditions are not needed.
- Works on common waste items (bottles, cans, paper, food, e-waste, medical...).
  Unusual objects it never trained on may get unreliable answers.
        """)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Locating and classifying waste..."):
            annotated, results = analyze_image(img)

        n_found = sum(1 for r in results if not r["fallback"])
        unrecognized = (n_found == 0 and results
                        and results[0]["confidence"] < UNRECOGNIZED_CONF)
        if n_found:
            st.toast(f"Found and classified {n_found} object(s)", icon="✅")
        elif unrecognized:
            st.toast("No supported waste item recognized", icon="❓")
        else:
            st.toast("No object localized — classified the whole photo", icon="⚠️")

        st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            if annotated is not None:
                st.image(annotated, width="stretch", caption="Detected object(s)")
            else:
                st.image(img, width="stretch", caption="Uploaded image")

        with col2:
            if unrecognized:
                st.markdown(f"""
                <div class="result-card card-none">
                  <div class="result-icon">🔍</div>
                  <div class="result-category">No supported item recognized</div>
                  <div class="result-instruction">
                    WasteWise found <strong>none of the waste items it is trained on</strong> in this
                    photo, and the whole-image check isn't confident enough to guess
                    (best guess was only {results[0]["confidence"]*100:.0f}%).<br><br>
                    Please upload a clear, close-up photo of one of the <strong>accepted items</strong> —
                    see the category list at the bottom of this page.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for r in results:
                    icon, card_css, display_name, instruction = CATEGORY_INFO[r["category"]]
                    obj_chip = (f'<div class="obj-chip">🔍 detected: {r["object_name"]}</div>'
                                if r["object_name"] else "")
                    st.markdown(f"""
                    <div class="result-card {card_css}">
                      {obj_chip}
                      <div class="label-eyebrow">This waste belongs to</div>
                      <div class="result-category">{display_name}</div>
                      <div class="conf-wrap">
                        <div class="conf-track"><div class="conf-fill" style="width:{r["confidence"]*100:.1f}%"></div></div>
                        <div class="conf-label">Confidence {r["confidence"]*100:.1f}%</div>
                      </div>
                      <div class="result-instruction">{instruction}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if r["confidence"] < LOW_CONF:
                        st.markdown("""
                        <div class="fallback-note">
                          ⚠️ <strong>Low confidence</strong> — the model is unsure about this one.
                          Please verify manually before disposing.
                        </div>
                        """, unsafe_allow_html=True)

                    # Stage disagreement: the detector's object label implies a different
                    # category than the classifier chose — flag it for the user.
                    implied = OBJ_IMPLIES.get(r["object_name"] or "", None)
                    final_cat = "Residual" if r["category"].startswith("Residual") else r["category"]
                    if implied is not None and implied != final_cat:
                        implied_disp = CATEGORY_INFO[implied][2]
                        st.markdown(f"""
                        <div class="fallback-note">
                          🤔 <strong>The two models disagree.</strong> The detector thinks this object
                          is <strong>{r["object_name"]}</strong> (usually {implied_disp}), but the
                          classifier assigned <strong>{display_name}</strong>. One of them is wrong —
                          please double-check this item.
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state.history.append({
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Filename": uploaded.name,
                        "Object": r["object_name"] or "whole image",
                        "Category": f"{icon} {display_name}",
                        "Instruction": instruction,
                    })

                if results and results[0]["fallback"]:
                    st.markdown("""
                    <div class="fallback-note">
                      ⚠️ No distinct object was localized — the whole image was classified instead.
                      For best results, take the photo closer to a single waste item.
                    </div>
                    """, unsafe_allow_html=True)

        # Bring the results into view (tall photos otherwise push them below the fold)
        components.html("""
        <script>
          const el = window.parent.document.querySelector('#results-anchor');
          if (el) setTimeout(() => el.scrollIntoView({behavior: 'smooth', block: 'start'}), 250);
        </script>
        """, height=0)

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="es-icon">♻️</div>
          Upload a photo of waste to find out where to throw it.<br>
          Any background is fine — just get close to the item.
        </div>
        """, unsafe_allow_html=True)

    _rows = "".join(f"""
      <div class="fm-cat">
        <div class="num">{num}</div>
        <div class="nm">{title}<span>{tagalog}</span></div>
        <div class="dsc">{desc}</div>
      </div>""" for num, title, tagalog, desc in WASTE_GUIDE)
    st.markdown(f"""
    <div class="fm-cats">
      <div class="fm-cats-title">What the classifier accepts · RA 9003</div>
      {_rows}
    </div>
    """, unsafe_allow_html=True)
    st.caption("The classifier is trained on the specific items listed above. "
               "Other objects (e.g. batteries, ceramics, light bulbs) are outside its "
               "training scope and may give unreliable results.")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════
elif page == "How It Works":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Two-Stage Pipeline · YOLO11 Object Localization + ResNet50 Classification</div>', unsafe_allow_html=True)

    steps = [
        ("Step 01", "Image Input", "The user uploads a photo of a waste item — an everyday phone photo works: any background, any lighting, even multiple items in frame."),
        ("Step 02", "Object Localization (YOLO11)", "A YOLO11 detector, fine-tuned on 21 waste object types, locates each waste item in the photo and draws a tight bounding box around it. The image is then <strong>cropped to just the object</strong> — removing the background entirely."),
        ("Step 03", "Why Cropping Matters", "Our first model saw whole images and learned <em>background shortcuts</em> (e.g. concrete floor = Recyclable) instead of the objects themselves. It scored ~95% on curated test images but failed on real photos. Cropping removes the background, forcing the classifier to judge the <strong>object</strong>."),
        ("Step 04", "Classification (ResNet50)", "Each crop passes through a 50-layer deep residual network pretrained on ImageNet and fine-tuned on our background-free waste crops via two-stage transfer learning: first the classification head is trained with the base frozen, then the upper layers are unfrozen and fine-tuned at a low learning rate."),
        ("Step 05", "Category & Disposal Instruction", "The classifier outputs one of the 5 Philippine RA 9003 waste categories, and the app tells the user exactly where to dispose of the item. If no object can be localized, the app falls back to classifying the whole image and says so."),
    ]
    for num, title, body in steps:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-number">{num}</div>
          <div class="step-title">{title}</div>
          <div class="step-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### SDG Alignment")
    st.markdown("""
    <span class="sdg-badge">🏙️ SDG 11 — Sustainable Cities & Communities</span>
    <span class="sdg-badge">♻️ SDG 12 — Responsible Consumption & Production</span>
    <br><br>
    """, unsafe_allow_html=True)
    st.caption("This system supports Republic Act No. 9003 and promotes proper waste segregation in Philippine communities.")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════
elif page == "History":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Classification History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Session log of all classified waste images</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.markdown("""
        <div style="text-align:center; color:#A0B8AC; padding:3rem 0; font-size:0.9rem;">
          <div style="font-size:2.5rem; margin-bottom:0.5rem;">🗂️</div>
          No classifications yet. Go to <strong>Home</strong> and upload a waste image first.
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(history)
        recyclable = sum(1 for h in history if "Recyclable" in h["Category"])
        bio = sum(1 for h in history if "Biodegradable" in h["Category"] and "Non" not in h["Category"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{total}</div><div class="stat-label">Total Classified</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{recyclable}</div><div class="stat-label">Recyclable</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{bio}</div><div class="stat-label">Biodegradable</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        df = pd.DataFrame([{"Time": h["Time"], "File": h["Filename"], "Object": h.get("Object", "—"), "Category": h["Category"]} for h in history])
        st.dataframe(df, width="stretch", hide_index=True)

        dl_col, clr_col = st.columns([1, 1])
        with dl_col:
            st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                               "wastewise_history.csv", "text/csv")
        with clr_col:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════
elif page == "About":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">About WasteWise</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Margus Solutions · Web Science Capstone Project · 2026</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:0.92rem; color:#3D5C4A; line-height:1.75; margin-bottom:1.5rem;">
      <strong>WasteWise</strong> is a deep learning-based waste classification system developed by
      <strong>Margus Solutions</strong> as a Web Science Capstone Project. It uses a two-stage pipeline — a YOLO11 detector that localizes and crops the waste object,
      followed by a ResNet50 convolutional neural network trained via two-stage transfer learning on background-free
      crops — to classify waste into five Philippine waste categories based on RA 9003.
      The system aims to address the challenge of improper waste segregation in the Philippines,
      where a significant portion of households still struggle to correctly identify and dispose of waste.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("#### Our Team · Margus Solutions")
    st.markdown("""
    <div class="team-grid">
      <div class="team-card">
        <div class="team-role">CEO · Chief Executive Officer</div>
        <div class="team-name">Cablao, Matt Lawrence M.</div>
        <div class="team-desc">Leads overall project direction, stakeholder communication, and capstone deliverables.</div>
      </div>
      <div class="team-card">
        <div class="team-role">CTO · Chief Technology Officer</div>
        <div class="team-name">Cablao, Matt Lawrence M.</div>
        <div class="team-desc">Oversees the YOLO11 + ResNet50 pipeline, transfer learning, and system deployment.</div>
      </div>
      <div class="team-card">
        <div class="team-role">CFO · Chief Financial Officer</div>
        <div class="team-name">Benico, Joemark Vincent</div>
        <div class="team-desc">Manages project resources, tooling costs, and budget for compute and dataset acquisition.</div>
      </div>
      <div class="team-card">
        <div class="team-role">COO · Chief Operating Officer</div>
        <div class="team-name">Llagas, Franzys Danille</div>
        <div class="team-desc">Coordinates team operations, timelines, and ensures milestones are met on schedule.</div>
      </div>
      <div class="team-card">
        <div class="team-role">CMO · Chief Marketing Manager</div>
        <div class="team-name">Francisco, Marcus John</div>
        <div class="team-desc">Handles project presentation, documentation design, and stakeholder-facing materials.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Technology Stack")
    st.markdown("""
    <span class="tech-pill">🎯 YOLO11 (Ultralytics)</span>
    <span class="tech-pill">🧠 ResNet50</span>
    <span class="tech-pill">🔥 TensorFlow / Keras</span>
    <span class="tech-pill">🐍 Python 3.12</span>
    <span class="tech-pill">🌐 Streamlit</span>
    <span class="tech-pill">☁️ Google Colab</span>
    <span class="tech-pill">🔄 Transfer Learning</span>
    <span class="tech-pill">🏷️ Roboflow Annotation</span>
    <br><br>
    """, unsafe_allow_html=True)
    st.caption("Web Science Capstone Project · 2026 · YOLO11 + ResNet50 · TensorFlow/Keras · Streamlit")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
  WasteWise by Margus Solutions · Web Science Capstone Project 2026 &nbsp;|&nbsp; YOLO11 + ResNet50 · TensorFlow/Keras · Streamlit
</div>
""", unsafe_allow_html=True)
