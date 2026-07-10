---
title: WasteWise
emoji: ♻️
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# WasteWise — Smart Waste Sorting Assistant
### by Margus Solutions · Web Science Capstone Project 2026

A two-stage deep-learning web app that classifies a waste photo into one of 5
Philippine RA 9003 categories (Biodegradable, E-Waste, Infectious Waste,
Recyclable, Residual/Non-Biodegradable) and tells the user which bin it belongs in.

**Pipeline:** YOLO11 locates and crops the waste object (removing the background),
then ResNet50 classifies the crop. Built with Ultralytics + TensorFlow/Keras
(models) and Streamlit (web UI).

---

## What's in this Space

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit web app (two-stage pipeline) |
| `capson_yolo11s_best.pt` | Stage 1 — YOLO11 object localizer (finds + crops the waste item) |
| `waste_classifier_crops.keras` | Stage 2 — ResNet50 classifier trained on background-free crops |
| `labels.json` | The 5 category names, in the exact order the classifier was trained on |
| `.streamlit/config.toml` | Forces the light "Field Manual" theme — do not delete |
| `requirements.txt` | Pinned Python dependencies |
| `HOW_TO_RUN.txt` | Plain-text setup + troubleshooting guide (for local/offline running) |

## How to use the app
1. On the **Home** page, click **Browse files** and upload a photo of a waste item.
2. The app shows the detected object(s), the predicted category, a disposal
   instruction, and a confidence %. Up to 3 objects per photo are classified.
3. If no object can be localized, the app classifies the whole image and shows a note.
4. Other tabs: **How It Works**, **History** (this session's classifications), **About**.

## Note on accuracy
Weak spots that remain: heterogeneous e-waste ("misc gadgets"), thin objects
(cords/wires), and TVs may sometimes not be detected — the app then falls back
to whole-image classification. Confidence below ~60% means the model is unsure.
