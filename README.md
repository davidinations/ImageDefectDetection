# YoloWeb — Streamlit YOLO Object Detection

A simple **Streamlit** web app that runs **YOLO (Ultralytics)** object detection
on user-uploaded images. Supports multiple models, adjustable confidence/IoU,
and works locally or on Streamlit Community Cloud.

## _Demo App_

[davidinations-yolo.streamlit.app](https://davidinations-yolo.streamlit.app/)

## Features

- **Model selection** in the sidebar:
  - Pick a **default model** from the `models/` folder (auto-loaded).
  - Or **upload your own** `.pt` model file directly in the UI.
- **Sample-image buttons** in the sidebar: each button auto-loads a test image
  from `test_images/` that matches the selected default model (e.g.
  `sample1.pt` ↔ `sample1.png`). A button is enabled only when its matching
  model is selected.
- **Image upload**: JPG, PNG, BMP.
- **Confidence threshold** slider and **IoU threshold** slider.
- Shows the **original** and **annotated** image side by side.
- Results **table** with class, confidence, bounding box, and per-class counts.

## Getting started

### Install

```bash
pip install -r requirements.txt
```

### Add a default model

Place one or more `.pt` model files into the `models/` folder:

```text
models/
  best.pt
  my_model.pt
```

They will appear in the "default model" dropdown automatically. Without any
model in `models/`, you can still upload a model file from the UI.

> Model weights are **not committed** to git. Add `models/*.pt` to your
> `.gitignore` if using git.

### Run locally

```bash
streamlit run app.py
```

### Deploy to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Enable Streamlit Community Cloud on the repo.
3. Streamlit reads `requirements.txt` automatically — no build step.

## Configuration defaults

The default confidence and IoU values match the original Flask template's
`.env`:

| Parameter            | Default |
| -------------------- | ------- |
| Confidence threshold | 0.8     |
| IoU threshold        | 0.9     |

Both are adjustable via sliders in the app's sidebar.

## Project layout

```text
app.py            # Streamlit application
requirements.txt  # Pinned dependencies
models/           # Optional default .pt model(s)
test_images/      # Optional sample test images matched by model name
README.md         # This file
```
