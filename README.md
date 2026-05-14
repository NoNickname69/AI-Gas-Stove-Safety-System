# 🔥 AI Gas Stove Safety System

> **Real-time kitchen safety monitoring using YOLOv8 and OpenCV**  
> A computer vision system that detects unattended stoves and fires an alert before accidents happen.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Example Output](#example-output)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Why This Project Matters](#why-this-project-matters)

---

## Overview

The **AI Gas Stove Safety System** is a real-time computer vision application that monitors a kitchen environment through a webcam feed.

It uses **YOLOv8** for object detection and **OpenCV** for video processing to:
- Detect humans present in the kitchen
- Detect active flames on a gas stove
- Trigger an alert if a flame is detected but no person has been nearby for a configurable duration

This project demonstrates a complete **end-to-end AI/CV pipeline**: from raw webcam input, through neural network inference, rule-based safety reasoning, and real-time UI overlays.

---

## Problem Statement

Kitchen fires are one of the leading causes of residential fires worldwide. A significant proportion of cooking fires start when food or cookware is left unattended on an active burner.

**The question:** Can a low-cost camera + AI system serve as an always-on safety observer that detects this exact scenario and alerts the occupant before it becomes dangerous?

This project explores that idea with a working prototype.

---

## Features

| Feature | Status |
|---|---|
| Real-time webcam input | ✅ |
| YOLOv8 person detection | ✅ |
| Flame detection (HSV heuristic) | ✅ |
| Unattended stove logic with configurable timer | ✅ |
| Risk level display (NORMAL / HIGH) | ✅ |
| Bounding boxes + confidence scores | ✅ |
| FPS counter | ✅ |
| Console alert on HIGH risk | ✅ |
| Sound alert (optional) | ✅ |
| Modular, extensible codebase | ✅ |
| Telegram / email alerts | 🔜 Stubbed |
| Custom fire-detection model | 🔜 Planned |
| Smoke detection | 🔜 Planned |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py  (orchestrator)                  │
│                                                                   │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────────┐  │
│  │ OpenCV  │───▶│ detector │───▶│  logic  │───▶│     ui      │  │
│  │ webcam  │    │ (YOLOv8) │    │ (rules) │    │ (overlays)  │  │
│  └─────────┘    └──────────┘    └────┬────┘    └─────────────┘  │
│                                       │                           │
│                                  ┌────▼────┐                     │
│                                  │ alerts  │                     │
│                                  │(console │                     │
│                                  │ sound   │                     │
│                                  │ future) │                     │
│                                  └─────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow per frame:**

1. **OpenCV** captures a raw BGR frame from the webcam.
2. **`detector.py`** runs the frame through YOLOv8 and the colour-based fire heuristic. Returns a list of detection dictionaries: `{label, confidence, bbox}`.
3. **`logic.py`** receives `flame_detected` and `person_detected` booleans, tracks timestamps, and returns a `risk_level` + `warning_message`.
4. **`ui.py`** draws bounding boxes, the status panel, and the warning banner onto the frame.
5. **`alerts.py`** fires console/sound alerts when `risk_level == "HIGH"`, with a cooldown to prevent spam.
6. The annotated frame is displayed in an OpenCV window.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Language |
| OpenCV | 4.8+ | Webcam capture, drawing, display |
| Ultralytics YOLOv8 | 8.0+ | Object detection (person) |
| NumPy | 1.24+ | Array operations, HSV masking |

No web framework. No database. No cloud. Runs entirely on your local machine.

---

## Project Structure

```
AI-Gas-Stove-Safety-System/
│
├── src/
│   ├── main.py       # Entry point — webcam loop, module integration
│   ├── detector.py   # YOLOv8 inference + colour-based fire heuristic
│   ├── logic.py      # Rule-based safety state machine
│   ├── ui.py         # All OpenCV drawing / overlay functions
│   ├── alerts.py     # Alert channels (console, sound, future: Telegram)
│   └── utils.py      # FPS calculator, resize, screenshot, IoU helpers
│
├── models/           # Place downloaded .pt weights here
├── datasets/         # Future: custom training data
├── screenshots/      # Auto-saved alert screenshots
├── videos/           # Test video clips
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A webcam (built-in or USB)
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Gas-Stove-Safety-System.git
cd AI-Gas-Stove-Safety-System

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note on model weights:** The first run will automatically download `yolov8n.pt` (~6 MB) from Ultralytics. An internet connection is required for this one-time download. After that, copy the downloaded file to the `models/` directory and update `model_path` in `main.py` if needed. Alternatively, Ultralytics caches it in `~/.ultralytics/`.

---

## Usage

```bash
# From the project root directory
cd src
python main.py
```

### Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |

### Configuration

Open `src/main.py` and adjust these variables at the top of `main()`:

```python
WEBCAM_INDEX          = 0    # Change if your webcam isn't index 0
FRAME_WIDTH           = 1280 # Display resolution
FRAME_HEIGHT          = 720
UNATTENDED_SECONDS    = 5    # Seconds before HIGH alert fires
CONFIDENCE_THRESHOLD  = 0.4  # YOLO detection confidence minimum
```

### Testing without a gas stove

- **Person detection:** Simply sit in front of your webcam.
- **Fire simulation:** Hold a bright orange/red object (e.g. a candle or an orange card) in front of the camera.
- **Alert test:** Step away from the camera while the "fire" object is still visible — the alert should fire after `UNATTENDED_SECONDS`.

---

## How It Works

### 1. Person Detection (YOLOv8)

YOLOv8n is a pre-trained convolutional neural network trained on the COCO dataset (80 classes). We use it out-of-the-box to detect the `"person"` class. Detection results include bounding box coordinates and a confidence score.

### 2. Fire Detection (HSV Colour Heuristic)

The COCO dataset does not include a "fire" class, so we use a classical computer vision approach as a stand-in:

1. Convert the frame from BGR to **HSV** colour space (Hue-Saturation-Value).
2. Apply a colour mask for the orange–red–yellow hue range typical of flames (Hue: 0–25° and 160–180°).
3. Perform morphological operations (opening + closing) to remove noise.
4. Find contours and filter by minimum area (1 500 px²).
5. Report any surviving contour as a "fire" detection.

This deliberately simple method is a placeholder. It produces false positives on brightly lit orange objects — which is expected and acceptable for a prototype. The codebase is structured so that swapping in a trained fire-detection model requires changing **one file** (`detector.py`).

### 3. Safety Logic

```
State: last_person_seen_time, flame_start_time

Each frame:
  IF person_detected:
    last_person_seen_time = now

  IF flame_detected AND now - last_person_seen_time >= THRESHOLD:
    risk_level = "HIGH"
    warning = "UNATTENDED STOVE DETECTED! (Xs without person)"
  ELSE:
    risk_level = "NORMAL"
```

The threshold defaults to 5 seconds and is fully configurable.

### 4. Alerts

Alerts are throttled (max 1 per 3 seconds) and dispatched to non-blocking daemon threads so they never slow down the video loop. Active channels: **console print** and optional **sound**. Telegram, email, GPIO, and push notifications are stubbed and ready to enable.

---

## Example Output

**Terminal output when a HIGH-risk event is detected:**

```
============================================================
  🔥  ALERT [14:32:07]
  UNATTENDED STOVE DETECTED! (8s without person)
============================================================
```

**On-screen overlay:**

```
┌────────────────────────┐
│ FLAME : DETECTED       │
│ PERSON: NOT DETECTED   │
│ RISK  : HIGH           │
│ FPS   : 28.4           │
└────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  WARNING
   UNATTENDED STOVE DETECTED! (8s without person)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Screenshots

> _Add your own screenshots after running the project._

## Screenshots

### Normal Monitoring State

![Normal State](screenshots/screenshot1.png)

---

### High Risk Alert

![High Risk Alert](screenshots/screenshot2.png)

## Future Improvements

The codebase is deliberately structured for extensibility. Here are the planned next steps:

### 🔴 High Priority
- [ ] **Custom YOLOv8 fire/smoke model** — Train on a fire/smoke dataset (e.g. [D-Fire](https://github.com/gaiasd/DFireDataset)) using `ultralytics train`. Replace `_detect_fire_heuristic()` in `detector.py` with a real YOLO inference call.
- [ ] **Smoke detection** — Add a second colour-band heuristic or a dedicated class in the custom model.

### 🟡 Medium Priority
- [ ] **Telegram alerts** — `alerts.py` already has the stub. Needs `python-telegram-bot` and a bot token.
- [ ] **Proximity logic** — Use IoU (`utils.iou()`) to check whether the detected person is actually near the stove region, not just anywhere in the frame.
- [ ] **Video file input** — Accept a `.mp4` path as an argument instead of the webcam.

### 🟢 Nice to Have
- [ ] **Raspberry Pi deployment** — The pipeline runs on CPU and is lightweight enough for a Pi 4. Add GPIO buzzer via `alerts._gpio_buzzer_alert()`.
- [ ] **LLM-generated warnings** — Use Claude or GPT-4 Vision to generate a natural-language description of the scene when an alert fires.
- [ ] **Mobile push notifications** — Implement `alerts._mobile_push_alert()` with Pushover or Firebase Cloud Messaging.
- [ ] **Web dashboard** — A lightweight Flask page showing the live feed and alert history.
- [ ] **Multi-camera support** — Monitor multiple stove burners with multiple capture devices.

---

## Why This Project Matters

Kitchen fires cause thousands of injuries and billions of dollars in property damage every year. Most of them are preventable.

This project demonstrates that a **consumer-grade webcam + a small neural network running on a laptop CPU** can provide meaningful safety monitoring. It's not a research paper — it's a working prototype that shows the full pipeline from sensor to alert.

From an engineering perspective, it covers:

- **Computer vision fundamentals** (colour spaces, contour analysis, morphological operations)
- **Deep learning inference** (loading and running a pre-trained YOLOv8 model)
- **Real-time processing** (maintaining >25 FPS with detection + drawing on a CPU)
- **Software design** (modular architecture, separation of concerns, extensibility)
- **Product thinking** (cooldown throttling, configurable thresholds, non-blocking alerts)

---

## Author

Built as a portfolio project for an AI/Computer Vision internship application.

Feel free to fork, extend, and deploy!

---

## License

MIT — use it, learn from it, build on it.
