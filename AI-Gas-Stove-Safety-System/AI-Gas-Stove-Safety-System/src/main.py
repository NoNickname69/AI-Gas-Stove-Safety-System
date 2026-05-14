"""
main.py — Entry Point
AI Gas Stove Safety System

This is the main loop that ties everything together:
  1. Opens the webcam
  2. Runs YOLO detection on each frame
  3. Applies safety logic
  4. Draws UI overlays
  5. Shows the result in a window

Run with:
    python src/main.py
"""

import cv2
import time

from detector import Detector
from logic import SafetyLogic
from ui import draw_ui
from alerts import trigger_alert
from utils import calculate_fps, resize_frame


def main():
    # ------------------------------------------------------------------ #
    #  Configuration — tweak these to suit your setup                     #
    # ------------------------------------------------------------------ #
    WEBCAM_INDEX = 0          # 0 = default webcam; change if using USB cam
    FRAME_WIDTH  = 1280       # Target display width
    FRAME_HEIGHT = 720        # Target display height
    UNATTENDED_SECONDS = 5    # Seconds before "unattended stove" alert fires
    CONFIDENCE_THRESHOLD = 0.4  # Minimum YOLO confidence to accept a detection

    # ------------------------------------------------------------------ #
    #  Initialise modules                                                 #
    # ------------------------------------------------------------------ #
    print("[INFO] Loading YOLOv8 model …")
    detector = Detector(
        model_path="models/yolov8n.pt",
        confidence=CONFIDENCE_THRESHOLD
    )

    safety_logic = SafetyLogic(unattended_threshold_seconds=UNATTENDED_SECONDS)

    print("[INFO] Opening webcam …")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check WEBCAM_INDEX in main.py.")
        return

    # Suggest (but don't force) a resolution — hardware may override this
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[INFO] Starting safety monitor — press Q to quit.")

    prev_time = time.time()   # Used for FPS calculation

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame — retrying …")
            continue

        # Resize frame to a consistent display size
        frame = resize_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)

        # ── 1. Run object detection ────────────────────────────────────
        detections = detector.detect(frame)
        # detections is a list of dicts:
        #   { "label": str, "confidence": float, "bbox": (x1,y1,x2,y2) }

        # ── 2. Apply safety logic ──────────────────────────────────────
        flame_detected  = any(d["label"] == "fire" for d in detections)
        person_detected = any(d["label"] == "person" for d in detections)

        risk_level, warning_message = safety_logic.evaluate(
            flame_detected=flame_detected,
            person_detected=person_detected,
            current_time=time.time()
        )

        # ── 3. Trigger alert if HIGH risk ──────────────────────────────
        if risk_level == "HIGH":
            trigger_alert(warning_message)

        # ── 4. Calculate FPS ───────────────────────────────────────────
        fps, prev_time = calculate_fps(prev_time)

        # ── 5. Draw everything onto the frame ─────────────────────────
        frame = draw_ui(
            frame=frame,
            detections=detections,
            flame_detected=flame_detected,
            person_detected=person_detected,
            risk_level=risk_level,
            warning_message=warning_message,
            fps=fps
        )

        # ── 6. Display ─────────────────────────────────────────────────
        cv2.imshow("AI Gas Stove Safety System", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quitting …")
            break

    # ------------------------------------------------------------------ #
    #  Cleanup                                                            #
    # ------------------------------------------------------------------ #
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended. Stay safe!")


if __name__ == "__main__":
    main()
