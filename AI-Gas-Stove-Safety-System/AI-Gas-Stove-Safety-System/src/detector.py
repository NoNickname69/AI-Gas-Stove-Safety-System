"""
detector.py — Object Detection Module
AI Gas Stove Safety System

Wraps YOLOv8 inference into a clean, reusable Detector class.

Design notes:
  - YOLOv8n (nano) is used for speed on CPU.  Swap to yolov8s.pt or a
    custom-trained model simply by changing `model_path` in main.py.
  - COCO-80 does not contain a "fire" class.  We simulate fire detection
    here using a simple HSV colour heuristic so the full pipeline works
    end-to-end.  A real deployment would swap this for a fine-tuned YOLO
    model trained on fire/flame data (see README — Future Improvements).
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ── COCO class names we care about ────────────────────────────────────────
# YOLOv8n is pre-trained on the 80-class COCO dataset.
# We only act on "person"; fire is handled by our colour heuristic below.
COCO_CLASSES_OF_INTEREST = {"person"}


class Detector:
    """
    Runs YOLOv8 inference on a frame and returns a normalised list of
    detection dictionaries.

    Each detection dict has the shape:
        {
            "label":      str,          # e.g. "person" or "fire"
            "confidence": float,        # 0.0 – 1.0
            "bbox":       (x1,y1,x2,y2) # pixel coordinates, int
        }
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence: float = 0.4):
        """
        Args:
            model_path:  Path to the .pt weights file.
                         If the file doesn't exist, Ultralytics auto-downloads it.
            confidence:  Minimum confidence score to keep a detection.
        """
        self.confidence = confidence

        # Load model — will auto-download yolov8n.pt on first run
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: "person", 1: "bicycle", …}

        print(f"[Detector] Model loaded: {model_path}")
        print(f"[Detector] Confidence threshold: {confidence}")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR frame.

        Returns:
            List of detection dicts (see class docstring).
        """
        detections = []

        # ── 1. YOLOv8 inference (for "person" and other COCO classes) ─
        yolo_detections = self._run_yolo(frame)
        detections.extend(yolo_detections)

        # ── 2. Colour-based fire heuristic ─────────────────────────────
        #    Simulates fire detection until a custom model is trained.
        #    Replace this section with a real fire-detection model later.
        fire_detections = self._detect_fire_heuristic(frame)
        detections.extend(fire_detections)

        return detections

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _run_yolo(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 and filter for classes we care about (COCO_CLASSES_OF_INTEREST).
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence:
                continue

            class_id = int(box.cls[0])
            label = self.class_names.get(class_id, "unknown")

            # Only keep classes relevant to our safety system
            if label not in COCO_CLASSES_OF_INTEREST:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label":      label,
                "confidence": round(confidence, 2),
                "bbox":       (x1, y1, x2, y2)
            })

        return detections

    def _detect_fire_heuristic(self, frame: np.ndarray) -> list[dict]:
        """
        Colour-based flame detector using HSV thresholding.

        How it works:
          1. Convert the frame to HSV colour space.
          2. Create a mask for orange-red hues typical of flame.
          3. If a large enough contiguous region passes the mask,
             treat it as a flame detection.

        ⚠️  This is intentionally simple — it's a placeholder that makes
            the pipeline functional without a trained fire model.
            It WILL false-positive on brightly lit orange/red objects.
            Replace with a custom YOLOv8 fire model for production use.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV ranges for fire colours (orange → yellow → bright red)
        lower_fire1 = np.array([0,   120, 120], dtype=np.uint8)
        upper_fire1 = np.array([25,  255, 255], dtype=np.uint8)

        lower_fire2 = np.array([160, 120, 120], dtype=np.uint8)
        upper_fire2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # Small noise removal
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fire_detections = []
        min_fire_area = 1500  # Pixels² — ignore tiny blobs

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_fire_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Confidence proxy: larger blob → higher simulated confidence
            sim_confidence = min(0.95, 0.50 + area / 50000)

            fire_detections.append({
                "label":      "fire",
                "confidence": round(sim_confidence, 2),
                "bbox":       (x, y, x + w, y + h)
            })

        return fire_detections
