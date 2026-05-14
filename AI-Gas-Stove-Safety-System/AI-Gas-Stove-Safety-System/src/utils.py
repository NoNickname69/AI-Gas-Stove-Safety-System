"""
utils.py — Utility / Helper Functions
AI Gas Stove Safety System

Small, reusable functions used across the project.
Keeping these isolated makes main.py cleaner and the helpers easy to test.
"""

import time
import cv2
import numpy as np


def calculate_fps(prev_time: float) -> tuple[float, float]:
    """
    Calculate frames per second based on elapsed time since the last frame.

    Args:
        prev_time: The timestamp (time.time()) of the previous frame.

    Returns:
        (fps, current_time) — fps rounded to 1 decimal, and the new timestamp.

    Usage:
        prev_time = time.time()
        while True:
            ...
            fps, prev_time = calculate_fps(prev_time)
    """
    current_time = time.time()
    elapsed = current_time - prev_time

    # Avoid division by zero on the very first frame
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    return round(fps, 1), current_time


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize a frame to the given (width, height) using bilinear interpolation.

    Maintains the requested dimensions regardless of source aspect ratio.
    For a letterbox-preserving resize, see `resize_letterbox()` below.

    Args:
        frame  : BGR frame from OpenCV.
        width  : Target width in pixels.
        height : Target height in pixels.

    Returns:
        Resized frame.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_letterbox(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    pad_colour: tuple = (0, 0, 0)
) -> np.ndarray:
    """
    Resize a frame to (target_w, target_h) while preserving the aspect ratio.
    Fills any empty space with `pad_colour` (black by default).

    Useful when you don't want the image to be stretched.

    Args:
        frame      : BGR source frame.
        target_w   : Target canvas width.
        target_h   : Target canvas height.
        pad_colour : BGR colour for padding areas.

    Returns:
        Padded and resized frame of size (target_h, target_w, 3).
    """
    src_h, src_w = frame.shape[:2]
    scale    = min(target_w / src_w, target_h / src_h)
    new_w    = int(src_w * scale)
    new_h    = int(src_h * scale)

    resized  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas   = np.full((target_h, target_w, 3), pad_colour, dtype=np.uint8)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    return canvas


def save_screenshot(frame: np.ndarray, output_dir: str = "screenshots") -> str:
    """
    Save the current frame as a JPEG screenshot with a timestamp filename.

    Args:
        frame      : BGR frame to save.
        output_dir : Directory where screenshots are stored.

    Returns:
        The full file path of the saved image.

    Usage:
        # Save a screenshot when risk goes HIGH
        if risk_level == "HIGH":
            path = save_screenshot(frame)
            print(f"Screenshot saved: {path}")
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(output_dir, f"alert_{timestamp}.jpg")

    cv2.imwrite(filename, frame)
    return filename


def clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp `value` to the closed interval [lo, hi].

    Args:
        value : The number to clamp.
        lo    : Lower bound.
        hi    : Upper bound.

    Returns:
        The clamped value.
    """
    return max(lo, min(value, hi))


def iou(boxA: tuple, boxB: tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Useful for future features like:
      • Checking whether a detected person is "near" the stove region.
      • Suppressing duplicate detections.

    Args:
        boxA, boxB : Each is (x1, y1, x2, y2) in pixel coordinates.

    Returns:
        IoU score in [0.0, 1.0].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
