import time
import cv2
import numpy as np


def calculate_fps(
        prev_time: float,
) -> tuple[float, float]:
    """
    Calculate frames per second based on elapsed time since the last frame.

    Args:
        prev_time: The timestamp (time.time()) of the previous frame.

    Returns:
        (fps, current_time) — fps rounded to 1 decimal, and the new timestamp.
    """

    current_time = time.time()
    elapsed = current_time - prev_time

    # Avoid division by zero on the very first frame
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    return round(fps, 1), current_time


def resize_frame(
        frame: np.ndarray,
        width: int,
        height: int,
) -> np.ndarray:
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