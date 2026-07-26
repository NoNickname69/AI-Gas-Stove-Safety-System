import cv2
import numpy as np


# Colour palette (BGR)
COLOUR_GREEN  = (0,   220,   0)
COLOUR_RED    = (0,    30, 220)
COLOUR_ORANGE = (0,   140, 255)
COLOUR_WHITE  = (255, 255, 255)
COLOUR_BLACK  = (0,     0,   0)
COLOUR_YELLOW = (0,   220, 220)
COLOUR_CYAN   = (220, 220,   0)

# Per-label box colours
LABEL_COLOURS = {
    "person": COLOUR_GREEN,
    "fire":   COLOUR_RED,
}

DEFAULT_LABEL_COLOUR = COLOUR_ORANGE


def draw_ui(
    frame:           np.ndarray,
    detections:      list[dict],
    flame_detected:  bool,
    person_detected: bool,
    risk_level:      str,
    warning_message: str,
    fps:             float
) -> np.ndarray:
    """
    Draw all overlays onto `frame` and return the annotated frame.

    Args:
        frame           : Raw BGR frame from OpenCV.
        detections      : List of detection dicts from detector.py.
        flame_detected  : Whether flame is present this frame.
        person_detected : Whether a person is present this frame.
        risk_level      : "NORMAL" | "HIGH"
        warning_message : Alert text (empty string if NORMAL).
        fps             : Frames per second to display.

    Returns:
        Annotated BGR frame.
    """
    # Draw in layers, back-to-front so text is on top of boxes

    _draw_bounding_boxes(frame, detections)
    _draw_status_panel(frame, flame_detected, person_detected, risk_level, fps)

    if risk_level == "HIGH":
        _draw_warning_banner(frame, warning_message)

    # Watermark in bottom-right corner
    _draw_watermark(frame)

    return frame

# Private drawing helpers

def _draw_bounding_boxes(
        frame: np.ndarray, 
        detections: list[dict],
) -> None:
    """
    Draw a bounding box + label chip for every detection.
    """
    for det in detections:
        label      = det["label"]
        confidence = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        colour = LABEL_COLOURS.get(label, DEFAULT_LABEL_COLOUR) # Orange as default colour

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label chip background
        text       = f"{label.upper()}  {confidence:.0%}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        chip_y1 = max(y1 - th - 10, 0)
        chip_y2 = y1
        cv2.rectangle(frame, (x1, chip_y1), (x1 + tw + 8, chip_y2), colour, cv2.FILLED)

        # Label text
        cv2.putText(
            frame, text,
            (x1 + 4, chip_y2 - 4),
            font, font_scale, COLOUR_BLACK, thickness, cv2.LINE_AA
        )


def _draw_status_panel(
    frame:           np.ndarray,
    flame_detected:  bool,
    person_detected: bool,
    risk_level:      str,
    fps:             float
) -> None:
    """
    Draw the semi-transparent status panel in the top-left corner.

    Panel contains:
        FLAME status
        PERSON status
        RISK level
        FPS counter
    """
    panel_x, panel_y = 15, 15
    line_height       = 30
    panel_w           = 300
    panel_h           = 5 * line_height + 20  # 4 rows + padding

    # Semi-transparent dark background
    _draw_translucent_rect(
        frame,
        x1=panel_x - 5, y1=panel_y - 5,
        x2=panel_x + panel_w, y2=panel_y + panel_h,
        colour=(20, 20, 20), alpha=0.65
    )

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_small = 0.62
    bold       = 2

    rows = [
        ("FLAME",    "DETECTED" if flame_detected  else "NOT DETECTED",
         COLOUR_RED    if flame_detected  else COLOUR_GREEN),
        ("PERSON",   "DETECTED" if person_detected else "NOT DETECTED",
         COLOUR_GREEN  if person_detected else COLOUR_YELLOW),
        ("RISK",     risk_level,
         COLOUR_RED    if risk_level == "HIGH" else COLOUR_GREEN),
        (f"FPS",     f"{fps:.1f}",
         COLOUR_CYAN),
    ]

    for i, (key, value, value_colour) in enumerate(rows):
        y = panel_y + 20 + i * line_height

        # Key (white)
        cv2.putText(
            frame, f"{key}: ",
            (panel_x, y),
            font, font_small, COLOUR_WHITE, bold, cv2.LINE_AA
        )

        # Measure key width so we can right-offset the value
        (kw, _), _ = cv2.getTextSize(f"{key}: ", font, font_small, bold)

        # Value (coloured)
        cv2.putText(
            frame, value,
            (panel_x + kw, y),
            font, font_small, value_colour, bold, cv2.LINE_AA
        )


def _draw_warning_banner(frame: np.ndarray, message: str) -> None:
    """
    Draw an animated-style warning banner at the bottom of the frame.
    Called only when risk_level == "HIGH".
    """
    h, w = frame.shape[:2]
    banner_h = 70

    # Red translucent bar across the full width
    _draw_translucent_rect(
        frame,
        x1=0, y1=h - banner_h,
        x2=w, y2=h,
        colour=(0, 0, 180), alpha=0.75
    )

    # " WARNING" tag
    cv2.putText(
        frame, "WARNING",
        (20, h - banner_h + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOUR_YELLOW, 2, cv2.LINE_AA
    )

    # Main alert message
    cv2.putText(
        frame, message,
        (20, h - banner_h + 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOUR_WHITE, 2, cv2.LINE_AA
    )

    # Blinking border — flickers every ~15 frames using time
    import time
    if int(time.time() * 3) % 2 == 0:
        cv2.rectangle(frame, (0, h - banner_h), (w - 1, h - 1), COLOUR_RED, 3)


def _draw_watermark(frame: np.ndarray) -> None:
    """
    Subtle project watermark in the bottom-right corner.
    """
    h, w = frame.shape[:2]
    text  = "AI Gas Stove Safety System"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    cv2.putText(
        frame, text,
        (w - tw - 10, h - 10),
        font, scale, (160, 160, 160), thick, cv2.LINE_AA
    )


# Utility
def _draw_translucent_rect(
    frame:  np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    colour: tuple,
    alpha:  float = 0.5
) -> None:
    """
    Blend a filled rectangle onto `frame` with transparency.

    Args:
        alpha : 0.0 = fully transparent, 1.0 = fully opaque.
    """
    # Clamp coordinates to frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)

    if x1 >= x2 or y1 >= y2:
        return  # Nothing to draw

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
