# Architecture Documentation — AI Gas Stove Safety System

This document is written for an engineer joining the project. It covers the system architecture, execution flow, every module's contract, the reasoning behind the major design decisions, the data flow through the pipeline, a set of interview questions drawn from the codebase, a prioritised improvement list, and a code quality review.

---

## Section 1 — Overall Architecture

```
   Webcam
     │
     ▼
  ┌─────────┐
  │  Frame  │   (raw BGR frame, resized to FRAME_WIDTH x FRAME_HEIGHT)
  └────┬────┘
       │
       ▼
  ┌───────────┐
  │  Detector │   YOLOv8 (person) + HSV heuristic (fire)
  └────┬──────┘
       │  list[dict]: {label, confidence, bbox}
       ▼
  ┌─────────────┐
  │ SafetyLogic │   tracks last_person_seen_time, flame_start_time
  └────┬────────┘
       │  (risk_level, warning_message)
       ▼
  ┌─────────┐        ┌────────┐         ┌──────┐
  │  Alerts │◄───────│  main  │────────►│  UI  │
  └─────────┘  (if   └────────┘  always └──────┘
                HIGH)                        │
                                              ▼
                                        cv2.imshow (Display)
```

**Camera** — `cv2.VideoCapture` opened once in `main()`, read every loop iteration.

**Frame** — the raw frame is immediately resized to a fixed `FRAME_WIDTH` × `FRAME_HEIGHT` via `utils.resize_frame`, so every downstream stage operates on a consistent resolution regardless of the webcam's native output size.

**Detector** — `detector.Detector.detect()` runs two independent detection passes over the same frame (YOLOv8 for people, HSV thresholding for flame) and returns both result sets merged into one flat list of detection dictionaries.

**Safety Logic** — `logic.SafetyLogic.evaluate()` reduces the detection list to two booleans (`flame_detected`, `person_detected`, computed in `main.py`) and turns them, plus elapsed time, into a `risk_level` (`"NORMAL"` or `"HIGH"`) and a human-readable `warning_message`.

**Alerts** — `alerts.trigger_alert()` is called only when `risk_level == "HIGH"`. It throttles repeated identical alerts and dispatches console/sound notifications on background threads.

**UI** — `ui.draw_ui()` draws bounding boxes, a status panel, and (when risk is `HIGH`) a warning banner and watermark onto the frame, then returns the annotated frame.

**Display** — `cv2.imshow` renders the annotated frame in a window; `cv2.waitKey(1)` polls for the `Q` keypress to exit the loop.

---

## Section 2 — Execution Flow

Walking through exactly what happens when a user runs `python src/main.py`:

1. **Import phase.** Python imports `cv2`, `time`, and the four project modules (`detector`, `logic`, `ui`, `alerts`, `utils`). Importing `detector` also imports `ultralytics.YOLO`, which is the heaviest import in the project.
2. **`main()` is called** (guarded by `if __name__ == "__main__":`).
3. **Local configuration constants are set**: `WEBCAM_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT`, `UNATTENDED_SECONDS`, `CONFIDENCE_THRESHOLD`.
4. **`Detector(...)` is constructed.** Its `__init__` calls `YOLO(model_path)`, which loads (or auto-downloads) the weights file, and stores `self.class_names` from `self.model.names`. Two lines are printed confirming the model path and confidence threshold.
5. **`SafetyLogic(unattended_threshold_seconds=UNATTENDED_SECONDS)` is constructed.** Its internal timestamps (`_last_person_seen_time`, `_flame_start_time`) start as `None`.
6. **The webcam is opened** via `cv2.VideoCapture(WEBCAM_INDEX)`. If `cap.isOpened()` is `False`, an error is printed and `main()` returns immediately — the rest of the loop never runs.
7. **Resolution is "suggested"** to the capture device with `cap.set(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT, ...)`. The webcam driver may or may not honour this exactly, which is why frames are still explicitly resized later.
8. **`prev_time = time.time()`** is recorded once, before the loop starts, as the seed for FPS calculation.
9. **The main `while True:` loop begins:**
   1. `cap.read()` grabs a frame. If `ret` is `False`, a warning is printed and the loop `continue`s immediately (no processing that iteration).
   2. `utils.resize_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)` forces the frame to the configured resolution.
   3. `detector.detect(frame)` runs both the YOLO pass and the HSV fire heuristic, returning a merged `detections` list.
   4. Two booleans are derived by scanning `detections`: `flame_detected` (any dict with `label == "fire"`) and `person_detected` (any dict with `label == "person"`).
   5. `safety_logic.evaluate(flame_detected, person_detected, current_time=time.time())` returns `(risk_level, warning_message)`.
   6. If `risk_level == "HIGH"`, `alerts.trigger_alert(warning_message)` is called. Internally this may be a no-op (if still within the cooldown window for an identical message) or may spawn two daemon threads (console print, sound).
   7. `utils.calculate_fps(prev_time)` returns the current FPS estimate and a new `prev_time` for the next iteration.
   8. `ui.draw_ui(...)` draws all overlays onto `frame` and returns the same (mutated) frame object.
   9. `cv2.imshow(...)` displays the frame.
   10. `cv2.waitKey(1)` is checked against `ord("q")`; if pressed, a message is printed and the loop `break`s.
10. **Cleanup**, after the loop exits (via `break` or an exception): `cap.release()` and `cv2.destroyAllWindows()` are called, and a final "Session ended" message is printed.

There is no explicit `try`/`except` around the loop — an unhandled exception inside it (e.g. a webcam disconnect mid-read) will propagate and terminate the process without running the cleanup calls.

---

## Section 3 — Module Breakdown

### `main.py`

- **Purpose:** Single entry point; orchestrates the per-frame pipeline.
- **Responsibilities:** Own the webcam handle, own the configuration constants, call each module's public function in the correct order, and manage program lifecycle (start/loop/cleanup).
- **Public API:** `main()` (also runnable as a script via `__main__` guard).
- **Private helpers:** None — all logic lives inside `main()`.
- **Dependencies:** `cv2`, `time`, `detector.Detector`, `logic.SafetyLogic`, `ui.draw_ui`, `alerts.trigger_alert`, `utils.calculate_fps`, `utils.resize_frame`.
- **Why this module exists:** To keep orchestration separate from implementation. It is the only file that needs to know all other modules exist.
- **How it communicates with other modules:** Exclusively through plain data — frames (`np.ndarray`), lists of dicts, booleans, floats, and strings. It never reaches into another module's private state.
- **Why it is isolated:** So that any individual stage (detection algorithm, alerting channel, drawing style) can change without `main.py` needing to change, as long as the four public function signatures are preserved.

### `detector.py`

- **Purpose:** Convert a raw frame into a list of normalised detections.
- **Responsibilities:** Load and hold the YOLO model; run YOLO inference and filter to classes of interest; run the HSV-based flame heuristic; merge both result sets into one common shape.
- **Public API:** `Detector.__init__(model_path, confidence)`, `Detector.detect(frame) -> list[dict]`.
- **Private Helpers:** `_run_yolo(frame)`, `_detect_fire_heuristic(frame)`.
- **Dependencies:** `cv2`, `numpy`, `ultralytics.YOLO`.
- **Why this module exists:** To contain every detection concern — both the neural-network based one and the classical-CV based one — behind a single `detect()` call, so nothing downstream needs to know two different detection strategies are being combined.
- **How it communicates with other modules:** Returns a `list[dict]`; every dict has exactly the keys `label`, `confidence`, `bbox`. This is the contract every other module (`main.py`, `ui.py`) relies on.
- **Why it is isolated:** Because it is the piece of the system most likely to change (e.g. replacing the HSV heuristic with a trained fire model). Isolating it means that change is contained to this one file.

### `logic.py`

- **Purpose:** Decide, given the current frame's detections, whether the current situation counts as a safety risk.
- **Responsibilities:** Track two pieces of state across frames (`_last_person_seen_time`, `_flame_start_time`); compute how long a flame has been present without a person; produce a risk level and a human-readable warning string.
- **Public API:** `SafetyLogic.__init__(unattended_threshold_seconds)`, `SafetyLogic.evaluate(flame_detected, person_detected, current_time) -> (risk_level, warning_message)`, `SafetyLogic.seconds_unattended(current_time) -> float`, `SafetyLogic.reset()`.
- **Private Helpers:** `_seconds_since_person_seen(current_time)`, `_build_warning(seconds_unattended)`.
- **Dependencies:** None beyond the standard library (it does not import `cv2`, `numpy`, or `time` — `current_time` is always passed in by the caller).
- **Why this module exists:** To isolate the one piece of business logic in the system — "how long is too long" — from both detection and rendering, so it can be reasoned about (and unit tested) using only floats and booleans.
- **How it communicates with other modules:** Takes primitives in (`bool`, `bool`, `float`) and returns primitives out (`str`, `str`). It never touches a frame or a detection dict directly.
- **Why it is isolated:** Because it encodes the one rule that defines the product ("alert if a flame has been unattended for N seconds"). Keeping it separate from `detector.py` and `ui.py` means the safety rule can change (e.g. to a different escalation policy) without touching detection or rendering code, and can be tested purely as arithmetic on timestamps.

### `ui.py`

- **Purpose:** Render all visual overlays onto the current frame.
- **Responsibilities:** Draw bounding boxes and label chips per detection; draw the top-left status panel (flame/person/risk/FPS); draw the bottom warning banner when risk is `HIGH`; draw a bottom-right watermark.
- **Public API:** `draw_ui(frame, detections, flame_detected, person_detected, risk_level, warning_message, fps) -> np.ndarray`.
- **Private Helpers:** `_draw_bounding_boxes`, `_draw_status_panel`, `_draw_warning_banner`, `_draw_watermark`, `_draw_translucent_rect`.
- **Dependencies:** `cv2`, `numpy`, `time` (imported locally inside `_draw_warning_banner` for the blink effect).
- **Why this module exists:** To keep every OpenCV drawing call in one place, so nothing else in the codebase needs to know pixel coordinates, fonts, or colour constants.
- **How it communicates with other modules:** Receives already-computed values (detections, booleans, strings, a float) from `main.py` and returns the same frame object with pixels drawn onto it — it never computes risk or detection results itself.
- **Why it is isolated:** So the visual presentation can be redesigned (different colours, layout, or a completely different rendering backend) without any risk of accidentally changing detection or safety-logic behaviour, since `draw_ui` has no side effects beyond mutating pixel data.

### `alerts.py`

- **Purpose:** Notify a human when the situation is `HIGH` risk, without blocking the video loop.
- **Responsibilities:** Throttle duplicate alerts within a cooldown window; dispatch each notification channel on its own daemon thread; hold the currently-active channels (console, optional sound) and one stubbed, not-yet-implemented channel (Telegram).
- **Public API:** `trigger_alert(message) -> None`.
- **Private Helpers:** `_console_alert(message)`, `_sound_alert()`, `_telegram_alert(message)` (stub, unimplemented — `pass`), `_run_in_thread(func, *args)`.
- **Dependencies:** `time`, `threading`; optionally `playsound` (imported lazily inside `_sound_alert`, with a fallback if it isn't installed).
- **Why this module exists:** To centralise "what happens when risk is HIGH" in one function, so the cooldown/throttle logic exists exactly once rather than being duplicated per channel or per call site.
- **How it communicates with other modules:** Receives only a `message: str` from `main.py` via `trigger_alert`; it has no return value and no channel back to the rest of the system.
- **Why it is isolated:** Because notification side effects (printing, playing audio, eventually hitting a network API) are exactly the kind of work that must never block a real-time video loop. Isolating them behind one function that always threads its work keeps that guarantee in one place.

### `utils.py`

- **Purpose:** Small, stateless helper functions used by other modules (or available for future use).
- **Responsibilities:** FPS calculation, frame resizing, screenshot saving, value clamping.
- **Public API:** `calculate_fps(prev_time) -> (fps, current_time)`, `resize_frame(frame, width, height) -> np.ndarray`, `save_screenshot(frame, output_dir="screenshots") -> str`, `clamp(value, lo, hi) -> float`.
- **Private Helpers:** None.
- **Dependencies:** `time`, `cv2`, `numpy`; `os` (imported locally inside `save_screenshot`).
- **Why this module exists:** To hold small, reusable, pure-ish functions that don't belong conceptually to detection, logic, UI, or alerting.
- **How it communicates with other modules:** `calculate_fps` and `resize_frame` are called every frame from `main.py`. `save_screenshot` and `clamp` are defined but currently **not called anywhere** in the codebase — see the Code Quality Review.
- **Why it is isolated:** So generic helpers don't get duplicated or embedded inside modules that have a more specific single responsibility (e.g. resizing logic doesn't belong inside `detector.py` just because the detector happens to need correctly-sized frames).

---

## Section 4 — Design Decisions

**Why `SafetyLogic` is a class, not a function.** The risk decision depends on state that must persist across frames (`_last_person_seen_time`, `_flame_start_time`). A plain function would require `main.py` to own and pass that state explicitly on every call; a class lets the state live where the logic that updates it lives, and exposes `reset()` for restarting a session cleanly.

**Why UI is isolated.** Drawing code (colours, fonts, pixel offsets) is high-churn and purely cosmetic. Keeping it behind one function (`draw_ui`) means visual changes can never accidentally affect detection or risk logic, since `ui.py` has no way to influence either.

**Why alerts are isolated.** Alert dispatch is the one place in the pipeline doing I/O (printing, audio playback, and eventually network calls), which is also the one category of work that must not block the frame loop. Centralising it behind `trigger_alert` keeps the threading and cooldown behaviour in a single, auditable place.

**Why helper functions exist (`_run_yolo`, `_detect_fire_heuristic`, the `_draw_*` functions, `_seconds_since_person_seen`, `_build_warning`).** Each public method (`detect()`, `draw_ui()`, `evaluate()`) does one coordinating job and delegates each distinct sub-task to a private helper. This keeps each function short and named for exactly what it does, and lets each concern (e.g. "how do I detect fire by colour" vs "how do I detect a person with YOLO") be read, tested, or replaced independently.

**Why `draw_ui` is the only public UI function.** Everything else in `ui.py` is an implementation detail of "how to draw." Exposing a single entry point means the calling code (`main.py`) never needs to know the drawing order or how many distinct visual layers exist.

**Why `main.py` only orchestrates.** Keeping business logic (detection, risk decisions, rendering, alerting) out of `main.py` means the entry point stays readable as a sequence of steps, and each concern can be unit-tested in isolation without needing a webcam or an OpenCV window.

**Why utility functions exist (`utils.py`).** FPS calculation and frame resizing are generic operations with no connection to safety-monitoring specifically; separating them keeps `main.py` and `detector.py` focused on their own domains.

**Why configuration is centralized.** All tunable values (webcam index, resolution, thresholds) are declared together at the top of `main()`, so anyone adjusting behaviour has one place to look rather than hunting through multiple files for magic numbers. (Note: this centralisation is only partial — some tunables, like the HSV bounds and alert cooldown, remain as local/module constants inside their own files rather than being surfaced in `main.py`.)

**Why the FPS helper returns two values.** `calculate_fps` returns both the computed `fps` and the new `current_time`. The caller needs the new timestamp to seed the *next* call, and returning it avoids `calculate_fps` needing to mutate a variable in the caller's scope — Python has no reference-parameter mechanism for primitives, so the "updated state" has to travel back out as a return value.

**Why helper drawing functions return `None`.** Every `_draw_*` helper in `ui.py` (except `draw_ui` itself) mutates the `frame` array in place via OpenCV's drawing calls (`cv2.rectangle`, `cv2.putText`, `cv2.addWeighted`) and has no other output to communicate — there is nothing meaningful to return.

**Why `draw_ui` returns `frame`.** Even though OpenCV drawing mutates arrays in place, `draw_ui` still returns `frame` so the call site (`main.py`) can use `frame = draw_ui(...)` as an explicit, readable assignment, making the data flow visible in `main()` rather than relying on the reader knowing that the frame was mutated invisibly.

**Why `dict.get()` is used** (e.g. `self.class_names.get(class_id, "unknown")`, `LABEL_COLOURS.get(label, DEFAULT_LABEL_COLOUR)`). Both guard against a lookup key that isn't present — an unexpected COCO class id, or a detection label with no configured colour — by supplying a safe default instead of raising a `KeyError` and crashing the video loop mid-frame.

**Why rendering order matters** (in `draw_ui`). Bounding boxes are drawn first, then the status panel, then (conditionally) the warning banner, then the watermark last. Later draws are composited on top of earlier ones, so this ordering ensures the status panel and warning banner are never obscured by a bounding box, and the watermark is always visible on top of everything.

**Why defensive programming is used.** Beyond `dict.get()`, examples include: `_draw_translucent_rect` clamping its rectangle coordinates to the frame's actual bounds before drawing (protecting against a partially off-screen bounding box), `calculate_fps` guarding the division by `elapsed` against a zero value, and `_sound_alert` catching `ImportError` and any general `Exception` around the optional `playsound` call so a missing dependency or a broken audio backend never crashes the alert thread.

**Why mutable NumPy arrays affect API design.** Because `cv2` drawing functions mutate the frame array in place, functions like `_draw_bounding_boxes` don't need to return anything — the effect is already visible to the caller through the shared array reference. This is also why `draw_ui`'s explicit `return frame` is a readability choice rather than a technical necessity (see above).

---

## Section 5 — Data Flow

```
Frame (np.ndarray, BGR, resized to FRAME_WIDTH x FRAME_HEIGHT)
   │
   ▼
Detections (list[dict]: {label, confidence, bbox})
   │
   ▼
Booleans (flame_detected, person_detected) — computed in main.py by scanning detections
   │
   ▼
Risk (risk_level: "NORMAL" | "HIGH") — computed in SafetyLogic.evaluate, using elapsed time
   │
   ▼
Warning (warning_message: str, empty unless risk_level == "HIGH")
   │
   ├──► Alert/UI (alerts.trigger_alert, only if risk_level == "HIGH")
   │
   └──► UI (ui.draw_ui — always called, draws boxes/panel and, if HIGH, the banner)
           │
           ▼
       Annotated frame ──► cv2.imshow (Display)
```

At every stage the data shape narrows: a full image becomes a handful of dictionaries, which become two booleans, which become one risk string and one message string. Nothing downstream of `logic.py` ever needs the original detection dictionaries again except `ui.py`, which receives them a second time (independently of the booleans) purely to draw the per-detection bounding boxes.

---

## Section 6 — Interview Questions

**Architecture**

1. **Q: Why is the pipeline split into `detector.py`, `logic.py`, `ui.py`, and `alerts.py` instead of one script?**
   A: Each module owns exactly one concern — producing detections, deciding risk, rendering, and notifying — and communicates with the others only through plain data (dicts, booleans, strings). This means any one concern can change (e.g. swapping the fire heuristic for a trained model) without requiring changes anywhere else.
   *Why interviewers ask:* To see whether the candidate can articulate separation of concerns concretely, using their own code as the example, rather than reciting the term abstractly.

2. **Q: What is the contract between `Detector.detect()` and the rest of the system?**
   A: It always returns a `list[dict]`, where each dict has exactly the keys `label`, `confidence`, and `bbox` (a 4-tuple of ints). Every downstream consumer (`main.py`, `ui.py`) relies only on this shape, never on how a detection was produced.
   *Why interviewers ask:* To probe understanding of interface design and how a consistent data contract enables swapping implementations.

3. **Q: Why does `main.py` never import `cv2` drawing functions directly?**
   A: All drawing is delegated to `ui.draw_ui`, so `main.py`'s only responsibility is orchestration — reading frames, calling each module, and checking for quit. This keeps `main.py` short and free of presentation logic.
   *Why interviewers ask:* Tests whether the candidate distinguishes orchestration code from implementation code.

4. **Q: What would you need to change to replace the webcam with a video file?**
   A: Only the line `cv2.VideoCapture(WEBCAM_INDEX)` in `main.py` — `cv2.VideoCapture` accepts a file path as well as a device index, and nothing else in the pipeline depends on the frame source.
   *Why interviewers ask:* Checks that the candidate correctly identifies the single point of coupling to a live camera.

5. **Q: If you wanted to add a `smoke` detection type, what would change?**
   A: A new colour heuristic or model call would go inside `detector.py`, appending dicts with `label: "smoke"` to the same list. `logic.py` and `ui.py` would need small updates (e.g. a new colour in `LABEL_COLOURS`, and a new boolean/rule in `SafetyLogic`), but the module boundaries themselves would not need to change.
   *Why interviewers ask:* Tests whether the candidate can reason about the blast radius of a new feature given the current architecture.

**Python**

6. **Q: What does `float | None` mean in `_last_person_seen_time: float | None = None`?**
   A: It's a PEP 604 union type hint (Python 3.10+) meaning the attribute is either a `float` or `None`. It documents that the timestamp may not have been set yet.
   *Why interviewers ask:* Checks familiarity with modern type-hint syntax and its version requirements.

7. **Q: Why use `global` in `alerts.trigger_alert` for `_last_alert_time` and `_last_alert_message`?**
   A: Both are module-level variables that must persist and be mutated across calls to `trigger_alert`; without `global`, an assignment inside the function would create new local variables instead of updating the module-level state.
   *Why interviewers ask:* Tests understanding of Python scoping rules and module-level state versus class-based state.

8. **Q: What's the difference between the module-level state in `alerts.py` and the instance state in `logic.SafetyLogic`?**
   A: `alerts.py`'s cooldown state is global to the process — there is only ever one cooldown timer regardless of how many "alert dispatchers" might exist. `SafetyLogic`'s state is per-instance, so multiple `SafetyLogic` objects (e.g. multiple camera feeds) could track independent timers.
   *Why interviewers ask:* Probes whether the candidate understands the tradeoffs of module-level singletons versus instantiable classes.

9. **Q: Why does `_run_yolo` use `map(int, box.xyxy[0])` instead of just using the values directly?**
   A: `box.xyxy[0]` returns floating point tensor values; `map(int, ...)` truncates each to an integer pixel coordinate, since `cv2.rectangle` and slicing require integer coordinates.
   *Why interviewers ask:* Checks attention to type conversion at library boundaries (PyTorch tensors → OpenCV ints).

10. **Q: Why is `import os` inside `save_screenshot` rather than at the top of `utils.py`?**
    A: It's a local import, scoping the `os` dependency to the one function that needs it. Functionally it behaves identically to a top-level import (Python caches modules), so this is a style/organisation choice rather than a performance one.
    *Why interviewers ask:* Tests understanding of Python's import caching and whether local imports are being used for correctness or simply style.

11. **Q: `divmod(int(seconds_unattended), 60)` is used in `_build_warning` — what does this do?**
    A: It returns `(minutes, seconds)` in one call by dividing the total seconds by 60, giving the quotient and remainder together, which is more concise than two separate `//` and `%` operations.
    *Why interviewers ask:* Checks familiarity with lesser-used built-ins and whether the candidate can read intent from concise code.

**Computer Vision**

12. **Q: Why convert the frame to HSV before thresholding for flame colour, rather than thresholding in BGR/RGB directly?**
    A: HSV separates colour (hue) from brightness (value) and saturation, making it far more robust to lighting changes than BGR, where a colour's raw channel values shift substantially under different lighting. Flame hue stays roughly constant across brightness levels in HSV.
    *Why interviewers ask:* A standard classical CV question testing whether the candidate understands why HSV is preferred for colour segmentation.

13. **Q: Why does the fire heuristic use two separate ranges (`fire_lower1/upper1` and `fire_lower2/upper2`) for red/orange hues?**
    A: Hue is a circular value (0–180 in OpenCV's 8-bit representation) and "red" wraps around both ends of that range (near 0 and near 180). Two ranges are needed to capture both ends without one contiguous range incorrectly spanning the entire hue wheel.
    *Why interviewers ask:* Tests understanding of hue's circular nature, a common gotcha in colour-based CV work.

14. **Q: What does `cv2.findContours` with `cv2.RETR_EXTERNAL` and `cv2.CHAIN_APPROX_SIMPLE` do here?**
    A: `RETR_EXTERNAL` retrieves only the outermost contours (ignoring nested/internal ones), and `CHAIN_APPROX_SIMPLE` compresses the contour points to just the corners of straight segments rather than storing every boundary pixel, reducing memory and computation.
    *Why interviewers ask:* Confirms the candidate understands the specific contour retrieval mode and approximation method chosen, not just that "contours are found."

15. **Q: Why filter contours by `min_fire_area` instead of using every contour found by the mask?**
    A: Small contours are typically noise — isolated pixels or tiny specks that pass the colour threshold by chance. Filtering by area (1000 px²) suppresses these false positives while keeping larger, more plausible flame regions.
    *Why interviewers ask:* Tests whether the candidate understands why raw thresholding output needs post-processing before being treated as a detection.

16. **Q: The fire "confidence" is computed as `min(0.95, 0.50 + area / 50000)` — what is this actually measuring, and is it a real confidence score?**
    A: No — it's a synthetic heuristic that increases with contour area, capped at 0.95, used only so fire detections share the same dictionary shape (`confidence` key) as YOLO's genuine model-output confidence. It says nothing about the model's certainty because there is no model; it is a proxy for "how large/plausible is this blob."
    *Why interviewers ask:* Checks that the candidate won't misrepresent a heuristic as a calibrated probability — an important distinction in ML-adjacent systems.

17. **Q: Why is YOLOv8n (the "nano" variant) used instead of a larger YOLOv8 variant?**
    A: The system needs to run in real time on CPU for a live video feed; the nano variant trades some accuracy for significantly faster inference, which matters more here than squeezing out extra detection accuracy on a single class (`person`).
    *Why interviewers ask:* Tests understanding of the accuracy/latency tradeoff in model selection for real-time systems.

18. **Q: Why filter YOLO's output to `COCO_CLASSES_OF_INTEREST = {"person"}` instead of using all 80 COCO classes?**
    A: The safety logic only cares about whether a person is present; keeping irrelevant classes (car, dog, etc.) would add noise to the detection list and complexity to `main.py`'s boolean extraction for no benefit.
    *Why interviewers ask:* Checks that the candidate filters model output to what the product actually needs, rather than over-exposing raw model capability.

**Object-Oriented Design**

19. **Q: Why is `Detector` a class but `trigger_alert` just a function?**
    A: `Detector` needs to hold state that's expensive to (re)create — the loaded YOLO model and its configured confidence threshold — across many calls to `detect()`. `trigger_alert` needs only module-level state (the cooldown timer), which doesn't warrant a class since there is conceptually only ever one alert dispatcher in the process.
    *Why interviewers ask:* Probes whether the candidate defaults to classes reflexively or chooses based on actual state/lifecycle needs.

20. **Q: What is the single responsibility of `SafetyLogic`, and how would you describe a violation of it?**
    A: Its single responsibility is deciding risk level from flame/person presence over time. It would violate SRP if it started drawing UI elements, playing sounds, or running detection itself — none of which it currently does.
    *Why interviewers ask:* Tests SRP understanding using the codebase as a concrete reference point rather than an abstract definition.

21. **Q: Why does `SafetyLogic` expose both `evaluate()` and `seconds_unattended()` as separate public methods?**
    A: `evaluate()` is the primary decision entry point called every frame from `main.py`. `seconds_unattended()` exposes the same underlying calculation for any caller (e.g. a UI element or test) that wants the raw elapsed time without going through a full risk evaluation.
    *Why interviewers ask:* Tests whether the candidate can justify multiple public methods on a class rather than treating "one public method per class" as a rule.

22. **Q: Why does `SafetyLogic` have a `reset()` method?**
    A: To explicitly clear internal state (both timestamps) — useful for starting a fresh monitoring session or, as noted in its docstring, for isolating test cases from one another.
    *Why interviewers ask:* Checks whether the candidate thinks about testability and object lifecycle management, not just the "happy path" usage.

**OpenCV**

23. **Q: What does `cv2.addWeighted` do in `_draw_translucent_rect`, and why is it needed for the semi-transparent panels?**
    A: `cv2.addWeighted` blends two images (the drawn-on `overlay` copy and the original `frame`) using per-image weights (`alpha` and `1 - alpha`), producing the translucent effect. OpenCV's basic drawing functions like `cv2.rectangle` are always fully opaque, so blending a copy is the standard technique for simulated transparency.
    *Why interviewers ask:* A very common OpenCV interview question, testing whether the candidate understands that "transparency" in OpenCV is achieved via blending, not a native alpha channel on the draw call.

24. **Q: Why does `_draw_translucent_rect` operate on `frame.copy()` rather than drawing directly onto `frame`?**
    A: The blend needs both the "before" state (the original `frame`) and the "after" state (a version with the rectangle drawn) to combine them with `addWeighted`; drawing directly onto `frame` would destroy the "before" state needed for the blend.
    *Why interviewers ask:* Tests attention to why a seemingly redundant `.copy()` is actually required by the algorithm.

25. **Q: What does `cv2.getTextSize` do, and why is it called before placing the label chip's background rectangle?**
    A: It measures the pixel width/height a given text string will occupy in a given font/scale/thickness, without drawing anything. It's called first so the background chip rectangle can be sized to exactly fit the text that will be drawn on top of it.
    *Why interviewers ask:* Checks understanding of the common "measure then draw" pattern needed for dynamically-sized text backgrounds.

26. **Q: Why does the code clamp coordinates (`x1, y1 = max(x1, 0), max(y1, 0)` etc.) before drawing in `_draw_translucent_rect`?**
    A: Detection or panel coordinates could fall outside the frame's actual bounds (e.g. a bounding box partially off-screen); drawing with out-of-bounds coordinates can raise errors or draw incorrectly, so clamping guarantees the rectangle stays within the valid frame region.
    *Why interviewers ask:* Tests whether the candidate anticipates edge cases in coordinate-based drawing rather than assuming inputs are always well-formed.

27. **Q: Why is `cv2.waitKey(1)` used instead of `cv2.waitKey(0)` in the main loop?**
    A: `cv2.waitKey(1)` waits only 1ms for a key press before returning, allowing the loop to keep rendering new frames continuously; `cv2.waitKey(0)` would block indefinitely until a key is pressed, freezing the video feed.
    *Why interviewers ask:* A fundamental OpenCV real-time-loop question — testing whether the candidate understands why the wait duration matters for live video.

**Software Engineering / Maintainability**

28. **Q: Where does configuration live in this project, and is that a good design choice?**
    A: Most tunables (webcam index, resolution, thresholds) are local constants at the top of `main()`. This is readable for a single-file/single-developer prototype but doesn't scale well — it means changing configuration requires editing source code, and other tunables (HSV bounds, cooldown seconds) live scattered in their own modules rather than in one place. A config file or environment variables would be the natural next step.
    *Why interviewers ask:* Tests whether the candidate can honestly critique their own configuration approach rather than only defending it.

29. **Q: The project has no automated tests. Which module would be easiest to unit test, and why?**
    A: `SafetyLogic` — its public methods take only primitives (`bool`, `bool`, `float`) and return primitives (`str`, `str`), with no dependency on a webcam, OpenCV, or a loaded model. It could be fully tested by calling `evaluate()` with synthetic timestamps.
    *Why interviewers ask:* Tests whether the candidate recognises that isolating I/O-free logic (as this project already does) is exactly what makes it testable.

30. **Q: What is the risk of `main.py`'s `while True: ... continue` on a failed frame read (`if not ret`)?**
    A: If the webcam disconnects permanently, this becomes a tight, unbounded retry loop that continuously calls `cap.read()` and prints a warning every iteration, consuming CPU with no backoff or maximum retry count.
    *Why interviewers ask:* Checks whether the candidate can spot an unbounded retry / missing backoff as a real operational risk.

31. **Q: How would you add persistence (e.g. logging every HIGH-risk event to a file) without violating the existing module boundaries?**
    A: Add the capability inside `alerts.py` (since it already owns "what happens on HIGH risk") as a new private helper dispatched via `_run_in_thread`, called from within `trigger_alert` alongside the existing console/sound channels — consistent with how `_telegram_alert` is already stubbed as a future channel.
    *Why interviewers ask:* Tests whether the candidate can extend the system in a way that respects the existing architecture rather than bolting logic onto `main.py`.

**Design Decisions**

32. **Q: Why does `Detector.detect()` merge YOLO and HSV results into one list instead of returning two separate lists?**
    A: Downstream code (`main.py`'s boolean checks, `ui.py`'s bounding-box loop) treats every detection identically regardless of source — a single merged list lets both consumers use one simple loop/any() check instead of needing to know about two detection sources.
    *Why interviewers ask:* Tests whether the candidate values a unified interface over exposing internal implementation detail.

33. **Q: Why does `SafetyLogic._seconds_since_person_seen` fall back to `_flame_start_time` when no person has ever been seen?**
    A: Without this fallback, `_last_person_seen_time` would remain `None` indefinitely if a person is never detected during the session, making "seconds since person seen" undefined. Falling back to the flame's own start time means the timer still starts counting from when the flame first appeared, rather than firing an alert instantly (which would happen if the fallback returned a very large number) or never firing at all.
    *Why interviewers ask:* This is a subtle piece of logic — testing whether the candidate can trace through an edge case (person never seen this session) and explain why the chosen fallback behaviour is reasonable.

**API Design**

34. **Q: Why does `SafetyLogic.evaluate()` return a tuple `(risk_level, warning_message)` instead of a small object or dataclass?**
    A: For a two-value return with no shared behaviour, a plain tuple keeps the API lightweight; a dataclass would add ceremony without adding capability, given both values are consumed together in every current call site. (A dataclass would arguably improve readability at call sites, e.g. `result.risk_level`, as the return grows.)
    *Why interviewers ask:* Tests whether the candidate can discuss the tradeoff between tuple simplicity and structured-object clarity rather than assuming one is always correct.

35. **Q: Why is `current_time` passed into `SafetyLogic.evaluate()` rather than the method calling `time.time()` itself?**
    A: Passing time in as a parameter makes `SafetyLogic` free of any dependency on the `time` module or wall-clock behaviour, which is what makes it possible to unit test with arbitrary synthetic timestamps instead of needing to control or mock global time.
    *Why interviewers ask:* A classic dependency-injection question — checks whether the candidate recognises this as a deliberate testability decision.

---

## Section 7 — Possible Improvements

**Quick Wins**
- Fix the `assets/` vs `assests/` path mismatch in `alerts._sound_alert` so the sound channel actually finds `alert.mp3`.
- Fix `Detector`'s default `model_path` (or document explicitly) so it reliably resolves to the checked-in `models/yolov8n.pt` regardless of the working directory the script is launched from (e.g. build the path relative to `__file__`).
- Wire `utils.save_screenshot` into `main.py` so a screenshot is actually captured when `risk_level == "HIGH"`, as already suggested in the function's own docstring.
- Add a small `try`/`except` around the main loop body so a webcam disconnect triggers `cap.release()`/`cv2.destroyAllWindows()` cleanup instead of an unhandled crash.

**Medium Improvements**
- Move all configuration constants (currently split between `main()` locals and module-level constants in `alerts.py`/`detector.py`) into a single config file or environment-variable-driven settings module.
- Add unit tests for `SafetyLogic`, which requires no external dependencies (webcam, model, OpenCV window) to test.
- Add a maximum retry count or backoff delay to the `if not ret: continue` frame-read failure path.
- Implement basic proximity awareness — using the fire and person bounding boxes' spatial relationship (e.g. an IoU or distance check) rather than treating any person anywhere in frame as "attending" the stove.

**Production Improvements**
- Replace the HSV colour heuristic with a trained fire/smoke detection model (the codebase's data contract — `{label, confidence, bbox}` — already supports this swap without touching `logic.py` or `ui.py`).
- Implement the stubbed `_telegram_alert` (or another push-notification channel) so alerts can reach a user who isn't watching the console.
- Add structured logging (timestamps, risk transitions, detection counts) instead of only `print()` statements, to support post-incident review.
- Add a health/monitoring endpoint or heartbeat if this is ever deployed as a long-running service, so an operator can tell the process is still alive without watching the OpenCV window.

---

## Section 8 — Code Quality Review

**Strengths**
- Clear separation of concerns across `detector.py`, `logic.py`, `ui.py`, and `alerts.py`, each with one public entry point.
- Consistent detection data contract (`{label, confidence, bbox}`) used uniformly regardless of detection source.
- `SafetyLogic` is fully decoupled from I/O (no `cv2`, no direct `time.time()` calls), which makes it the most maintainable and testable module in the project.
- Defensive coding in several places: `dict.get()` with sensible defaults, coordinate clamping before drawing, division-by-zero guard in FPS calculation, and broad exception handling around the optional sound dependency.
- Alert dispatch is deliberately non-blocking (background daemon threads), appropriate for a real-time video loop.

**Weaknesses**
- **Path fragility:** `Detector`'s default `model_path="models/yolov8n.pt"` is relative to the current working directory, not to the file's own location, and the documented run command (`cd src && python main.py`) means this default silently resolves to the wrong location.
- **Asset path mismatch:** `alerts._sound_alert`'s `SOUND_FILE = "assets/audio/alert.mp3"` does not match the repository's actual `assests/audio/` folder name, so the sound channel silently degrades to the bell fallback under normal usage.
- **No automated tests** exist anywhere in the repository, despite at least `SafetyLogic` being straightforward to test in isolation.
- **No exception handling around the main loop**, so any unexpected error (e.g. the webcam disconnecting) skips the `cap.release()`/`cv2.destroyAllWindows()` cleanup entirely.
- **Unbounded retry** on a failed frame read (`if not ret: continue`) has no backoff or retry limit.
- **Synthetic confidence values** for fire detections are stored under the same `confidence` key as genuine YOLO confidence, which could mislead a future developer into treating them as comparable, calibrated numbers.

**Unused / Dead Code**
- `utils.save_screenshot` — fully implemented, documented with a usage example in its own docstring, but never called from `main.py` or anywhere else.
- `utils.clamp` — implemented but never called anywhere in the codebase.
- `alerts._telegram_alert` — an intentional, documented stub (`pass`) representing a not-yet-implemented feature rather than leftover dead code; it is clearly commented as such.

**Coupling**
- Coupling between modules is low and limited to well-defined data shapes (dicts, booleans, tuples, strings) rather than shared mutable state or direct cross-imports of internals.
- The one implicit coupling in the system is the detection dictionary shape (`label`/`confidence`/`bbox`) — an informal contract enforced only by convention/docstrings, not by a shared type definition (e.g. a `TypedDict` or dataclass).

**Maintainability**
- High for `logic.py` and `alerts.py`, both small, single-purpose, and easy to reason about in isolation.
- Moderate for `detector.py` and `ui.py` — each contains two somewhat distinct concerns (YOLO detection + HSV heuristic; box/panel drawing + banner/watermark drawing) that are already well-separated into private helpers within the file, but could eventually warrant splitting into separate files if either grows further (e.g. a dedicated `fire_detector.py`).

**Scalability**
- The architecture scales conceptually to more detection types or alert channels without structural changes, since both are already list/threaded-based rather than hardcoded to exactly two entries.
- The current implementation only handles a single camera/session; there is no mechanism for running multiple camera feeds concurrently, though `SafetyLogic`'s instance-based state means multiple instances could in principle track multiple feeds independently.

**Readability**
- Generally high — functions are short, named for what they do, and consistently documented with docstrings describing arguments and return values.
- Minor inconsistencies exist in comment style (e.g. "Dependancies" typo in `alerts.py`, inconsistent spacing/blank-line conventions between files) that don't affect functionality but are worth cleaning up for polish.