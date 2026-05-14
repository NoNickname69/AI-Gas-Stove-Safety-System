"""
logic.py — Safety Logic Module
AI Gas Stove Safety System

Implements the core rule-based safety state machine:

    IF   flame is detected
    AND  no person has been seen for >= UNATTENDED_THRESHOLD seconds
    THEN risk_level = "HIGH"   → trigger alert
    ELSE risk_level = "NORMAL"

The module is intentionally simple so the decision flow is easy to trace,
test, and extend (e.g. adding smoke detection, multi-burner tracking, etc.)
"""

import time


class SafetyLogic:
    """
    Tracks detector outputs over time and decides the current risk level.

    State kept between frames:
        _last_person_seen_time : float | None
            Timestamp of the most recent frame in which a person was present.
            None means we've never seen a person this session.

        _flame_start_time : float | None
            Timestamp when the current uninterrupted flame detection began.
            Resets when flame disappears.
    """

    # Risk level constants — keeps strings consistent across the codebase
    RISK_NORMAL = "NORMAL"
    RISK_MEDIUM  = "MEDIUM"   # Reserved for future use (e.g. person near stove but not watching)
    RISK_HIGH    = "HIGH"

    def __init__(self, unattended_threshold_seconds: float = 5.0):
        """
        Args:
            unattended_threshold_seconds:
                How long (in seconds) flame must be detected without a nearby
                person before escalating to HIGH risk.
        """
        self.threshold = unattended_threshold_seconds

        # Internal state
        self._last_person_seen_time: float | None = None
        self._flame_start_time:      float | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        flame_detected:  bool,
        person_detected: bool,
        current_time:    float
    ) -> tuple[str, str]:
        """
        Evaluate the current safety state.

        Args:
            flame_detected:  True if at least one fire/flame was detected.
            person_detected: True if at least one person was detected.
            current_time:    time.time() value from the calling frame.

        Returns:
            (risk_level, warning_message)
                risk_level     : "NORMAL" | "MEDIUM" | "HIGH"
                warning_message: Human-readable alert string (empty if NORMAL).
        """
        # ── Update person tracker ──────────────────────────────────────
        if person_detected:
            self._last_person_seen_time = current_time

        # ── Update flame tracker ───────────────────────────────────────
        if flame_detected:
            if self._flame_start_time is None:
                # Flame just appeared — start the clock
                self._flame_start_time = current_time
        else:
            # Flame gone — reset the clock
            self._flame_start_time = None

        # ── Apply safety rules ─────────────────────────────────────────
        risk_level      = self.RISK_NORMAL
        warning_message = ""

        if flame_detected:
            seconds_unattended = self._seconds_since_person_seen(current_time)

            if seconds_unattended >= self.threshold:
                risk_level      = self.RISK_HIGH
                warning_message = self._build_warning(seconds_unattended)

        return risk_level, warning_message

    def seconds_unattended(self, current_time: float) -> float:
        """
        How many seconds since a person was last detected.
        Returns 0 if a person was seen in the current frame, or
        the full session duration if a person was never seen.
        """
        return self._seconds_since_person_seen(current_time)

    def reset(self):
        """
        Reset all internal state (e.g. between test cases or camera sessions).
        """
        self._last_person_seen_time = None
        self._flame_start_time      = None

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _seconds_since_person_seen(self, current_time: float) -> float:
        """
        Returns how many seconds have elapsed since a person was last visible.

        If we have never seen a person this session, use the flame start time
        as the reference (so we don't trigger on startup before detection warms up).
        """
        if self._last_person_seen_time is not None:
            return current_time - self._last_person_seen_time

        if self._flame_start_time is not None:
            # Person has never been seen — measure from when flame first appeared
            return current_time - self._flame_start_time

        # Neither person nor flame seen yet — no concern
        return 0.0

    def _build_warning(self, seconds_unattended: float) -> str:
        """
        Build a human-readable warning string.
        """
        mins, secs = divmod(int(seconds_unattended), 60)
        if mins > 0:
            time_str = f"{mins}m {secs}s"
        else:
            time_str = f"{secs}s"

        return f"UNATTENDED STOVE DETECTED! ({time_str} without person)"
