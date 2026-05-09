"""
Marker visibility state monitor using EWMA-smoothed binary observations
with dual-threshold hysteresis (Schmitt trigger) and confirmation debounce.

See articles/heuristic_based_live_multicamera_map_building/marker_visibility_algorithm.tex
for the formal description.
"""

from __future__ import annotations

from logging import getLogger

from classes.marker import CompositeDetection, MarkerVisibilityState
from classes.types import MarkerData
from utils.constants import (
    DEFAULT_RECALC_HEURISTIC_APPEARING_MARKER_CONSECUTIVE_ROUNDS_DELAY,
    DEFAULT_RECALC_HEURISTIC_EXTINGUISHING_MARKER_CONSECUTIVE_ROUNDS_DELAY,
)

class _MarkerState:
    """Per-marker state for EWMA + hysteresis tracking."""

    def __init__(self) -> None:
        self.visibility_score: float = 0.0
        self.last_confirmed: MarkerVisibilityState = MarkerVisibilityState.INVISIBLE
        self.consecutive_appear_rounds: int = 0
        self.consecutive_disappear_rounds: int = 0


class MarkerAppearanceStateMonitor:
    """
    Tracks appearance state changes of markers using an EWMA-smoothed visibility
    signal with dual-threshold hysteresis and confirmation debounce.

    The algorithm reduces false invalidation triggers from transient detection
    failures (e.g. motion blur, brief occlusion, lighting flicker) while remaining
    responsive to genuine marker appearance/disappearance events.
    """

    def __init__(self, logLevel: int | str):
        self._markerStates: dict[MarkerData, _MarkerState] = {}
        self._markersVisibleLastRound: set[MarkerData] = set()
        self._markerPositionsLastRound: dict[MarkerData, CompositeDetection] = {}

        self.logger = getLogger("MarkerAppearanceStateMonitor")
        self.logger.setLevel(logLevel)

    def step(
        self,
        markersVisibleNow: set[MarkerData],
        markerPositions: dict[MarkerData, CompositeDetection],
        appearingConsecutiveRoundsDelay: int = DEFAULT_RECALC_HEURISTIC_APPEARING_MARKER_CONSECUTIVE_ROUNDS_DELAY,
        extinguishingConsecutiveRoundsDelay: int = DEFAULT_RECALC_HEURISTIC_EXTINGUISHING_MARKER_CONSECUTIVE_ROUNDS_DELAY,
        ewmaAlpha: float = 0.35,
        hysteresisThresholdHigh: float = 0.75,
        hysteresisThresholdLow: float = 0.25,
    ) -> list[tuple[MarkerData, str]]:
        """Updates internal state and detects appearance/disappearance events.

        Uses EWMA smoothing of the binary visibility signal, dual-threshold
        hysteresis to prevent oscillation, and confirmation rounds for debounce.

        Returns a list of (MarkerData, change_description) for each detected change.
        """
        changes: list[tuple[MarkerData, str]] = []
        allTrackedMarkers = set(self._markerStates.keys()) | markersVisibleNow | self._markersVisibleLastRound

        for marker in allTrackedMarkers:
            x_t = 1.0 if marker in markersVisibleNow else 0.0
            state = self._markerStates.get(marker)
            if state is None:
                state = _MarkerState()
                state.visibility_score = x_t
                # Newly tracked: treat as previously invisible so we can confirm "appeared" after k rounds
                state.last_confirmed = MarkerVisibilityState.INVISIBLE
                self._markerStates[marker] = state

            # EWMA update: s_t = α·x_t + (1-α)·s_{t-1}
            state.visibility_score = ewmaAlpha * x_t + (1.0 - ewmaAlpha) * state.visibility_score

            # Schmitt trigger with confirmation debounce
            if state.visibility_score >= hysteresisThresholdHigh:
                if state.last_confirmed == MarkerVisibilityState.INVISIBLE:
                    state.consecutive_appear_rounds += 1
                    state.consecutive_disappear_rounds = 0
                    if state.consecutive_appear_rounds >= appearingConsecutiveRoundsDelay:
                        state.last_confirmed = MarkerVisibilityState.VISIBLE
                        state.consecutive_appear_rounds = 0
                        changes.append((marker, "appeared"))
                else:
                    state.consecutive_appear_rounds = 0
                    state.consecutive_disappear_rounds = 0

            elif state.visibility_score <= hysteresisThresholdLow:
                if state.last_confirmed == MarkerVisibilityState.VISIBLE:
                    state.consecutive_disappear_rounds += 1
                    state.consecutive_appear_rounds = 0
                    if state.consecutive_disappear_rounds >= extinguishingConsecutiveRoundsDelay:
                        state.last_confirmed = MarkerVisibilityState.INVISIBLE
                        state.consecutive_disappear_rounds = 0
                        changes.append((marker, "disappeared"))
                else:
                    state.consecutive_disappear_rounds = 0
                    state.consecutive_appear_rounds = 0

            else:
                # In hysteresis band: do not change confirmed state; reset counters
                state.consecutive_appear_rounds = 0
                state.consecutive_disappear_rounds = 0

        self._markersVisibleLastRound = markersVisibleNow.copy()
        self._markerPositionsLastRound = dict(markerPositions)

        return changes
