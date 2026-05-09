"""
Adaptive numpy buffer allocator based on queuing systems theory.

See docs/source/adaptive_preallocation.md for the full algorithm description
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
import threading
from typing import Callable

import numpy as np

_logger = getLogger(__name__)


@dataclass
class AdaptiveBufferAllocatorConfig:
    """Configuration for the adaptive preallocation algorithm."""

    growth_factor: float = 1.25
    """Multiplicative factor when enlarging capacity (queuing: prevents starvation)."""

    shrink_utilization_threshold: float = 0.5
    """Only shrink when observed usage / capacity < this (reservoir drain threshold)."""

    shrink_consecutive_rounds: int = 10
    """Number of consecutive rounds below threshold before shrinking (hysteresis)."""

    ewma_alpha: float = 0.15
    """Smoothing factor for EWMA of observed demand (λ in Little's Law analogy)."""

    min_capacity: int = 64
    """Minimum allocated dimension to avoid tiny buffers."""

    dtype: np.dtype = np.uint8
    """Default dtype for image buffers."""


@dataclass
class BufferSlot:
    """A preallocated buffer with metadata."""

    buffer: np.ndarray
    """The underlying numpy array (may be larger than needed)."""

    slot_height: int
    slot_width: int
    """The logical size of the slot (used region)."""

    allocator_id: str
    """Identifier for logging and WebSocket events."""


class AdaptiveBufferAllocator:
    """
    Adaptively preallocates numpy buffers using queuing-theory-inspired heuristics.

    The algorithm combines:
    - EWMA of observed demand (steady-state demand estimation)
    - Growth with multiplicative factor (prevents allocation thrashing)
    - Shrink with hysteresis (avoids oscillating shrink/grow)
    """

    def __init__(
        self,
        allocator_id: str,
        dtype: np.dtype = np.uint8,
        config: AdaptiveBufferAllocatorConfig | None = None,
        on_reallocate: Callable[[str, str, tuple[int, ...], tuple[int, ...]], None] | None = None,
    ) -> None:
        self.allocator_id = allocator_id
        self._dtype = dtype
        self._config = config or AdaptiveBufferAllocatorConfig()
        self._on_reallocate = on_reallocate

        self._buffer: np.ndarray | None = None
        self._current_height: int = 0
        self._current_width: int = 0

        self._ewma_height: float = 0.0
        self._ewma_width: float = 0.0
        self._rounds_below_threshold: int = 0
        self._lock = threading.RLock()

    def ensure_capacity_for(self, height: int, width: int, channels: int = 3) -> BufferSlot:
        """
        Ensure the allocator has capacity for at least (height, width, channels).
        Returns a BufferSlot with a view into the buffer for the requested size.
        """
        with self._lock:
            if self._buffer is not None and height <= self._current_height and width <= self._current_width:
                slot = self._buffer[:height, :width]
                return BufferSlot(
                    buffer=slot,
                    slot_height=height,
                    slot_width=width,
                    allocator_id=self.allocator_id,
                )

            needed_h = max(height, self._config.min_capacity)
            needed_w = max(width, self._config.min_capacity)

            # Update EWMA of observed demand
            alpha = self._config.ewma_alpha
            if self._ewma_height == 0:
                self._ewma_height = float(needed_h)
                self._ewma_width = float(needed_w)
            else:
                self._ewma_height = alpha * needed_h + (1 - alpha) * self._ewma_height
                self._ewma_width = alpha * needed_w + (1 - alpha) * self._ewma_width

            old_shape = (self._current_height, self._current_width) if self._buffer is not None else (0, 0)
            reallocated = False

            if self._buffer is None:
                # Initial allocation: use requested size with growth factor headroom
                cap_h = int(needed_h * self._config.growth_factor)
                cap_w = int(needed_w * self._config.growth_factor)
                self._buffer = np.zeros((cap_h, cap_w, channels), dtype=self._dtype)
                self._current_height = cap_h
                self._current_width = cap_w
                reallocated = True
                _logger.debug(
                    "%s: initial alloc (h=%d, w=%d)",
                    self.allocator_id,
                    cap_h,
                    cap_w,
                )
            elif needed_h > self._current_height or needed_w > self._current_width:
                # Enlarge: multiplicative growth (queuing: prevent starvation under burst)
                new_h = max(
                    int(needed_h * self._config.growth_factor),
                    self._current_height,
                )
                new_w = max(
                    int(needed_w * self._config.growth_factor),
                    self._current_width,
                )
                self._buffer = np.zeros((new_h, new_w, channels), dtype=self._dtype)
                self._current_height = new_h
                self._current_width = new_w
                reallocated = True
                _logger.debug(
                    "%s: enlarged (h=%d->%d, w=%d->%d)",
                    self.allocator_id,
                    old_shape[0],
                    new_h,
                    old_shape[1],
                    new_w,
                )
                if self._on_reallocate:
                    self._on_reallocate(
                        self.allocator_id,
                        "enlarge",
                        old_shape,
                        (new_h, new_w),
                    )
            else:
                # Check shrink condition (hysteresis)
                util_h = needed_h / self._current_height
                util_w = needed_w / self._current_width
                utilization = min(util_h, util_w)

                if utilization < self._config.shrink_utilization_threshold:
                    self._rounds_below_threshold += 1
                    if self._rounds_below_threshold >= self._config.shrink_consecutive_rounds:
                        # Shrink based on EWMA (smoothed demand)
                        new_h = max(
                            int(self._ewma_height * self._config.growth_factor),
                            needed_h,
                            self._config.min_capacity,
                        )
                        new_w = max(
                            int(self._ewma_width * self._config.growth_factor),
                            needed_w,
                            self._config.min_capacity,
                        )
                        if new_h < self._current_height or new_w < self._current_width:
                            self._buffer = np.zeros((new_h, new_w, channels), dtype=self._dtype)
                            self._current_height = new_h
                            self._current_width = new_w
                            _logger.debug(
                                "%s: shrunk (h=%d->%d, w=%d->%d)",
                                self.allocator_id,
                                old_shape[0],
                                new_h,
                                old_shape[1],
                                new_w,
                            )
                            if self._on_reallocate:
                                self._on_reallocate(
                                    self.allocator_id,
                                    "shrink",
                                    old_shape,
                                    (new_h, new_w),
                                )
                        self._rounds_below_threshold = 0
                else:
                    self._rounds_below_threshold = 0

            assert self._buffer is not None
            slot = self._buffer[:height, :width]
            return BufferSlot(
                buffer=slot,
                slot_height=height,
                slot_width=width,
                allocator_id=self.allocator_id,
            )

    def get_view(self, height: int, width: int) -> np.ndarray | None:
        """
        Get a view of the buffer for the given size, or None if not allocated or too small.
        """
        with self._lock:
            if self._buffer is None:
                return None
            if height > self._current_height or width > self._current_width:
                return None
            return self._buffer[:height, :width]

    def get_full_buffer_shape(self) -> tuple[int, int] | None:
        """Return (height, width) of the full allocated buffer, or None."""
        with self._lock:
            if self._buffer is None:
                return None
            return (self._current_height, self._current_width)
