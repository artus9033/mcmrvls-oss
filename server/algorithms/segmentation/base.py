"""Abstract base class and result container for map segmentation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationMasks:
    """Segmentation output masks, uint8 {0, 255}, same HxW as the input image."""

    roadComponents: np.ndarray
    redStopLines: np.ndarray
    yellowLines: np.ndarray


class MapSegmentationStrategy(ABC):
    """A pluggable map segmentation backend.

    Implementations receive the top-down BGR map image and an optional robot
    exclusion mask (uint8 {0, 255}; areas occupied by robots that must be
    treated as navigable road) and produce the three masks consumed by the
    rest of the pipeline.
    """

    name: str = "abstract"

    def warmup(self, image_shape: tuple[int, int]) -> None:
        """Optional: run any one-off setup for the given (height, width)."""

    @abstractmethod
    def segment(
        self,
        imageBGR: np.ndarray,
        robotMask: np.ndarray | None = None,
    ) -> SegmentationMasks | None:
        """Segment the top-down map image; None signals 'no usable result'."""
