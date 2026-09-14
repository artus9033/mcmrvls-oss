"""Map segmentation strategy registry.

Mirrors ``utils.detections.configure_marker_detector``: the strategy is
selected once from config at startup and consumed through
``get_map_segmentation_strategy()`` on the live path.
"""

from __future__ import annotations

from logging import Logger
from typing import Any, Literal

from .base import MapSegmentationStrategy, SegmentationMasks  # noqa: F401 - re-export
from .classical import ClassicalMapSegmentationStrategy

MapSegmentationStrategyName = Literal["classical", "unet"]

_active_strategy: MapSegmentationStrategy | None = None


def configure_map_segmentation(
    strategy: MapSegmentationStrategyName = "classical",
    params: dict[str, Any] | None = None,
    logger: Logger | None = None,
) -> MapSegmentationStrategy:
    """Instantiate and activate the requested strategy.

    If the ``unet`` strategy cannot be initialized (typically: the model
    checkpoint has not been copied to ``res/models/`` on this machine, or
    torch is unavailable), the server falls back to the classical strategy
    with a warning instead of refusing to start.
    """
    global _active_strategy

    params = params or {}
    if strategy == "unet":
        try:
            from .unet import UNetMapSegmentationStrategy

            _active_strategy = UNetMapSegmentationStrategy(params.get("unet"), logger=logger)
        except (FileNotFoundError, ImportError) as e:
            if logger is not None:
                logger.warning(
                    f"UNet map segmentation unavailable ({e}); falling back to classical strategy"
                )
            _active_strategy = ClassicalMapSegmentationStrategy(params.get("classical"))
    elif strategy == "classical":
        _active_strategy = ClassicalMapSegmentationStrategy(params.get("classical"))
    else:
        raise ValueError(f"Unknown map segmentation strategy: {strategy}")

    if logger is not None:
        logger.info(f"Map segmentation strategy: {_active_strategy.name}")
    return _active_strategy


def get_map_segmentation_strategy() -> MapSegmentationStrategy:
    global _active_strategy
    if _active_strategy is None:
        _active_strategy = ClassicalMapSegmentationStrategy()
    return _active_strategy
