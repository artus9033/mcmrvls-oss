from functools import total_ordering
from typing import Iterable, cast

from classes.marker import CompositeDetection, MarkerDetection
from errors.ProcessingError import ProcessingError
import numpy as np
from utils.detections import getDetections
from utils.tracing import Tracing

from ..types import MarkerData
from .StitchingStageDescription import StitchingStageDescription

@total_ordering
class ImagePack:
    def __init__(
        self,
        image: np.ndarray | None,
        stitchingStageDescription: StitchingStageDescription,
        isStitchingResult: bool = False,
        dirty: bool = False,
    ) -> None:
        self.image = image
        self.stitchingStageDescription = stitchingStageDescription
        self.dirty = dirty

        self.compositeDetectionsStore: dict[MarkerData, CompositeDetection] = {}

        self.isStitchingResult = isStitchingResult

        # Cumulative homography from each source camera's raw pixels into this
        # pack's image frame, keyed by the camera's stage id. Raw packs map to
        # themselves; the stitcher composes these through every stage so any
        # raw-camera point can be placed exactly on the final mosaic.
        self.cameraWarps: dict[str, np.ndarray] = {}
        self.cameraSizes: dict[str, tuple[int, int]] = {}
        if not self.isStitchingResult:
            self.cameraWarps[str(stitchingStageDescription)] = np.eye(3, dtype=np.float64)
            if image is not None:
                self.cameraSizes[str(stitchingStageDescription)] = (image.shape[1], image.shape[0])

        if not self.isStitchingResult:
            self.findAndAppendDetections()

    def findAndAppendDetections(self):
        with Tracing.ScopedZone("findAndAppendDetections"):
            if self.image is None:
                raise ProcessingError(
                    f"ImagePack '{self.stitchingStageDescription}' that is not a result of stitching has no image",
                    "NONSTITCHED_IMAGE_PACK_HAS_NO_IMAGE",
                )

            detections = getDetections(self.image)

            self.appendStitchedDetections(detections)

    def appendStitchedDetections(self, detections: Iterable[MarkerDetection | CompositeDetection]):
        with Tracing.ScopedZone("appendStitchedDetections"):
            for detection in detections:
                if detection.data in self.compositeDetectionsStore:
                    if isinstance(detection, CompositeDetection):
                        # a composite detection is being appended to an existing composite detection
                        for subDetection in cast(CompositeDetection, detection).detections:
                            self.compositeDetectionsStore[detection.data].appendDetection(subDetection)
                    else:
                        # a marker detection is being appended to an existing composite detection
                        self.compositeDetectionsStore[detection.data].appendDetection(cast(MarkerDetection, detection))
                else:
                    if isinstance(detection, CompositeDetection):
                        # a composite detection is being inserted as a new composite detection
                        self.compositeDetectionsStore[detection.data] = cast(CompositeDetection, detection)
                    else:
                        # a marker detection is being inserted as a new composite detection
                        self.compositeDetectionsStore[detection.data] = CompositeDetection(detection.data, [detection])

    def __str__(self) -> str:
        return f"ImagePack<stitchingStageDescription={self.stitchingStageDescription}, detections=[{', '.join([f'{key} x{len(value)}' for key, value in self.compositeDetectionsStore.items()])}]>"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other):
        if isinstance(other, ImagePack):
            return self.stitchingStageDescription == other.stitchingStageDescription
        else:
            return False

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ImagePack):
            return NotImplemented

        return self.stitchingStageDescription < other.stitchingStageDescription

    def __hash__(self) -> MarkerData:
        return hash(self.stitchingStageDescription)
