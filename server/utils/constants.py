import os

from .lang import genRunDatetimeStr

serverRootPath = os.path.dirname(os.path.dirname(__file__))
"""The root path to the server module's directory"""

outputsWriteDirnameForRun = os.path.join(serverRootPath, "img", "output", genRunDatetimeStr())
"""
unused in notebook, only for standalone script

only used if WRITE_CAPTURES_TO_DISK == True
"""

neutralBorderColorBGR = (0, 0, 0)
"""The neutral border color in BGR format for stitching images."""

disposeStitchingResourceHoldersAfterUnusedRounds: int = 4
"""The number of unused rounds after which the stitching resource holders"""

RECALC_HEURISTIC_DISPLACED_DETECTIONS_MIN_DIFF_NORM_PERCENT_THRESH: float = 0.07  # normalized value
"""
The percentage threshold expressed as a normalized value (0-1) w.r.t.
min image dimension, that is the lowest (minimal) displacement
value for a distance between two coordinates on that axis
to be considered displaced enough to cause a recalculation of homography.
"""

DEFAULT_RECALC_HEURISTIC_APPEARING_MARKER_CONSECUTIVE_ROUNDS_DELAY: int = 4
"""
The rounds after which a newly-appeared marker's state change becomes
effective in homography recalculation heuristics.
Default value if YAML config does not specify the value.

Note: even if a newly-appeared marker's state transition is still "masked" by this
effect, it is still present in the processing algorithm.
"""

DEFAULT_RECALC_HEURISTIC_EXTINGUISHING_MARKER_CONSECUTIVE_ROUNDS_DELAY: int = 4
"""
The rounds after which a newly-extinguished marker's state change becomes
effective in homography recalculation heuristics.
Default value if YAML config does not specify the value.
"""

CROP_IMAGE_TO_ROI_IF_EXCEEDS_BASE_IMAGE_SIZE_PERCENT: float = 0.5
"""
The percentage threshold expressed as a normalized value (0-1) w.r.t.
the appropriate image dimension, that is the lowest (minimal) oversize (offset)
over that dimension size that causes the image to be cropped to the ROI
(coordinates of detected corner points).

E.g. 0.5 would cause the threshold to equal 150% of the base image smallest dimension,
meaning < 150% -> no cropping, >= 150% -> cropping.
"""
