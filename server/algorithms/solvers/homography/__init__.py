"""
Homography-based solver for planar scenes.

Assumes all markers lie on a single plane and uses homography transformations
to relate camera views. This is the original approach used by the system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.solvers import GeometrySolver, SolverResult, SolverState
from algorithms.homography import calculateHomography
from utils.tracing import Tracing

if TYPE_CHECKING:
    from classes.marker import CompositeDetection, MarkerDetection
    from classes.types import MarkerData


class HomographyState(SolverState):
    """State for homography-based solver."""
    
    def __init__(self):
        self.H_cache: dict[str, np.ndarray] = {}
        self._dirty = True
    
    def get_camera_pair_key(self, cam1_id: int | str, cam2_id: int | str) -> str:
        """Get cache key for camera pair."""
        ids = sorted([str(cam1_id), str(cam2_id)])
        return f"{ids[0]}_{ids[1]}"
    
    def get_homography(self, cam1_id: int | str, cam2_id: int | str) -> np.ndarray | None:
        """Get cached homography for camera pair."""
        key = self.get_camera_pair_key(cam1_id, cam2_id)
        return self.H_cache.get(key)
    
    def set_homography(self, cam1_id: int | str, cam2_id: int | str, H: np.ndarray) -> None:
        """Cache homography for camera pair."""
        key = self.get_camera_pair_key(cam1_id, cam2_id)
        self.H_cache[key] = H
        self._dirty = True
    
    def clear(self) -> None:
        """Clear all cached state."""
        self.H_cache.clear()
        self._dirty = True
    
    def is_dirty(self) -> bool:
        """Check if state needs recalculation."""
        return self._dirty
    
    def mark_clean(self) -> None:
        """Mark state as clean."""
        self._dirty = False


class HomographySolver(GeometrySolver):
    """
    Homography-based geometry solver.
    
    Uses planar homography to relate camera views. Best for scenes where
    all markers lie on a flat surface.
    """
    
    def __init__(self, ransac_threshold: float = 5.0):
        self.ransac_threshold = ransac_threshold
    
    def create_state(self) -> HomographyState:
        """Create a new homography state instance."""
        return HomographyState()
    
    def process_camera_pair(
        self,
        detections_cam1: list[MarkerDetection | CompositeDetection],
        detections_cam2: list[MarkerDetection | CompositeDetection],
        cam1_id: int | str,
        cam2_id: int | str,
        state: SolverState,
        use_cache: bool = True,
    ) -> SolverResult:
        """
        Process camera pair using homography.
        
        Computes homography matrix H such that points in cam2 = H @ points in cam1.
        """
        if not isinstance(state, HomographyState):
            raise TypeError(f"Expected HomographyState, got {type(state)}")
        
        with Tracing.ScopedZone("HomographySolver.process_camera_pair"):
            # Try cache first
            H = None
            if use_cache:
                H = state.get_homography(cam1_id, cam2_id)
            
            # Compute if not cached
            if H is None:
                # Filter to common detections
                cam1_common, cam2_common = self._get_common_detections(
                    detections_cam1, detections_cam2
                )
                
                # Compute homography
                H = calculateHomography(cam2_common, cam1_common)
                
                # Cache it
                state.set_homography(cam1_id, cam2_id, H)
            
            # For homography, we don't compute 2D positions here
            # They're computed later via warp operations
            return SolverResult(
                transform_matrix=H,
                inlier_mask=None,
                positions_2d={},
                metadata={"solver_type": "homography"},
            )
    
    def get_robot_measurements(
        self,
        detections: list[MarkerDetection | CompositeDetection],
        state: SolverState,
        config: Any,
    ) -> list[Any]:
        """
        Extract 2D robot measurements from detections.
        
        For homography solver, measurements are in 2D normalized coordinates.
        """
        from algorithms.robot_kalman_tracker import (
            RobotTagMeasurement,
            compute_tag_measurement_sigmas,
            quad_area_tl_tr_bl_br,
        )
        import math
        
        measurements = []
        
        for detection in detections:
            found_robot = config.findRobotWithID(detection.data)
            if found_robot is None:
                continue
            
            # Get detection coordinates
            if hasattr(detection, "aggregatedTopLeft"):
                tl = detection.aggregatedTopLeft
                tr = detection.aggregatedTopRight
                bl = detection.aggregatedBottomLeft
                br = detection.aggregatedBottomRight
                centroid = detection.aggregatedCentroid
            else:
                tl = detection.topLeft
                tr = detection.topRight
                bl = detection.bottomLeft
                br = detection.bottomRight
                centroid = detection.centroid
            
            # Compute area and baseline for uncertainty
            area_px = quad_area_tl_tr_bl_br(tl, tr, bl, br)
            dx = float(tr[0] - tl[0])
            dy = float(tr[1] - tl[1])
            baseline_px = float(np.hypot(dx, dy))
            
            # Compute heading (azimuth)
            front_axle = ((tl[0] + tr[0]) / 2, (tl[1] + tr[1]) / 2)
            rear_axle = ((bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2)
            dx_heading = front_axle[0] - rear_axle[0]
            dy_heading = front_axle[1] - rear_axle[1]
            azimuth = math.degrees(math.atan2(dx_heading, -dy_heading)) % 360.0
            
            # Compute measurement uncertainties
            # Assume we have image dimensions in metadata or use detection bounds
            img_w = 1920  # Default, should come from config
            img_h = 1080
            
            sx, sy, spsi = compute_tag_measurement_sigmas(
                area_px=area_px,
                baseline_px=baseline_px,
                img_w=img_w,
                img_h=img_h,
                p=found_robot.kalman_params,
            )
            
            measurements.append(
                RobotTagMeasurement(
                    robot=found_robot,
                    x=float(centroid[0]) / img_w,  # Normalize
                    y=float(centroid[1]) / img_h,
                    azimuth_deg=float(azimuth),
                    sigma_x=sx,
                    sigma_y=sy,
                    sigma_psi_rad=spsi,
                    composite=detection,
                )
            )
        
        return measurements
    
    def render_to_topdown(
        self,
        state: SolverState,
        img_width: int,
        img_height: int,
        corner_markers: list[MarkerData],
    ) -> dict[MarkerData, tuple[float, float]]:
        """
        Render markers to top-down view using homography.
        
        For homography solver, this is handled by the existing warp pipeline.
        Returns empty dict as positions are computed during warping.
        """
        return {}
    
    @staticmethod
    def _get_common_detections(
        detections_cam1: list[MarkerDetection | CompositeDetection],
        detections_cam2: list[MarkerDetection | CompositeDetection],
    ) -> tuple[list[MarkerDetection | CompositeDetection], list[MarkerDetection | CompositeDetection]]:
        """Filter to only common detections between cameras."""
        from errors import ProcessingError
        
        ids_cam1 = {d.data for d in detections_cam1}
        ids_cam2 = {d.data for d in detections_cam2}
        common_ids = ids_cam1 & ids_cam2
        
        if not common_ids:
            raise ProcessingError(
                "No common detections found between cameras",
                code="NO_COMMON_DETECTIONS",
            )
        
        cam1_common = [d for d in detections_cam1 if d.data in common_ids]
        cam2_common = [d for d in detections_cam2 if d.data in common_ids]
        
        return cam1_common, cam2_common
