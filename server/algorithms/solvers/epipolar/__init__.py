"""
Epipolar geometry solver for non-coplanar scenes.

Uses fundamental matrix and 3D triangulation to handle markers that don't
lie on a single plane. No camera calibration required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algorithms.solvers import GeometrySolver, SolverResult, SolverState
from algorithms.epipolar import (
    EpipolarGeometry,
    compute_fundamental_matrix_from_detections,
    detection_to_points,
)
from algorithms.pointcloud import CENTROID_INDEX, MultiViewReconstructor
from utils.tracing import Tracing

if TYPE_CHECKING:
    from classes.marker import CompositeDetection, MarkerDetection
    from classes.types import MarkerData


class EpipolarState(SolverState):
    """State for epipolar-based solver with 3D reconstruction."""
    
    def __init__(self):
        self.F_cache: dict[str, EpipolarGeometry] = {}
        self.reconstructor = MultiViewReconstructor()
        self._dirty = True
    
    def get_camera_pair_key(self, cam1_id: int | str, cam2_id: int | str) -> str:
        """Get cache key for camera pair."""
        ids = sorted([str(cam1_id), str(cam2_id)])
        return f"{ids[0]}_{ids[1]}"
    
    def get_fundamental_matrix(self, cam1_id: int | str, cam2_id: int | str) -> EpipolarGeometry | None:
        """Get cached fundamental matrix for camera pair."""
        key = self.get_camera_pair_key(cam1_id, cam2_id)
        return self.F_cache.get(key)
    
    def set_fundamental_matrix(
        self,
        cam1_id: int | str,
        cam2_id: int | str,
        epipolar_geom: EpipolarGeometry,
    ) -> None:
        """Cache fundamental matrix for camera pair."""
        key = self.get_camera_pair_key(cam1_id, cam2_id)
        self.F_cache[key] = epipolar_geom
        self._dirty = True
    
    def clear(self) -> None:
        """Clear all cached state."""
        self.F_cache.clear()
        self.reconstructor.clear()
        self._dirty = True
    
    def is_dirty(self) -> bool:
        """Check if state needs recalculation."""
        return self._dirty
    
    def mark_clean(self) -> None:
        """Mark state as clean."""
        self._dirty = False


class EpipolarSolver(GeometrySolver):
    """
    Epipolar geometry solver with 3D reconstruction.
    
    Uses fundamental matrix and triangulation to handle non-coplanar markers.
    No camera calibration required - works with uncalibrated cameras.
    """
    
    def __init__(self, ransac_threshold: float = 3.0):
        self.ransac_threshold = ransac_threshold
    
    def create_state(self) -> EpipolarState:
        """Create a new epipolar state instance."""
        return EpipolarState()
    
    def process_camera_pair(
        self,
        detections_cam1: list[MarkerDetection | CompositeDetection],
        detections_cam2: list[MarkerDetection | CompositeDetection],
        cam1_id: int | str,
        cam2_id: int | str,
        state: SolverState,
        use_cache: bool = True,
        calib1: Any = None,
        calib2: Any = None,
    ) -> SolverResult:
        """
        Process camera pair using epipolar geometry.
        
        Computes fundamental matrix F and triangulates 3D points.
        When calib1/calib2 are provided, triangulation is metric (map cm).
        """
        if not isinstance(state, EpipolarState):
            raise TypeError(f"Expected EpipolarState, got {type(state)}")
        
        with Tracing.ScopedZone("EpipolarSolver.process_camera_pair"):
            # Try cache first
            epipolar_geom = None
            if use_cache:
                epipolar_geom = state.get_fundamental_matrix(cam1_id, cam2_id)
            
            # Compute if not cached
            if epipolar_geom is None:
                # Filter to common detections
                cam1_common, cam2_common = self._get_common_detections(
                    detections_cam1, detections_cam2
                )
                
                # Compute fundamental matrix
                epipolar_geom = compute_fundamental_matrix_from_detections(
                    cam1_common,
                    cam2_common,
                    ransac_threshold=self.ransac_threshold,
                )
                
                # Cache it
                state.set_fundamental_matrix(cam1_id, cam2_id, epipolar_geom)
            
            # Triangulate and update point cloud
            cam1_common, cam2_common = self._get_common_detections(
                detections_cam1, detections_cam2
            )
            self._triangulate_and_update(
                cam1_common,
                cam2_common,
                epipolar_geom,
                state,
                calib1=calib1,
                calib2=calib2,
                cam1_id=cam1_id,
                cam2_id=cam2_id,
            )
            
            return SolverResult(
                transform_matrix=epipolar_geom.F,
                inlier_mask=epipolar_geom.inlier_mask,
                positions_2d={},  # Computed in render_to_topdown
                metadata={
                    "solver_type": "epipolar",
                    "metric": bool(calib1 is not None and calib2 is not None),
                },
            )
    
    def get_robot_measurements(
        self,
        detections: list[MarkerDetection | CompositeDetection],
        state: SolverState,
        config: Any,
    ) -> list[Any]:
        """
        Extract 3D robot measurements from detections.
        
        For epipolar solver, measurements are in 3D coordinates.
        """
        from algorithms.robot_kalman_tracker_3d import RobotTagMeasurement3D
        from algorithms.robot_kalman_tracker import (
            compute_tag_measurement_sigmas,
            quad_area_tl_tr_bl_br,
        )
        import math
        
        if not isinstance(state, EpipolarState):
            raise TypeError(f"Expected EpipolarState, got {type(state)}")
        
        measurements = []
        
        for detection in detections:
            found_robot = config.findRobotWithID(detection.data)
            if found_robot is None:
                continue
            
            # Get 3D position from point cloud
            point_3d = state.reconstructor.point_cloud.get_point(detection.data)
            if point_3d is None:
                continue
            
            try:
                xyz = point_3d.xyz
            except (ValueError, ZeroDivisionError):
                continue  # Point at infinity
            
            # Get detection coordinates for uncertainty computation
            if hasattr(detection, "aggregatedTopLeft"):
                tl = detection.aggregatedTopLeft
                tr = detection.aggregatedTopRight
                bl = detection.aggregatedBottomLeft
                br = detection.aggregatedBottomRight
            else:
                tl = detection.topLeft
                tr = detection.topRight
                bl = detection.bottomLeft
                br = detection.bottomRight
            
            # Compute area and baseline for uncertainty
            area_px = quad_area_tl_tr_bl_br(tl, tr, bl, br)
            dx = float(tr[0] - tl[0])
            dy = float(tr[1] - tl[1])
            baseline_px = float(np.hypot(dx, dy))
            
            # Compute heading (azimuth) from 2D projection
            front_axle = ((tl[0] + tr[0]) / 2, (tl[1] + tr[1]) / 2)
            rear_axle = ((bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2)
            dx_heading = front_axle[0] - rear_axle[0]
            dy_heading = front_axle[1] - rear_axle[1]
            azimuth = math.degrees(math.atan2(dx_heading, -dy_heading)) % 360.0
            
            # Pitch angle (for 3D orientation)
            pitch = 0.0  # Assume horizontal for now
            
            # Compute measurement uncertainties
            img_w = 1920  # Default
            img_h = 1080
            
            sx, sy, spsi = compute_tag_measurement_sigmas(
                area_px=area_px,
                baseline_px=baseline_px,
                img_w=img_w,
                img_h=img_h,
                p=found_robot.kalman_params,
            )
            
            # Add z uncertainty (proportional to depth)
            sz = sx * 2.0  # Depth uncertainty typically larger
            
            measurements.append(
                RobotTagMeasurement3D(
                    robot=found_robot,
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2]),
                    azimuth_deg=float(azimuth),
                    pitch_deg=float(pitch),
                    sigma_x=sx,
                    sigma_y=sy,
                    sigma_z=sz,
                    sigma_psi_rad=spsi,
                    sigma_theta_rad=math.radians(5.0),  # Default pitch uncertainty
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
        Render 3D point cloud to 2D top-down view.
        
        Projects 3D points to 2D and scales to image coordinates.
        """
        if not isinstance(state, EpipolarState):
            raise TypeError(f"Expected EpipolarState, got {type(state)}")
        
        with Tracing.ScopedZone("EpipolarSolver.render_to_topdown"):
            map_w = None
            map_h = None
            # Prefer metric mapping when the reconstructor was fed calibrated pairs.
            return state.reconstructor.get_2d_positions_for_topdown(
                img_width,
                img_height,
                anchor_marker_ids=list(corner_markers),
                map_width_cm=map_w,
                map_height_cm=map_h,
            )
    
    def _triangulate_and_update(
        self,
        detections_cam1: list[MarkerDetection | CompositeDetection],
        detections_cam2: list[MarkerDetection | CompositeDetection],
        epipolar_geom: EpipolarGeometry,
        state: EpipolarState,
        calib1: Any = None,
        calib2: Any = None,
        cam1_id: str | None = None,
        cam2_id: str | None = None,
    ) -> None:
        """Triangulate 3D points and update the point cloud."""
        with Tracing.ScopedZone("triangulate_and_update"):
            # Collect point correspondences
            points1_list = []
            points2_list = []
            # Triangulate each tag's four corners as well as its centroid.
            # Map registration fits a homography over the reconstructed floor,
            # and a homography over single points per tag is constrained only
            # by however those few centroids happen to be spread; the corners
            # give four well-separated clusters instead, which is what keeps
            # the perspective terms from extrapolating wildly.
            entries: list[tuple[Any, int]] = []

            def quad_or_none(detection):
                if hasattr(detection, "toAggregatedPointsList"):
                    return detection.toAggregatedPointsList()
                if hasattr(detection, "toPointsList"):
                    return detection.toPointsList()
                return None

            def centroid_of(detection):
                if hasattr(detection, "aggregatedCentroid"):
                    return detection.aggregatedCentroid
                return detection.centroid

            for d1 in detections_cam1:
                d2 = next((d for d in detections_cam2 if d.data == d1.data), None)
                if d2 is None:
                    continue

                quad1, quad2 = quad_or_none(d1), quad_or_none(d2)
                if quad1 is not None and quad2 is not None and len(quad1) == len(quad2) == 4:
                    for corner_index, (c1, c2) in enumerate(zip(quad1, quad2)):
                        points1_list.append(c1)
                        points2_list.append(c2)
                        entries.append((d1.data, corner_index))

                points1_list.append(centroid_of(d1))
                points2_list.append(centroid_of(d2))
                entries.append((d1.data, CENTROID_INDEX))
            
            if not points1_list:
                return
            
            points1 = np.array(points1_list, dtype=np.float32).reshape(-1, 2)
            points2 = np.array(points2_list, dtype=np.float32).reshape(-1, 2)
            
            # Update point cloud
            state.reconstructor.update_from_stereo_pair(
                points1,
                points2,
                epipolar_geom.F,
                entries,
                calib1=calib1,
                calib2=calib2,
                cam1_id=str(cam1_id) if cam1_id is not None else None,
                cam2_id=str(cam2_id) if cam2_id is not None else None,
            )
    
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
