"""
Epipolar-based stitching for non-coplanar AprilTags.

This is an alternative to homography-based stitching that uses epipolar geometry
and 3D reconstruction. It handles non-coplanar markers by maintaining a 3D point cloud.

Key differences from homography approach:
- Uses fundamental matrix (F) instead of homography (H)
- Maintains 3D point cloud via triangulation
- Projects 3D points to 2D for final output
- No assumption of planarity
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np

from algorithms.epipolar import (
    EpipolarGeometry,
    compute_fundamental_matrix_from_detections,
    detection_to_points,
)
from algorithms.pointcloud import MultiViewReconstructor
from classes.marker import MarkerDetection
from classes.marker.CompositeDetection import CompositeDetection
from errors import ProcessingError
from utils.tracing import Tracing

if TYPE_CHECKING:
    from classes.types import MarkerData


class EpipolarStitchingState:
    """
    State for epipolar-based stitching pipeline.
    
    Replaces homography caching with fundamental matrix and 3D reconstruction.
    """
    
    def __init__(self):
        self.F_cache: dict[str, EpipolarGeometry] = {}  # Camera pair -> F matrix
        self.reconstructor = MultiViewReconstructor()
        self.dirty = True
    
    def get_camera_pair_key(self, cam1_id: int | str, cam2_id: int | str) -> str:
        """Get cache key for camera pair (order-independent)."""
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
        self.dirty = True
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.F_cache.clear()
        self.reconstructor.clear()
        self.dirty = True


def compute_or_get_fundamental_matrix(
    detections_cam1: list[MarkerDetection | CompositeDetection],
    detections_cam2: list[MarkerDetection | CompositeDetection],
    cam1_id: int | str,
    cam2_id: int | str,
    state: EpipolarStitchingState,
    use_cache: bool = True,
    ransac_threshold: float = 3.0,
) -> EpipolarGeometry:
    """
    Compute fundamental matrix or retrieve from cache.
    
    Args:
        detections_cam1: Detections in first camera
        detections_cam2: Detections in second camera
        cam1_id: First camera ID
        cam2_id: Second camera ID
        state: Stitching state with cache
        use_cache: Whether to use cached F matrix
        ransac_threshold: RANSAC threshold for F estimation
    
    Returns:
        EpipolarGeometry with fundamental matrix
    """
    with Tracing.ScopedZone("compute_or_get_fundamental_matrix"):
        # Try cache first
        if use_cache:
            cached = state.get_fundamental_matrix(cam1_id, cam2_id)
            if cached is not None:
                return cached
        
        # Compute new fundamental matrix
        epipolar_geom = compute_fundamental_matrix_from_detections(
            detections_cam1,
            detections_cam2,
            ransac_threshold=ransac_threshold,
        )
        
        # Cache it
        state.set_fundamental_matrix(cam1_id, cam2_id, epipolar_geom)
        
        return epipolar_geom


def triangulate_and_update_point_cloud(
    detections_cam1: list[MarkerDetection | CompositeDetection],
    detections_cam2: list[MarkerDetection | CompositeDetection],
    epipolar_geom: EpipolarGeometry,
    state: EpipolarStitchingState,
) -> None:
    """
    Triangulate 3D points from two camera views and update the point cloud.
    
    Args:
        detections_cam1: Detections in first camera
        detections_cam2: Detections in second camera
        epipolar_geom: Epipolar geometry with F matrix
        state: Stitching state with point cloud
    """
    with Tracing.ScopedZone("triangulate_and_update_point_cloud"):
        # Collect point correspondences
        points1_list = []
        points2_list = []
        marker_ids_list = []
        
        for d1 in detections_cam1:
            # Find corresponding detection in cam2
            d2 = next((d for d in detections_cam2 if d.data == d1.data), None)
            if d2 is not None:
                # Use centroid for triangulation
                if hasattr(d1, "aggregatedCentroid"):
                    p1 = d1.aggregatedCentroid
                else:
                    p1 = d1.centroid
                
                if hasattr(d2, "aggregatedCentroid"):
                    p2 = d2.aggregatedCentroid
                else:
                    p2 = d2.centroid
                
                points1_list.append(p1)
                points2_list.append(p2)
                marker_ids_list.append(d1.data)
        
        if not points1_list:
            return
        
        points1 = np.array(points1_list, dtype=np.float32).reshape(-1, 2)
        points2 = np.array(points2_list, dtype=np.float32).reshape(-1, 2)
        
        # Update point cloud with triangulated points
        state.reconstructor.update_from_stereo_pair(
            points1,
            points2,
            epipolar_geom.F,
            marker_ids_list,
        )


def get_common_detections(
    detections_cam1: list[MarkerDetection | CompositeDetection],
    detections_cam2: list[MarkerDetection | CompositeDetection],
) -> tuple[list[MarkerDetection | CompositeDetection], list[MarkerDetection | CompositeDetection]]:
    """
    Filter to only common detections between two cameras.
    
    Returns:
        Tuple of (cam1_common, cam2_common) with matching markers
    """
    with Tracing.ScopedZone("get_common_detections"):
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


def process_camera_pair_epipolar(
    detections_cam1: list[MarkerDetection | CompositeDetection],
    detections_cam2: list[MarkerDetection | CompositeDetection],
    cam1_id: int | str,
    cam2_id: int | str,
    state: EpipolarStitchingState,
    use_cache: bool = True,
) -> EpipolarGeometry:
    """
    Process a camera pair using epipolar geometry.
    
    Computes fundamental matrix and updates 3D point cloud.
    
    Args:
        detections_cam1: Detections in first camera
        detections_cam2: Detections in second camera
        cam1_id: First camera ID
        cam2_id: Second camera ID
        state: Stitching state
        use_cache: Whether to use cached F matrix
    
    Returns:
        EpipolarGeometry with fundamental matrix
    """
    with Tracing.ScopedZone("process_camera_pair_epipolar"):
        # Get common detections
        cam1_common, cam2_common = get_common_detections(detections_cam1, detections_cam2)
        
        # Compute or get fundamental matrix
        epipolar_geom = compute_or_get_fundamental_matrix(
            cam1_common,
            cam2_common,
            cam1_id,
            cam2_id,
            state,
            use_cache=use_cache,
        )
        
        # Triangulate and update point cloud
        triangulate_and_update_point_cloud(
            cam1_common,
            cam2_common,
            epipolar_geom,
            state,
        )
        
        return epipolar_geom


def render_3d_to_2d_topdown(
    state: EpipolarStitchingState,
    img_width: int,
    img_height: int,
    corner_markers: list[MarkerData],
) -> dict[MarkerData, tuple[float, float]]:
    """
    Render 3D point cloud to 2D top-down view.
    
    Args:
        state: Stitching state with point cloud
        img_width: Width of output image
        img_height: Height of output image
        corner_markers: List of marker IDs that represent map corners
    
    Returns:
        Dictionary mapping marker ID to (x, y) pixel coordinates
    """
    with Tracing.ScopedZone("render_3d_to_2d_topdown"):
        return state.reconstructor.get_2d_positions_for_topdown(img_width, img_height)
