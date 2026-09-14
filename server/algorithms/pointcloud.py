"""
3D point cloud management for non-coplanar AprilTags.

Maintains a projective 3D reconstruction of the scene from multiple camera views.
Points are in projective coordinates (scale is arbitrary), but relative positions
are preserved, which is sufficient for robot tracking and localization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from algorithms.epipolar import ProjectivePoint3D, triangulate_points_uncalibrated
from utils.tracing import Tracing

if TYPE_CHECKING:
    from classes.marker import MarkerData
    from algorithms.camera_calibration import CameraCalibration


# Sentinel distinguishing a tag's centroid from its four corners (indices 0-3)
# in the (marker_id, index) entries fed to update_from_stereo_pair.
CENTROID_INDEX = -1


@dataclass
class PointCloud3D:
    """Maintains 3D positions of markers in projective space."""
    
    points: dict[MarkerData, ProjectivePoint3D] = field(default_factory=dict)
    
    def add_or_update_point(self, marker_id: MarkerData, point: ProjectivePoint3D) -> None:
        """Add or update a 3D point for a marker."""
        self.points[marker_id] = point
    
    def get_point(self, marker_id: MarkerData) -> ProjectivePoint3D | None:
        """Get 3D point for a marker, or None if not available."""
        return self.points.get(marker_id)
    
    def has_point(self, marker_id: MarkerData) -> bool:
        """Check if we have a 3D point for this marker."""
        return marker_id in self.points
    
    def get_all_points(self) -> list[tuple[MarkerData, ProjectivePoint3D]]:
        """Get all points as (marker_id, point) tuples."""
        return list(self.points.items())
    
    def project_to_2d_topdown(
        self,
        reference_plane_z: float = 0.0,
        scale: float = 1.0,
    ) -> dict[MarkerData, tuple[float, float]]:
        """
        Project 3D points to 2D top-down view.
        
        For simplicity, we use an orthographic projection onto the XY plane.
        The Z coordinate determines height above the reference plane.
        
        Args:
            reference_plane_z: Z coordinate of the reference plane
            scale: Scaling factor for projection
        
        Returns:
            Dictionary mapping marker_id to (x, y) in 2D top-down coordinates
        """
        with Tracing.ScopedZone("project_to_2d_topdown"):
            result = {}
            
            for marker_id, point in self.points.items():
                try:
                    xyz = point.xyz
                    # Simple orthographic projection
                    x_2d = float(xyz[0]) * scale
                    y_2d = float(xyz[1]) * scale
                    result[marker_id] = (x_2d, y_2d)
                except (ValueError, ZeroDivisionError):
                    # Point at infinity or invalid
                    continue
            
            return result
    
    def normalize_scale(self, reference_marker_id: MarkerData, target_distance: float) -> None:
        """
        Normalize the scale of the point cloud using a reference marker distance.
        
        This allows us to convert from arbitrary projective scale to a more
        meaningful metric scale if we know the true distance of some marker.
        
        Args:
            reference_marker_id: Marker to use as scale reference
            target_distance: Desired distance from origin for this marker
        """
        with Tracing.ScopedZone("normalize_scale"):
            ref_point = self.get_point(reference_marker_id)
            if ref_point is None:
                return
            
            try:
                xyz = ref_point.xyz
                current_distance = float(np.linalg.norm(xyz))
                
                if current_distance < 1e-6:
                    return
                
                scale_factor = target_distance / current_distance
                
                # Scale all points
                for marker_id in list(self.points.keys()):
                    point = self.points[marker_id]
                    scaled_coords = point.coords.copy()
                    scaled_coords[:3] *= scale_factor
                    self.points[marker_id] = ProjectivePoint3D(
                        coords=scaled_coords,
                        marker_id=point.marker_id,
                    )
            except (ValueError, ZeroDivisionError):
                pass
    
    def clear(self) -> None:
        """Clear all points."""
        self.points.clear()


@dataclass
class PairReconstruction:
    """
    One camera pair's reconstruction, kept self-contained.

    An uncalibrated reconstruction is defined only up to a 3D homography, and
    that homography differs per camera pair. Points from different pairs are
    therefore NOT commensurable and must never be merged before each pair has
    been registered into the map frame. Each pair carries the camera matrices
    it was reconstructed with, since the points are meaningless without them.
    """

    points: dict[MarkerData, ProjectivePoint3D] = field(default_factory=dict)
    # Per-tag reconstructed quads, keyed by marker id -> {corner index: point}.
    # Registration fits over these rather than over `points`, because four
    # spread corners per tag condition a homography far better than one
    # centroid does.
    corner_points: dict[MarkerData, dict[int, ProjectivePoint3D]] = field(default_factory=dict)
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    metric: bool = False  # True when points are already map-frame centimeters
    cam1_id: str | None = None
    cam2_id: str | None = None


@dataclass
class MultiViewReconstructor:
    """
    Manages multi-view 3D reconstruction from multiple cameras.

    Holds one independent reconstruction per camera pair (`pairs`), which is
    what robot measurement uses. `point_cloud` additionally accumulates every
    pair's points in a single dict purely for visualization and for
    availability probes; because the pairs live in different projective frames,
    that merged cloud is not a metrically meaningful structure.
    """

    point_cloud: PointCloud3D = field(default_factory=PointCloud3D)
    pairs: list[PairReconstruction] = field(default_factory=list)
    metric: bool = False  # True when the most recent pair was calibrated

    def update_from_stereo_pair(
        self,
        points_cam1: np.ndarray,
        points_cam2: np.ndarray,
        F: np.ndarray,
        entries: list[tuple[MarkerData, int]],
        calib1: "CameraCalibration | None" = None,
        calib2: "CameraCalibration | None" = None,
        cam1_id: str | None = None,
        cam2_id: str | None = None,
    ) -> None:
        """
        Update the 3D point cloud from a stereo camera pair.

        `entries` labels each triangulated point as a tag centroid
        (CENTROID_INDEX) or one of that tag's four corners (0-3), aligned with
        the point arrays.

        When both cameras have calibrations, triangulates metric points in the
        map frame (centimeters). Otherwise falls back to uncalibrated
        projective reconstruction via F.
        """
        with Tracing.ScopedZone("update_from_stereo_pair"):
            pair = PairReconstruction(cam1_id=cam1_id, cam2_id=cam2_id)

            if calib1 is not None and calib2 is not None:
                from algorithms.camera_calibration import triangulate_points_calibrated
                from algorithms.epipolar import ProjectivePoint3D

                points_xyz = triangulate_points_calibrated(points_cam1, points_cam2, calib1, calib2)
                pair.metric = True
                points_3d = [
                    # Store as homogeneous coords with W=1 (already Euclidean cm).
                    ProjectivePoint3D(
                        coords=np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64),
                        marker_id=int(marker_id) if isinstance(marker_id, (int, np.integer)) else None,
                    )
                    for xyz, (marker_id, _) in zip(points_xyz, entries)
                ]
            else:
                points_3d, P1, P2 = triangulate_points_uncalibrated(
                    points_cam1,
                    points_cam2,
                    F,
                    marker_ids=[
                        int(mid) if isinstance(mid, (int, np.integer)) else mid
                        for mid, _ in entries
                    ],
                )
                pair.P1, pair.P2 = P1, P2
                pair.metric = False

            for point, (marker_id, index) in zip(points_3d, entries):
                if index == CENTROID_INDEX:
                    pair.points[marker_id] = point
                    self.point_cloud.add_or_update_point(marker_id, point)
                else:
                    pair.corner_points.setdefault(marker_id, {})[index] = point

            self.pairs.append(pair)
            self.metric = pair.metric
    
    def get_2d_positions_for_topdown(
        self,
        img_width: int,
        img_height: int,
        anchor_marker_ids: list[MarkerData] | None = None,
        map_width_cm: float | None = None,
        map_height_cm: float | None = None,
    ) -> dict[MarkerData, tuple[float, float]]:
        """
        Get 2D positions for rendering on top-down view.

        Metric clouds (calibrated) are mapped with known map dimensions into
        pixel coordinates. Projective clouds return raw XY for external registration.
        """
        del anchor_marker_ids
        with Tracing.ScopedZone("get_2d_positions_for_topdown"):
            points_2d = self.point_cloud.project_to_2d_topdown()
            if not points_2d:
                return {}

            if self.metric and map_width_cm and map_height_cm and map_width_cm > 0 and map_height_cm > 0:
                return {
                    marker_id: (
                        float(x) / float(map_width_cm) * float(img_width),
                        float(y) / float(map_height_cm) * float(img_height),
                    )
                    for marker_id, (x, y) in points_2d.items()
                }

            del img_width, img_height
            return points_2d
    
    def clear(self) -> None:
        """Clear the point cloud."""
        self.point_cloud.clear()
        self.pairs.clear()
        self.metric = False
