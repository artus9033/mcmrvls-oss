"""
Uncalibrated epipolar geometry for non-coplanar AprilTag support.

This module implements epipolar geometry without requiring camera calibration matrices.
Instead of homography (which assumes planarity), we use:
- Fundamental matrix (F) to capture epipolar constraints
- Projective reconstruction for 3D triangulation
- Relative geometry (no metric scale required)

Key insight: For robot tracking, relative positions matter more than absolute metric scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from errors import ProcessingError
from utils.tracing import Tracing

if TYPE_CHECKING:
    from classes.marker import CompositeDetection, MarkerDetection


@dataclass
class EpipolarGeometry:
    """Stores epipolar geometry between two camera views."""
    F: np.ndarray  # Fundamental matrix (3x3)
    inlier_mask: np.ndarray | None  # RANSAC inliers
    
    def __post_init__(self):
        if self.F.shape != (3, 3):
            raise ValueError(f"Fundamental matrix must be 3x3, got {self.F.shape}")


@dataclass
class ProjectivePoint3D:
    """3D point in projective space (4D homogeneous coordinates)."""
    coords: np.ndarray  # shape (4,) - [X, Y, Z, W] in homogeneous coords
    marker_id: int | None = None
    
    @property
    def xyz(self) -> np.ndarray:
        """Convert to Euclidean 3D coordinates."""
        if abs(self.coords[3]) < 1e-10:
            raise ValueError("Point at infinity")
        return self.coords[:3] / self.coords[3]
    
    @property
    def x(self) -> float:
        return float(self.xyz[0])
    
    @property
    def y(self) -> float:
        return float(self.xyz[1])
    
    @property
    def z(self) -> float:
        return float(self.xyz[2])


def compute_fundamental_matrix(
    points_cam1: np.ndarray,
    points_cam2: np.ndarray,
    ransac_threshold: float = 3.0,
    confidence: float = 0.99,
) -> EpipolarGeometry:
    """
    Compute fundamental matrix from point correspondences using RANSAC.
    
    Args:
        points_cam1: Points in camera 1, shape (N, 2)
        points_cam2: Corresponding points in camera 2, shape (N, 2)
        ransac_threshold: RANSAC reprojection threshold in pixels
        confidence: RANSAC confidence level
    
    Returns:
        EpipolarGeometry with fundamental matrix and inlier mask
    """
    with Tracing.ScopedZone("compute_fundamental_matrix"):
        if len(points_cam1) < 8:
            raise ProcessingError(
                f"Need at least 8 point correspondences for fundamental matrix, got {len(points_cam1)}",
                code="INSUFFICIENT_CORRESPONDENCES",
            )
        
        # Compute fundamental matrix using 8-point algorithm with RANSAC
        F, mask = cv2.findFundamentalMat(
            points_cam1,
            points_cam2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence,
        )
        
        if F is None:
            raise ProcessingError(
                "Fundamental matrix estimation failed: cv2.findFundamentalMat returned None",
                code="FUNDAMENTAL_MATRIX_NONE",
            )
        
        if F.shape[0] == 9:  # Multiple solutions
            F = F[:3, :]  # Take first solution
        
        # Enforce rank-2 constraint (fundamental matrix should have rank 2)
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0  # Set smallest singular value to 0
        F = U @ np.diag(S) @ Vt
        
        return EpipolarGeometry(F=F, inlier_mask=mask)


def canonical_camera_pair(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Canonical projective camera pair consistent with a fundamental matrix.

    P1 = [I | 0] and P2 = [[e']_x F | e'], where e' is the epipole in image 2
    (the right null vector of F^T). This is the standard construction from
    Hartley & Zisserman; the pair reproduces F exactly and yields a genuine
    projective reconstruction, i.e. one that differs from the metric truth by
    a single 3D homography shared by every point.

    Deriving R/t from F via recoverPose instead would require F to be an
    essential matrix, which holds only for calibrated (normalized) image
    coordinates. Feeding raw pixels with K = I asserts a 1-pixel focal length
    and bends the reconstruction out of shape, so coplanar points no longer
    reconstruct coplanar.

    Returns:
        (P1, P2), each 3x4.
    """
    # e' spans the left null space of F, i.e. the null space of F^T.
    _, _, Vt = np.linalg.svd(F.T)
    epipole = Vt[-1]
    norm = np.linalg.norm(epipole[:3])
    if norm < 1e-12:
        raise ProcessingError(
            "Degenerate fundamental matrix: epipole is not recoverable",
            code="DEGENERATE_EPIPOLAR_GEOMETRY",
        )
    epipole = epipole / norm

    epipole_cross = np.array([
        [0.0, -epipole[2], epipole[1]],
        [epipole[2], 0.0, -epipole[0]],
        [-epipole[1], epipole[0], 0.0],
    ], dtype=np.float64)

    P1 = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    P2 = np.hstack([epipole_cross @ F, epipole.reshape(3, 1)])
    return P1, P2


def camera_centres(P1: np.ndarray, P2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Camera centres of a projective pair, as 4D homogeneous points.

    The centre is the null space of the 3x4 projection matrix. For the
    canonical P1 = [I | 0] this is the origin; P2's centre is recovered by SVD.
    Callers need these to intersect a viewing ray with a reconstructed plane.
    """
    centre1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    _, _, Vt = np.linalg.svd(P2)
    centre2 = Vt[-1]
    return centre1, centre2


def triangulate_points_uncalibrated(
    points_cam1: np.ndarray,
    points_cam2: np.ndarray,
    F: np.ndarray,
    marker_ids: list[int] | None = None,
) -> tuple[list[ProjectivePoint3D], np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from two views without camera calibration.

    Uses a canonical projective camera pair derived from F, so the result is a
    projective reconstruction: correct up to an unknown 3D homography, with
    incidence and coplanarity preserved. Metric quantities (distances, angles)
    are NOT recoverable from it without further constraints.

    Args:
        points_cam1: Points in camera 1, shape (N, 2)
        points_cam2: Corresponding points in camera 2, shape (N, 2)
        F: Fundamental matrix (3x3)
        marker_ids: Optional marker IDs for each point

    Returns:
        (points, P1, P2) - the projective points plus the camera pair they
        live in, since the reconstruction is only meaningful alongside it.
    """
    with Tracing.ScopedZone("triangulate_points_uncalibrated"):
        if len(points_cam1) != len(points_cam2):
            raise ValueError(f"Point counts must match: {len(points_cam1)} != {len(points_cam2)}")

        P1, P2 = canonical_camera_pair(F)

        points_4d = cv2.triangulatePoints(
            P1, P2,
            np.ascontiguousarray(points_cam1.T, dtype=np.float64),
            np.ascontiguousarray(points_cam2.T, dtype=np.float64),
        )  # Returns 4xN homogeneous coordinates

        result = []
        for i in range(points_4d.shape[1]):
            marker_id = marker_ids[i] if marker_ids else None
            point = ProjectivePoint3D(
                coords=points_4d[:, i],
                marker_id=marker_id,
            )
            result.append(point)

        return result, P1, P2


@dataclass
class ReconstructedPlane:
    """Best-fit plane through reconstructed points, with an in-plane 2D basis."""
    centroid: np.ndarray  # (3,)
    basis_u: np.ndarray   # (3,) unit, in-plane
    basis_v: np.ndarray   # (3,) unit, in-plane, orthogonal to basis_u
    normal: np.ndarray    # (3,) unit
    flatness: float       # smallest/largest singular value; ~0 for truly coplanar input

    def to_plane_coords(self, points_xyz: np.ndarray) -> np.ndarray:
        """Project 3D points onto the plane's 2D basis. Input (N,3) -> output (N,2)."""
        centred = np.asarray(points_xyz, dtype=np.float64) - self.centroid
        return np.column_stack([centred @ self.basis_u, centred @ self.basis_v])


def fit_plane(points_xyz: np.ndarray) -> ReconstructedPlane:
    """
    Fit a plane to reconstructed points via SVD.

    The floor tags are coplanar in the world, and a projective reconstruction
    preserves incidence, so they stay coplanar here however distorted the frame
    is. That makes this fit the one well-conditioned structure available for
    registering the cloud, unlike a full 3D->2D DLT, which is degenerate on
    coplanar references.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ProcessingError(
            f"Plane fit needs at least 3 points, got {pts.shape[0]}",
            code="INSUFFICIENT_PLANE_SUPPORT",
        )
    centroid = pts.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(pts - centroid)
    largest = float(singular_values[0])
    flatness = float(singular_values[2] / largest) if largest > 1e-12 else 1.0
    return ReconstructedPlane(
        centroid=centroid,
        basis_u=Vt[0],
        basis_v=Vt[1],
        normal=Vt[2],
        flatness=flatness,
    )


def intersect_ray_with_plane(
    origin_xyz: np.ndarray,
    through_xyz: np.ndarray,
    plane: ReconstructedPlane,
) -> np.ndarray | None:
    """
    Intersect the ray origin->through with a plane; None if near-parallel.

    This is the plane+parallax step: the robot tag sits above the floor, so its
    reconstructed point is not where the robot touches the ground. Casting back
    through the camera centre recovers the floor footprint, which is what the
    map frame wants. Incidence is projectively invariant, so doing this in the
    reconstruction's affine chart is valid.
    """
    origin = np.asarray(origin_xyz, dtype=np.float64)
    direction = np.asarray(through_xyz, dtype=np.float64) - origin
    denominator = float(direction @ plane.normal)
    if abs(denominator) < 1e-12:
        return None
    t = float((plane.centroid - origin) @ plane.normal) / denominator
    return origin + t * direction


def compute_epipolar_line(point: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute epipolar line in second image for a point in first image.
    
    Args:
        point: Point in first image (2D or 3D homogeneous)
        F: Fundamental matrix
    
    Returns:
        Epipolar line in second image as [a, b, c] where ax + by + c = 0
    """
    if len(point) == 2:
        point = np.array([point[0], point[1], 1.0])
    
    return F @ point


def point_to_epipolar_distance(
    point1: np.ndarray,
    point2: np.ndarray,
    F: np.ndarray,
) -> float:
    """
    Compute distance from point2 to its epipolar line (determined by point1 and F).
    
    This measures how well the epipolar constraint is satisfied.
    
    Args:
        point1: Point in first image (x, y)
        point2: Corresponding point in second image (x, y)
        F: Fundamental matrix
    
    Returns:
        Distance in pixels
    """
    # Epipolar line in second image
    line = compute_epipolar_line(point1, F)
    
    # Point in homogeneous coordinates
    p2_homog = np.array([point2[0], point2[1], 1.0])
    
    # Algebraic distance
    dist_algebraic = abs(np.dot(line, p2_homog))
    
    # Normalize by line coefficients to get geometric distance
    dist_geometric = dist_algebraic / np.sqrt(line[0]**2 + line[1]**2)
    
    return float(dist_geometric)


def detection_to_points(
    detection: MarkerDetection | CompositeDetection,
) -> np.ndarray:
    """Extract corner points from detection as Nx2 array."""
    if hasattr(detection, "toPointsList"):
        points = detection.toPointsList()
    elif hasattr(detection, "toAggregatedPointsList"):
        points = detection.toAggregatedPointsList()
    else:
        raise TypeError(f"Unknown detection type: {type(detection)}")
    
    return np.array(points, dtype=np.float32).reshape(-1, 2)


def compute_fundamental_matrix_from_detections(
    detections_cam1: list[MarkerDetection | CompositeDetection],
    detections_cam2: list[MarkerDetection | CompositeDetection],
    ransac_threshold: float = 3.0,
) -> EpipolarGeometry:
    """
    Compute fundamental matrix from marker detections.
    
    Extracts all corner points from all markers and uses them as correspondences.
    
    Args:
        detections_cam1: Detections in camera 1
        detections_cam2: Corresponding detections in camera 2 (must have same marker IDs)
        ransac_threshold: RANSAC threshold in pixels
    
    Returns:
        EpipolarGeometry with fundamental matrix
    """
    with Tracing.ScopedZone("compute_fundamental_matrix_from_detections"):
        # Extract points from all detections
        points1_list = []
        points2_list = []
        
        for d1 in detections_cam1:
            # Find corresponding detection in cam2
            d2 = next((d for d in detections_cam2 if d.data == d1.data), None)
            if d2 is not None:
                pts1 = detection_to_points(d1)
                pts2 = detection_to_points(d2)
                points1_list.append(pts1)
                points2_list.append(pts2)
        
        if not points1_list:
            raise ProcessingError(
                "No common detections found between cameras",
                code="NO_COMMON_DETECTIONS",
            )
        
        points1 = np.vstack(points1_list)
        points2 = np.vstack(points2_list)
        
        return compute_fundamental_matrix(points1, points2, ransac_threshold)
