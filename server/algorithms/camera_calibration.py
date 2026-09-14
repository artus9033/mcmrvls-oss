"""AprilTag-based camera intrinsic/extrinsic calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics + extrinsics for one camera in the map/world frame.

    ``R`` and ``t`` follow OpenCV solvePnP convention: ``X_cam = R @ X_world + t``
    with world points in centimeters on the map plane (z=0).
    """

    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (N,) or (N,1)
    rvec: np.ndarray  # 3x1
    tvec: np.ndarray  # 3x1
    image_size: tuple[int, int]  # (width, height)
    reprojection_error: float
    source_video: str | None = None
    camera_index: int | None = None

    @property
    def R(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def t(self) -> np.ndarray:
        return self.tvec.reshape(3, 1)

    def projection_matrix(self) -> np.ndarray:
        """3x4 projection matrix P = K [R|t]."""
        return self.camera_matrix @ np.hstack([self.R, self.t])

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(),
            "rvec": self.rvec.reshape(-1).tolist(),
            "tvec": self.tvec.reshape(-1).tolist(),
            "reprojection_error": float(self.reprojection_error),
            "source_video": self.source_video,
            "camera_index": self.camera_index,
            "world_units": "cm",
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CameraCalibration":
        return CameraCalibration(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            rvec=np.asarray(data["rvec"], dtype=np.float64).reshape(3, 1),
            tvec=np.asarray(data["tvec"], dtype=np.float64).reshape(3, 1),
            image_size=(int(data["image_width"]), int(data["image_height"])),
            reprojection_error=float(data.get("reprojection_error", float("nan"))),
            source_video=data.get("source_video"),
            camera_index=data.get("camera_index"),
        )



def corner_tag_object_quads_cm(config: dict[str, Any]) -> dict[int, np.ndarray]:
    """Map-corner tag quads in centimeters: (TL, TR, BL, BR), z=0."""
    dims = config["dimensions"]
    map_w = float(dims["mapWidth"])
    map_h = float(dims["mapHeight"])
    mw = float(dims["markerWidth"])
    mh = float(dims["markerHeight"])

    def quad(x0: float, y0: float) -> np.ndarray:
        return np.array(
            [
                [x0, y0, 0.0],
                [x0 + mw, y0, 0.0],
                [x0, y0 + mh, 0.0],
                [x0 + mw, y0 + mh, 0.0],
            ],
            dtype=np.float64,
        )

    ids = config["map"]
    return {
        int(ids["TL"]): quad(0.0, 0.0),
        int(ids["TR"]): quad(map_w - mw, 0.0),
        int(ids["BL"]): quad(0.0, map_h - mh),
        int(ids["BR"]): quad(map_w - mw, map_h - mh),
    }


def atlas_quads_norm_to_cm(
    atlas_norm: dict[int, np.ndarray],
    map_w_cm: float,
    map_h_cm: float,
) -> dict[int, np.ndarray]:
    """Convert normalized (TL,TR,BL,BR) quads to cm object points (z=0)."""
    out: dict[int, np.ndarray] = {}
    for tag_id, quad in atlas_norm.items():
        pts = np.zeros((4, 3), dtype=np.float64)
        pts[:, 0] = quad[:, 0] * map_w_cm
        pts[:, 1] = quad[:, 1] * map_h_cm
        out[int(tag_id)] = pts
    return out


def detection_image_quad(marker: Any) -> np.ndarray:
    return np.array(
        [marker.topLeft, marker.topRight, marker.bottomLeft, marker.bottomRight],
        dtype=np.float64,
    )


def estimate_camera_calibration(
    object_points: np.ndarray,
    image_points: np.ndarray,
    image_size: tuple[int, int],
    *,
    source_video: str | None = None,
    camera_index: int | None = None,
    fixed_camera_matrix: np.ndarray | None = None,
) -> CameraCalibration:
    """
    Estimate K, distortion, and extrinsics from a single planar board view.

    Fixed overhead cameras only see one geometric view of the floor, so we
    constrain intrinsics (fixed aspect, principal point near image center,
    no tangential distortion) and solve the well-posed extrinsics + focal length.

    When ``fixed_camera_matrix`` is provided (e.g. shared fx from a better-observed
    camera), only extrinsics are solved via PnP.
    """
    if len(object_points) < 4:
        raise ValueError(f"Need ≥4 point correspondences, got {len(object_points)}")

    w, h = image_size
    obj = object_points.astype(np.float32).reshape(-1, 1, 3)
    img = image_points.astype(np.float32).reshape(-1, 1, 2)

    if fixed_camera_matrix is not None:
        camera_matrix = np.asarray(fixed_camera_matrix, dtype=np.float64).copy()
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed with fixed intrinsics")
    else:
        camera_matrix = np.array(
            [[float(w), 0.0, w / 2.0], [0.0, float(w), h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
        )

        _rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [obj],
            [img],
            image_size,
            camera_matrix,
            dist_coeffs,
            flags=flags,
        )
        rvec = np.asarray(rvecs[0], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvecs[0], dtype=np.float64).reshape(3, 1)

        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed after calibrateCamera")

    fx = float(camera_matrix[0, 0])
    if not (0.25 * w <= fx <= 4.0 * w):
        raise RuntimeError(
            f"Implausible focal length fx={fx:.1f} for image width {w} "
            "(planar view underconstrained; need a shared-K fallback)"
        )

    projected, _ = cv2.projectPoints(
        object_points.astype(np.float32), rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    err = float(np.sqrt(np.mean(np.sum((projected - image_points.reshape(-1, 2)) ** 2, axis=1))))

    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        image_size=image_size,
        reprojection_error=err,
        source_video=source_video,
        camera_index=camera_index,
    )


def triangulate_points_calibrated(
    points_cam1: np.ndarray,
    points_cam2: np.ndarray,
    calib1: CameraCalibration,
    calib2: CameraCalibration,
) -> np.ndarray:
    """
    Triangulate metric 3D points (Nx3, cm) in the map/world frame.

    Undistorts image points, then uses each camera's calibrated projection matrix.
    """
    pts1 = np.asarray(points_cam1, dtype=np.float64).reshape(-1, 1, 2)
    pts2 = np.asarray(points_cam2, dtype=np.float64).reshape(-1, 1, 2)

    und1 = cv2.undistortPoints(pts1, calib1.camera_matrix, calib1.dist_coeffs, P=calib1.camera_matrix)
    und2 = cv2.undistortPoints(pts2, calib2.camera_matrix, calib2.dist_coeffs, P=calib2.camera_matrix)

    P1 = calib1.projection_matrix()
    P2 = calib2.projection_matrix()
    points_4d = cv2.triangulatePoints(P1, P2, und1.reshape(-1, 2).T, und2.reshape(-1, 2).T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d
