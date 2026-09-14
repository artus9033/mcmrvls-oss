"""
Geometric correctness of the uncalibrated epipolar reconstruction.

The scene these tests synthesise mirrors the arena: coplanar floor tags plus a
robot tag raised above them, seen by two overhead cameras with a realistic
focal length. Ground truth is known exactly, so the reconstruction can be held
to a real tolerance rather than to self-consistency.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from algorithms.epipolar import (
    canonical_camera_pair,
    camera_centres,
    fit_plane,
    intersect_ray_with_plane,
    triangulate_points_uncalibrated,
)

IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
FOCAL_PX = 2459.0  # matches what runtime calibration estimates on these cameras
ARENA_HALF_CM = 150.0  # the arena is 300x300 cm
ROBOT_TAG_HEIGHT_CM = 10.0  # tag sits on the robot's top plate, not the floor


def _intrinsics() -> np.ndarray:
    return np.array(
        [[FOCAL_PX, 0.0, IMAGE_WIDTH / 2.0], [0.0, FOCAL_PX, IMAGE_HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera(rot_x: float, rot_y: float, tx: float, height: float):
    rotation, _ = cv2.Rodrigues(np.array([rot_x, rot_y, 0.0], dtype=np.float64))
    translation = np.array([[tx], [0.0], [height]], dtype=np.float64)
    return rotation, translation


def _project(rotation, translation, points_xyz: np.ndarray) -> np.ndarray:
    camera_frame = (rotation @ points_xyz.T + translation).T
    image = (_intrinsics() @ camera_frame.T).T
    return image[:, :2] / image[:, 2:3]


def _scene(robot_xy=(40.0, -25.0)):
    """Floor tags on z=0 in a grid, plus one raised robot tag."""
    floor = np.array(
        [[x, y, 0.0] for x in np.linspace(-120, 120, 4) for y in np.linspace(-120, 120, 4)],
        dtype=np.float64,
    )
    robot = np.array([[robot_xy[0], robot_xy[1], ROBOT_TAG_HEIGHT_CM]], dtype=np.float64)
    return floor, robot


def _views(floor: np.ndarray, robot: np.ndarray):
    all_points = np.vstack([floor, robot])
    cam1 = _camera(0.0, 0.0, 0.0, 400.0)
    cam2 = _camera(0.15, 0.20, 60.0, 400.0)
    return _project(*cam1, all_points), _project(*cam2, all_points)


def _fundamental(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
    return F


def test_coplanar_tags_reconstruct_coplanar():
    """
    The defining property of a projective reconstruction: incidence survives.

    Floor tags are coplanar in the world, so they must come back coplanar
    however distorted the projective frame is. This is exactly what the old
    K=I/recoverPose path destroyed.
    """
    floor, robot = _scene()
    points1, points2 = _views(floor, robot)
    F = _fundamental(points1, points2)

    reconstructed, _, _ = triangulate_points_uncalibrated(points1, points2, F)
    floor_xyz = np.array([p.xyz for p in reconstructed[: len(floor)]])

    plane = fit_plane(floor_xyz)
    assert plane.flatness < 1e-6, f"floor tags reconstructed non-coplanar: {plane.flatness:.2e}"


def test_identity_intrinsics_reconstruction_is_not_planar():
    """
    Regression guard documenting the bug this module was fixed for.

    Treating F as an essential matrix under K = I asserts a one-pixel focal
    length. The reconstruction that follows is neither metric nor projective,
    and it visibly bends the floor. If this ever starts passing, someone has
    reintroduced a calibration assumption that does not hold.
    """
    floor, robot = _scene()
    points1, points2 = _views(floor, robot)
    F = _fundamental(points1, points2)

    _, rotation, translation, _ = cv2.recoverPose(F.copy(), points1, points2, np.eye(3))
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([rotation, translation])
    homogeneous = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    floor_xyz = (homogeneous[:3] / homogeneous[3]).T[: len(floor)]

    plane = fit_plane(floor_xyz)
    assert plane.flatness > 1e-4, (
        "the K=I path unexpectedly produced a flat floor; if triangulation was "
        "corrected, delete this regression guard rather than loosening it"
    )


def test_coplanar_references_cannot_fix_the_projective_gauge():
    """
    Why registration fits a plane instead of a full 3D->2D projection.

    With only coplanar references, the 11-DoF DLT is rank-deficient, so no
    amount of correctness in the triangulation rescues a direct 3D->map fit.
    """
    floor, robot = _scene()
    points1, points2 = _views(floor, robot)
    F = _fundamental(points1, points2)
    reconstructed, _, _ = triangulate_points_uncalibrated(points1, points2, F)
    floor_xyz = np.array([p.xyz for p in reconstructed[: len(floor)]])

    rows = []
    for xyz, target in zip(floor_xyz, floor[:, :2]):
        homogeneous = np.append(xyz, 1.0)
        rows.append(np.r_[homogeneous, np.zeros(4), -target[0] * homogeneous])
        rows.append(np.r_[np.zeros(4), homogeneous, -target[1] * homogeneous])
    singular_values = np.linalg.svd(np.array(rows))[1]

    conditioning = singular_values[-1] / singular_values[0]
    assert conditioning < 1e-10, (
        f"expected a degenerate DLT on coplanar references, got {conditioning:.2e}"
    )


@pytest.mark.parametrize("robot_xy", [(40.0, -25.0), (-90.0, 60.0), (0.0, 0.0), (110.0, 110.0)])
def test_robot_footprint_recovered_in_map_frame(robot_xy):
    """
    End-to-end: reconstruction -> plane fit -> map homography -> parallax.

    The recovered point is where the raised tag meets the floor along the
    viewing ray. A few centimetres of residual offset from the robot's true
    footprint is expected and irreducible: converting the reconstructed
    off-plane point into a vertical drop would need the vertical vanishing
    point, which coplanar references cannot supply (see the DLT test above).
    That ceiling is the substantive result -- on a planar arena, a correct
    uncalibrated reconstruction has no map-frame advantage left to spend.
    """
    floor, robot = _scene(robot_xy)
    points1, points2 = _views(floor, robot)
    F = _fundamental(points1, points2)

    reconstructed, P1, P2 = triangulate_points_uncalibrated(points1, points2, F)
    floor_xyz = np.array([p.xyz for p in reconstructed[: len(floor)]])
    robot_xyz = reconstructed[-1].xyz

    plane = fit_plane(floor_xyz)
    plane_coords = plane.to_plane_coords(floor_xyz)

    # Only the four map-corner tags are known in the map frame, matching what
    # the live solver is given.
    corner_indices = [0, 3, 12, 15]
    homography, _ = cv2.findHomography(
        plane_coords[corner_indices].astype(np.float64),
        floor[corner_indices, :2].astype(np.float64),
        cv2.RANSAC,
        8.0,
    )
    assert homography is not None

    centre1_h, _ = camera_centres(P1, P2)
    centre1 = centre1_h[:3] / centre1_h[3]
    footprint = intersect_ray_with_plane(centre1, robot_xyz, plane)
    assert footprint is not None

    mapped = cv2.perspectiveTransform(
        plane.to_plane_coords(footprint.reshape(1, 3)).reshape(1, 1, 2), homography
    )
    recovered = np.array([mapped[0, 0, 0], mapped[0, 0, 1]])

    error_cm = float(np.linalg.norm(recovered - np.array(robot_xy)))
    # Bounded by the residual parallax of a 10 cm tag under a 400 cm camera,
    # i.e. well under 2% of the 300 cm arena side. A reconstruction that is
    # actually broken misses by tens of centimetres, not this.
    assert error_cm < 5.0, f"robot footprint off by {error_cm:.2f} cm (recovered {recovered})"


def test_canonical_pair_reproduces_the_fundamental_matrix():
    """P2's construction is only valid if the pair induces the same F it came from."""
    floor, robot = _scene()
    points1, points2 = _views(floor, robot)
    F = _fundamental(points1, points2)

    P1, P2 = canonical_camera_pair(F)
    homogeneous1 = np.hstack([points1, np.ones((len(points1), 1))])
    homogeneous2 = np.hstack([points2, np.ones((len(points2), 1))])

    reconstructed, _, _ = triangulate_points_uncalibrated(points1, points2, F)
    for point, image1, image2 in zip(reconstructed, homogeneous1, homogeneous2):
        for P, expected in ((P1, image1), (P2, image2)):
            projected = P @ point.coords
            assert abs(projected[2]) > 1e-12
            reprojection = projected[:2] / projected[2]
            assert np.allclose(reprojection, expected[:2], atol=1e-3)
