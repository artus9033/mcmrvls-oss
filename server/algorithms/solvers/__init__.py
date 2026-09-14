"""
Solver abstraction for multi-view geometry.

Provides a common interface for different geometry solvers (homography, epipolar).
Each solver handles camera pair relationships and 3D/2D reconstruction differently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from classes.marker import CompositeDetection, MarkerDetection
    from classes.types import MarkerData


@dataclass
class SolverResult:
    """Result from processing a camera pair with a solver."""
    transform_matrix: np.ndarray | None  # Transformation matrix (H for homography, F for epipolar)
    inlier_mask: np.ndarray | None  # RANSAC inliers
    positions_2d: dict[MarkerData, tuple[float, float]]  # 2D positions for rendering
    metadata: dict[str, Any]  # Solver-specific metadata


class SolverState(ABC):
    """
    Base class for solver state management.
    
    Each solver maintains its own state (e.g., cached transforms, point clouds).
    """
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached state."""
        pass
    
    @abstractmethod
    def is_dirty(self) -> bool:
        """Check if state needs recalculation."""
        pass
    
    @abstractmethod
    def mark_clean(self) -> None:
        """Mark state as clean after recalculation."""
        pass


class GeometrySolver(ABC):
    """
    Abstract base class for multi-view geometry solvers.
    
    Solvers compute camera relationships and provide 2D/3D reconstruction
    from marker detections in multiple views.
    """
    
    @abstractmethod
    def create_state(self) -> SolverState:
        """Create a new solver state instance."""
        pass
    
    @abstractmethod
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
        Process a pair of cameras and compute their geometric relationship.
        
        Args:
            detections_cam1: Detections in first camera
            detections_cam2: Detections in second camera
            cam1_id: First camera identifier
            cam2_id: Second camera identifier
            state: Solver state for caching
            use_cache: Whether to use cached computations
        
        Returns:
            SolverResult with transformation and 2D positions
        """
        pass
    
    @abstractmethod
    def get_robot_measurements(
        self,
        detections: list[MarkerDetection | CompositeDetection],
        state: SolverState,
        config: Any,
    ) -> list[Any]:
        """
        Extract robot measurements from detections.
        
        Args:
            detections: Marker detections
            state: Solver state
            config: Algorithm configuration
        
        Returns:
            List of robot measurements (2D or 3D depending on solver)
        """
        pass
    
    @abstractmethod
    def render_to_topdown(
        self,
        state: SolverState,
        img_width: int,
        img_height: int,
        corner_markers: list[MarkerData],
    ) -> dict[MarkerData, tuple[float, float]]:
        """
        Render markers to 2D top-down view.
        
        Args:
            state: Solver state with geometry information
            img_width: Output image width
            img_height: Output image height
            corner_markers: List of corner marker IDs
        
        Returns:
            Dictionary mapping marker ID to (x, y) pixel coordinates
        """
        pass


def create_solver(solver_type: str, **kwargs) -> GeometrySolver:
    """
    Factory function to create a solver instance.
    
    Args:
        solver_type: Type of solver ("homography" or "epipolar")
        **kwargs: Additional solver-specific parameters
    
    Returns:
        Configured GeometrySolver instance
    """
    if solver_type == "homography":
        from algorithms.solvers.homography import HomographySolver
        return HomographySolver(**kwargs)
    elif solver_type == "epipolar":
        from algorithms.solvers.epipolar import EpipolarSolver
        return EpipolarSolver(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
