from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class ViewTransformer:
    source: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        self.source = np.asarray(self.source, dtype=np.float32)
        self.target = np.asarray(self.target, dtype=np.float32)

        if self.source.ndim != 2 or self.target.ndim != 2:
            raise ValueError("source and target must be 2D arrays of shape (N, 2)")
        if self.source.shape != self.target.shape:
            raise ValueError(f"source and target must have same shape, got {self.source.shape} vs {self.target.shape}")
        if self.source.shape[0] < 4:
            raise ValueError("At least 4 point correspondences are required")

        matrix, mask = cv2.findHomography(self.source, self.target, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if matrix is None:
            raise ValueError("Could not compute homography matrix")
        
        matrix_rank = np.linalg.matrix_rank(matrix)
        if matrix_rank < 3:
            raise ValueError(f"Degenerate homography matrix: rank={matrix_rank} < 3")

        self.matrix = matrix
        self.mask = mask

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        reshaped = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.matrix)
        return transformed.reshape(-1, 2)