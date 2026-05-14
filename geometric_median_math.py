from __future__ import annotations
import numpy as np


def _to_float64_contiguous_2d(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input 'points' must have shape (n_points, 2).")
    if points.shape[0] == 0:
        raise ValueError("Input 'points' must contain at least two points.")
    return np.ascontiguousarray(points, dtype=np.float64)


def find_geometric_median(
    points: np.ndarray, tol: float = 1e-9, max_iter: int = 1000
) -> np.ndarray:

    p: np.ndarray = _to_float64_contiguous_2d(points)
    current: np.ndarray = np.ascontiguousarray(np.mean(p, axis=0), dtype=np.float64)

    for _ in range(max_iter):
        deltas: np.ndarray = p - current
        distances: np.ndarray = np.linalg.norm(deltas, axis=1)
        zero_mask: np.ndarray = distances < tol
        if np.any(zero_mask):
            point_index: int = int(np.argmax(zero_mask))
            return np.ascontiguousarray(p[point_index], dtype=np.float64)

        weights: np.ndarray = 1.0 / distances
        numerator: np.ndarray = np.sum(p * weights[:, None], axis=0)
        denominator: float = float(np.sum(weights))
        next_point: np.ndarray = np.ascontiguousarray(
            numerator / denominator, dtype=np.float64
        )

        if np.linalg.norm(next_point - current) < tol:
            return next_point

        current = next_point

    return current
