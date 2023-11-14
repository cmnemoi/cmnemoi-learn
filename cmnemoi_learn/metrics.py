"""Module defining machine learning metrics and distances"""

import numpy as np


def euclidian_distance(v_1: np.ndarray, v_2: np.ndarray) -> float:
    """Compute Euclidian distance (L2 norm) between two vectors.

    Args:
        v_1: Vector as numpy array
        v_2 Vector as numpy array

    Returns:
        float: Euclidian distance
    """
    if v_1.shape != v_2.shape:
        raise ValueError("v_1 and v_2 should have the same shape")
    return np.sqrt(np.sum((v_1 - v_2) ** 2))


def manhattan_distance(v_1: np.ndarray, v_2: np.ndarray) -> float:
    """Compute Manhattan distance (L1 norm) between two vectors.

    Args:
        v_1: Vector as numpy array
        v_2 Vector as numpy array

    Returns:
        float: Manhattan distance
    """
    if v_1.shape != v_2.shape:
        raise ValueError("v_1 and v_2 should have the same shape")
    return np.sum(np.abs(v_1 - v_2))
