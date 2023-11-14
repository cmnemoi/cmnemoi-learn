"""Module defining machine learning metrics and distances"""

import numpy as np


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan distance (L1 norm) between a and b.

    Args:
        a (np.ndarray): numpy array
        b (np.ndarray): numpy array

    Returns:
        float: Manhattan distance
    """
    if a.shape != b.shape:
        raise ValueError("a and b should have the same shape")
    return np.sum(np.abs(a - b))
