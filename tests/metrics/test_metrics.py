"""
Unit tests for metrics and distances model against well-known implementations
"""
import numpy as np
import pytest
from scipy.spatial import distance

from cmnemoi_learn.metrics import manhattan_distance


def test_manhattan_distance() -> None:
    """Test for manhattan distance"""
    a = np.random.randn(1, 10).ravel()
    b = np.random.randn(1, 10).ravel()

    assert manhattan_distance(a, b) == distance.cityblock(a, b)


def test_manhattan_distance_raises_exception_if_arrays_not_same_size() -> None:
    """Test for manhattan distance"""
    a = np.random.randn(1, 9).ravel()
    b = np.random.randn(1, 10).ravel()

    with pytest.raises(ValueError, match="a and b should have the same shape"):
        manhattan_distance(a, b)
