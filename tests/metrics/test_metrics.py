"""
Unit tests for metrics and distances model against well-known implementations
"""
import numpy as np
import pytest
from scipy.spatial import distance

from cmnemoi_learn.metrics import euclidian_distance, manhattan_distance


def test_manhattan_distance() -> None:
    """Test for manhattan distance"""
    v_1 = np.random.randn(1, 10).ravel()
    v_2 = np.random.randn(1, 10).ravel()

    assert manhattan_distance(v_1, v_2) == distance.cityblock(v_1, v_2)


def test_manhattan_distance_raises_exception_if_arrays_not_same_size() -> None:
    """Test for manhattan distance"""
    v_1 = np.random.randn(1, 9).ravel()
    v_2 = np.random.randn(1, 10).ravel()

    with pytest.raises(ValueError, match="v_1 and v_2 should have the same shape"):
        manhattan_distance(v_1, v_2)


def test_euclidian_distance() -> None:
    """Test for euclidian distance"""
    v_1 = np.random.randn(1, 10).ravel()
    v_2 = np.random.randn(1, 10).ravel()

    assert np.isclose(euclidian_distance(v_1, v_2), distance.euclidean(v_1, v_2))


def test_euclidian_distance_raises_exception_if_arrays_not_same_size() -> None:
    """Test for euclidian distance"""
    v_1 = np.random.randn(1, 9).ravel()
    v_2 = np.random.randn(1, 10).ravel()

    with pytest.raises(ValueError, match="v_1 and v_2 should have the same shape"):
        euclidian_distance(v_1, v_2)
