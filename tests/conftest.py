"""
Fixtures for unit tests
"""

import numpy as np
from sklearn.datasets import make_regression, make_circles
import pytest

RANDOM_STATE = 42


@pytest.fixture
def regression_circle_dataset() -> np.ndarray:
    """Regression dataset which follows circles pattern
    `X, y = regression_circle_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_circles(n_samples=100, shuffle=False, random_state=RANDOM_STATE)


@pytest.fixture
def regression_linear_dataset() -> np.ndarray:
    """Regression dataset which follows a linear pattern
    `X, y = regression_circle_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_regression(
        n_samples=100,
        n_features=10,
        n_informative=10,
        shuffle=False,
        random_state=RANDOM_STATE,
    )


@pytest.fixture
def regression_linear_dataset_with_noise() -> np.ndarray:
    """Regression dataset which follows a linear pattern
    `X, y = regression_circle_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_regression(
        n_samples=100,
        n_features=10,
        n_informative=10,
        noise=2,
        shuffle=False,
        random_state=RANDOM_STATE,
    )
