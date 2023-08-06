"""
Fixtures for unit tests
"""

import numpy as np
from sklearn.datasets import make_regression, make_circles
import pytest

BIAS = 5
NOISE = 2
NUMBER_OF_FEATURES = 2
NUMBER_OF_SAMPLES = 50
RANDOM_STATE = 42


@pytest.fixture
def regression_circle_dataset() -> np.ndarray:
    """Regression dataset which follows circles pattern
    `X, y = regression_circle_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_circles(
        n_samples=NUMBER_OF_SAMPLES, shuffle=False, random_state=RANDOM_STATE
    )


@pytest.fixture
def regression_linear_dataset() -> np.ndarray:
    """Regression dataset which follows a linear pattern
    `X, y = regression_circle_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_regression(
        n_samples=NUMBER_OF_SAMPLES,
        n_features=NUMBER_OF_FEATURES,
        n_informative=NUMBER_OF_FEATURES,
        bias=BIAS,
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
        n_samples=NUMBER_OF_SAMPLES,
        n_features=NUMBER_OF_FEATURES,
        n_informative=NUMBER_OF_FEATURES,
        bias=BIAS,
        noise=NOISE,
        shuffle=False,
        random_state=RANDOM_STATE,
    )
