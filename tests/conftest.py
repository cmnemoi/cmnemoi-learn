"""
Fixtures for unit tests
"""

import numpy as np
from sklearn.datasets import (
    make_classification,
    make_moons,
    make_regression,
    make_friedman2,
)
import pytest

BIAS = 5
NUMBER_OF_FEATURES = 2
NUMBER_OF_SAMPLES = 50
RANDOM_STATE = 42


@pytest.fixture
def classification_moons_dataset() -> np.ndarray:
    """Classification dataset with circles pattern (non linear)
    `X, y = classification_moons_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_moons(
        n_samples=NUMBER_OF_SAMPLES, shuffle=False, random_state=RANDOM_STATE
    )


@pytest.fixture
def classification_linear_dataset() -> np.ndarray:
    """Classification dataset linearly separable
    `X, y = classification_linear_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_classification(
        n_samples=NUMBER_OF_SAMPLES,
        n_features=NUMBER_OF_FEATURES,
        n_informative=NUMBER_OF_FEATURES,
        n_redundant=0,
        shuffle=False,
        random_state=RANDOM_STATE,
    )


@pytest.fixture
def classification_linear_dataset_with_small_n_big_p() -> np.ndarray:
    """Classification dataset linearly separable with small n and big p
    (under determined problem).
    `X, y = classification_linear_dataset_with_small_n_big_p` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_classification(
        n_samples=NUMBER_OF_SAMPLES,
        n_features=NUMBER_OF_SAMPLES + 1,
        n_informative=NUMBER_OF_SAMPLES + 1,
        n_redundant=0,
        shuffle=False,
        random_state=RANDOM_STATE,
    )


@pytest.fixture
def regression_friedman_dataset() -> np.ndarray:
    """Regression dataset which follows friedman #2 problem pattern
    `X, y = regression_friedman_dataset` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_friedman2(random_state=RANDOM_STATE)


@pytest.fixture
def regression_linear_dataset() -> np.ndarray:
    """Regression dataset which follows a linear pattern
    `X, y = regression_linear_dataset` to use

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
def regression_linear_dataset_with_small_n_big_p() -> np.ndarray:
    """Regression dataset which follows a linear pattern
    `X, y = regression_linear_dataset_with_small_n_big_p` to use

    Returns:
        np.ndarray: The dataset
    """
    return make_regression(
        n_samples=NUMBER_OF_SAMPLES,
        n_features=NUMBER_OF_SAMPLES + 1,
        n_informative=NUMBER_OF_SAMPLES + 1,
        bias=BIAS,
        shuffle=False,
        random_state=RANDOM_STATE,
    )
