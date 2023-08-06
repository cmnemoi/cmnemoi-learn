"""
Unit tests for Linear Regression model against sklearn implementation
"""
import numpy as np

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error

from cmnemoi_learn.regression.linear_regression import LinearRegression

np.random.seed(42)


def test_predict_friedman_dataset(regression_friedman_dataset: np.ndarray) -> None:
    """
    Test `predict` on the Friedman #2 problem.
    """
    X, y = regression_friedman_dataset
    cmnemoi_model = LinearRegression()
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLinearRegression()
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_predict_linear_dataset(regression_linear_dataset: np.ndarray) -> None:
    """
    Test `predict` on a linear dataset.
    """
    X, y = regression_linear_dataset
    cmnemoi_model = LinearRegression()
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLinearRegression()
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_predict_linear_dataset_with_small_n_big_p(
    regression_linear_dataset_with_small_n_big_p: np.ndarray,
) -> None:
    """Test `predict` on a linear dataset with small `n` and big `p`
    (underdetermined system)
    """
    X, y = regression_linear_dataset_with_small_n_big_p
    cmnemoi_model = LinearRegression()
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLinearRegression()
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_score(regression_friedman_dataset) -> None:
    """Test `score` method against sklearn implementation.

    Args:
        regression_friedman_dataset (np.ndarray): Dataset with a friedman pattern.
    """
    X, y = regression_friedman_dataset
    model = LinearRegression()
    model = model.fit(X, y)

    y_pred = model.predict(X)

    cmnemoi_mse = model.score(X, y)
    sklearn_mse = mean_squared_error(y_pred, y)

    assert np.isclose(cmnemoi_mse, sklearn_mse)
