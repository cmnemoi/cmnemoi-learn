"""
Unit tests for Linear Regression model
"""
import numpy as np

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error

from cmnemoi_learn.regression.linear_regression import LinearRegression

np.random.seed(42)


def test_linear_predict(regression_linear_dataset) -> None:
    """
    Test `predict` against sklearn implementation.
    """
    X, y = regression_linear_dataset
    cmnemoi_model = LinearRegression()
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLinearRegression()
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_linear_with_noise_predict(regression_linear_dataset_with_noise) -> None:
    """
    Test `predict` against sklearn implementation.
    """
    X, y = regression_linear_dataset_with_noise
    cmnemoi_model = LinearRegression()
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLinearRegression()
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_circle_predict(regression_circle_dataset) -> None:
    """
    Test `predict` against sklearn implementation.
    """
    X, y = regression_circle_dataset
    cmnemoi_model = LinearRegression()
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLinearRegression()
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_score(regression_circle_dataset) -> None:
    """Test `score` method against sklearn implementation.

    Args:
        regression_circle_dataset (np.ndarray): Dataset with a circle pattern.
    """
    X, y = regression_circle_dataset
    model = LinearRegression()
    model = model.fit(X, y)

    y_pred = model.predict(X)

    cmnemoi_mse = model.score(X, y)
    sklearn_mse = mean_squared_error(y_pred, y)

    assert np.isclose(cmnemoi_mse, sklearn_mse)
