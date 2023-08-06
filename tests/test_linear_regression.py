"""
Unit tests for Linear Regression model
"""
import numpy as np

from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from cmnemoi_learn.linear_regression import LinearRegression

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
