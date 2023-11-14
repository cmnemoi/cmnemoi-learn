"""
Unit tests for Logistic Regression model against sklearn implementation
"""
import numpy as np
import pytest

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as SklearnKNNClassifier
from sklearn.metrics import accuracy_score

from cmnemoi_learn.classification import KNNClassifier

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_predict_moons_dataset(classification_moons_dataset: np.ndarray, k: int) -> None:
    """
    Test `predict` on a circle pattern dataset.
    """
    X, y = classification_moons_dataset
    cmnemoi_model = KNNClassifier(k=k)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnKNNClassifier(n_neighbors=k, p=1)
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.array_equal(cmnemoi_prediction, sklearn_prediction)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_predict_linear_dataset(classification_linear_dataset: np.ndarray, k: int) -> None:
    """
    Test `predict` on a linearly separable dataset.
    """
    X, y = classification_linear_dataset
    cmnemoi_model = KNNClassifier(k=k)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnKNNClassifier(n_neighbors=k, p=1)  # p = 1 for Manhattan distance
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.array_equal(cmnemoi_prediction, sklearn_prediction)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_predict_iris_dataset(k: int) -> None:
    """
    Test `predict` on Iris dataset.
    """
    X, y = load_iris(return_X_y=True)
    cmnemoi_model = KNNClassifier(k=k)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnKNNClassifier(n_neighbors=k, p=1)
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.array_equal(cmnemoi_prediction, sklearn_prediction)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_predict_with_l2_norm(k: int) -> None:
    """
    Test `predict` on Iris dataset.
    """
    X, y = load_iris(return_X_y=True)
    cmnemoi_model = KNNClassifier(k=k)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnKNNClassifier(n_neighbors=k, p=2)
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.array_equal(cmnemoi_prediction, sklearn_prediction)


@pytest.mark.parametrize("k", [1, 3, 7])
def test_score(classification_linear_dataset: np.ndarray, k: int) -> None:
    """Test `score` method against sklearn implementation.

    Args:
        classification_moons_dataset (np.ndarray): Dataset with a moons pattern.
    """
    X, y = classification_linear_dataset
    model = KNNClassifier(k=k)
    model = model.fit(X, y)

    y_pred = model.predict(X)

    cmnemoi_accuracy = model.score(X, y)
    sklearn_accuracy = accuracy_score(y_pred, y)

    assert cmnemoi_accuracy == sklearn_accuracy
