"""
Unit tests for Logistic Regression model against sklearn implementation
"""
import numpy as np

from sklearn.neighbors import KNeighborsClassifier as SklearnKNNClassifier
from sklearn.metrics import accuracy_score

from cmnemoi_learn.classification import KNNClassifier

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def test_predict_moons_dataset_with_k_equals_1(classification_moons_dataset: np.ndarray) -> None:
    """
    Test `predict` on a circle pattern dataset with k=1.
    """
    X, y = classification_moons_dataset
    cmnemoi_model = KNNClassifier(k=1)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnKNNClassifier(n_neighbors=1)
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.array_equal(cmnemoi_prediction, sklearn_prediction)


def test_predict_linear_dataset_with_k_equals_1(classification_linear_dataset: np.ndarray) -> None:
    """
    Test `predict` on a linearly separable dataset with k=1.
    """
    X, y = classification_linear_dataset
    cmnemoi_model = KNNClassifier(k=1)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnKNNClassifier(n_neighbors=1)
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.array_equal(cmnemoi_prediction, sklearn_prediction)


def test_score_with_k_equals_1(classification_linear_dataset: np.ndarray) -> None:
    """Test `score` method against sklearn implementation with k=1.

    Args:
        classification_moons_dataset (np.ndarray): Dataset with a moons pattern.
    """
    X, y = classification_linear_dataset
    model = KNNClassifier(k=1)
    model = model.fit(X, y)

    y_pred = model.predict(X)

    cmnemoi_accuracy = model.score(X, y)
    sklearn_accuracy = accuracy_score(y_pred, y)

    assert cmnemoi_accuracy == sklearn_accuracy
