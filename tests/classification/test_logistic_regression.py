"""
Unit tests for Logistic Regression model against sklearn implementation
"""
import numpy as np

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score

from cmnemoi_learn.classification import LogisticRegression

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def test_predict_moons_dataset(classification_moons_dataset: np.ndarray) -> None:
    """
    Test `predict` on a circle pattern dataset.
    """
    X, y = classification_moons_dataset
    cmnemoi_model = LogisticRegression(random_state=RANDOM_STATE)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLogisticRegression(penalty=None, random_state=RANDOM_STATE)
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_predict_linear_dataset(classification_linear_dataset: np.ndarray) -> None:
    """
    Test `predict` on a linearly separable dataset.
    """
    X, y = classification_linear_dataset
    cmnemoi_model = LogisticRegression(random_state=RANDOM_STATE)
    cmnemoi_model = cmnemoi_model.fit(X, y)

    sklearn_model = SklearnLogisticRegression(penalty=None, random_state=RANDOM_STATE)
    sklearn_model = sklearn_model.fit(X, y)

    cmnemoi_prediction = cmnemoi_model.predict(X)
    sklearn_prediction = sklearn_model.predict(X)

    assert np.allclose(cmnemoi_prediction, sklearn_prediction)


def test_score(classification_moons_dataset) -> None:
    """Test `score` method against sklearn implementation.

    Args:
        classification_moons_dataset (np.ndarray): Dataset with a moons pattern.
    """
    X, y = classification_moons_dataset
    model = LogisticRegression(random_state=RANDOM_STATE)
    model = model.fit(X, y)

    y_pred = model.predict(X)

    cmnemoi_accuracy = model.score(X, y)
    sklearn_accuracy = accuracy_score(y_pred, y)

    assert np.isclose(cmnemoi_accuracy, sklearn_accuracy)
