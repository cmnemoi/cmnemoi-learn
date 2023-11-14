"""
File defining a K-Nearest Neighbors classifier model.
"""
from typing import Self
import numpy as np

from ._abstract_classifier import AbstractClassifier
from ..metrics import manhattan_distance


class KNNClassifier(AbstractClassifier):
    """K-Nearest Neighbors (KNN) classifier model.

    The KNN classifier memorizes all instances of
    the training set passed in `fit` method.

    Then, for each new instance passed in the `predict`
    method, it finds the K-nearest instances given
    a specific norm (L1, L2,...) and returns
    the majority label as prediction.

    (For the moment only the L2 norm, ie. Euclidian distance, is available)

    Args:
        k (int): The number of neighbors to evaluate.
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.dataset = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.dataset = np.concatenate([X, self._reshape_ndarray(y)], axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.full(X.shape, np.inf)
        max_column = X.shape[1]
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(self.dataset[:, :max_column]):
                distances[i, j] = manhattan_distance(x_i, x_j)

        predictions = []
        minimums = np.argmin(distances, axis=1)
        for minimum in minimums:
            predictions.append(self.dataset[minimum, max_column])

        return np.array(predictions)
