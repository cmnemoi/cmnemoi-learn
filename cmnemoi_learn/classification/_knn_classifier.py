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
        nb_rows, max_column = X.shape
        distances_between_input_and_dataset = np.full((nb_rows, self.dataset.shape[0]), np.inf)
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(self.dataset[:, :max_column]):
                distances_between_input_and_dataset[i, j] = manhattan_distance(x_i, x_j)

        nearest_neighbor_indexes = np.full((nb_rows, self.k), 0)
        nearest_neighbor_labels = np.full((nb_rows, self.k), 0)
        for i, distance in enumerate(distances_between_input_and_dataset):
            nearest_neighbor_indexes[i] = np.argpartition(distance, kth=self.k, axis=-1)[: self.k]
            nearest_neighbor_labels[i] = self.dataset[nearest_neighbor_indexes[i], max_column]

        return np.array(
            [
                np.argmax(np.bincount(nearest_neighbor_labels[i]))
                for i in range(len(nearest_neighbor_labels))
            ]
        )
