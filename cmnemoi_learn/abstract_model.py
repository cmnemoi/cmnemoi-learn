"""
File defining a class for a base machine learning model.
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class AbstractModel(ABC):
    """Abstract class to implement a base machine learning model."""

    def __init__(self) -> None:
        super().__init__()
        self.X = np.array([])
        self.y = np.array([])

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model.

        Args:
            X (np.ndarray): Inputs
            y (np.ndarray): Output

        Returns:
            Self: Fitted model.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values with the model for the inputs given on argument.

        Args:
            X (np.ndarray): Inputs

        Returns:
            np.ndarray: Predicted values.
        """

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns the score of the model for the
        inputs and output given on arguments.

        Args:
            X (np.ndarray): Inputs
            y (np.ndarray): Output

        Returns:
            float: Model score
        """

    def _reshape_ndarray(self, ndarray: np.ndarray) -> np.ndarray:
        """Reshape 1-D ndarray to ensure it has 2D shape.

        Args:
            ndarray (np.ndarray): ndarray to rehasepe

        Returns:
            np.ndarray: Reshaped ndarray
        """
        return (
            ndarray.reshape((ndarray.shape[0], 1))
            if len(ndarray.shape) == 1
            else ndarray
        )
