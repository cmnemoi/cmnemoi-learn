"""
File defining a Linear Regression model.
"""

from typing import Self
import numpy as np
from numpy.linalg import inv


class LinearRegression:
    """
    Linear Regression model.
    `y = X.theta` where `theta` are the parameters of the model.
    """

    def __init__(self) -> None:
        self.X = np.array([])
        self.y = np.array([])
        self.theta = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the Linear Regression model with normal equations solution

        Args:
            X (np.ndarray): Inputs
            y (np.ndarray): Output

        Returns:
            LinearRegression: Fitted Linear Regression model.
        """
        self.theta = inv(X.T @ X) @ (X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict new values with the Linear Regression model for the inputs given on argument.

        Args:
            X (np.ndarray): New inputs on which to predict.

        Returns:
            LinearRegression: Linear Regression model used to predict.
        """
        return X.T @ self.theta
