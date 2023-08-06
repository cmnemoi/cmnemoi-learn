"""
File defining a Linear Regression model.
"""

from typing import Self
import numpy as np
from numpy.linalg import pinv

from cmnemoi_learn.regression.abstract_regressor import AbstractRegressor


class LinearRegression(AbstractRegressor):
    """
    Linear Regression model.
    `y = X.theta` where `theta` are the parameters of the model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.theta = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the Linear Regression model with normal equations solution.

        The optimal parameters `theta` of the model are the ones which minimize
        Residuals Sum of Squares : `RSS = Sum(y - X.theta)**2`.

        Args:
            X (np.ndarray): Inputs
            y (np.ndarray): Output

        Returns:
            LinearRegression: Fitted Linear Regression model.
        """
        self.X = self._get_inputs_with_bias_column(X)
        self.y = y

        self.theta = pinv(self.X.T @ self.X) @ (self.X.T @ self.y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict new values with the Linear Regression model for the inputs given on argument.

        Args:
            X (np.ndarray): New inputs on which to predict.

        Returns:
            LinearRegression: Linear Regression model used to predict.
        """
        X = self._get_inputs_with_bias_column(X)
        return X @ self.theta

    def _get_inputs_with_bias_column(self, X: np.ndarray) -> np.ndarray:
        """Returns the inputs `X` with a `1`-filled bias column.

        Args:
            X (np.ndarray): Model inputs

        Returns:
            np.ndarray: New inputs with a bias column.
        """
        number_of_rows = X.shape[0]
        bias_column = np.ones((number_of_rows, 1))
        return np.hstack((bias_column, X))
