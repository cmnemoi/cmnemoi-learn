"""
File defining a logistic regression model.
"""
from typing import Optional, Self
import numpy as np

from cmnemoi_learn.classification.abstract_classifier import AbstractClassifier


class LogisticRegression(AbstractClassifier):
    """Logistic Regression model.

    The logistic regression tries to compute the
    probability a sample x to belong to the
    `Y=1` positive class : `P(Y=1|X=x)`.

    For this, it assumes `P(Y=1|X=x) = z(X.weights)`
    where z is the logistic function `z(x) = (1 + exp(-x))**-1`
    """

    def __init__(self, n_iter: int = 500, random_state: Optional[int] = None) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed=random_state)
        self.weights = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the model.

        The `weights` of the model are found
        by minimizing the negative log-likelihood of the problem
        (aka. log loss or cross entropy) :
        `−∑ ylog(z(X.weights)) + (1-y)log(1-z(X.weights))` with
        gradient descent algorithm.

        Args:
            X (np.ndarray): Inputs
            y (np.ndarray): Output

        Returns:
            Self: Fitted model.
        """
        self.X = self._get_inputs_with_bias_column(X)
        self.y = self._reshape_ndarray(y)
        self.weights = self._gradient_descent(n_iter=self.n_iter)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class of the inputs given on argument.

        Args:
            X (np.ndarray): Inputs

        Returns:
            np.ndarray: Output
        """
        X = self._get_inputs_with_bias_column(X)
        probabilities = self._logistic_function(X @ self.weights)
        return np.array(
            [1 if probability > self.threshold else 0 for probability in probabilities]
        )

    def _gradient_descent(self, n_iter: int, learning_rate: float = 0.5) -> np.ndarray:
        """Gradient descent algorithm.
        Find the best `weights` for logistic regression by minimizing the log loss.

        Args:
            n_iter (int, optional): Number of iterations.
            learning_rate (float, optional): Learning rate. Defaults to 0.5.

        Returns:
            Tuple[np.ndarray, float]: Weights and bias of the model.
        """
        n = self.X.shape[0]
        weights = self.rng.random((self.X.shape[1], 1))
        weights_gradient = np.array([])
        for _ in range(n_iter):
            weights_gradient = (
                (1 / n)
                * self.X.T
                @ (self._logistic_function(self.X @ weights) - self.y)
            )
            weights = weights - learning_rate * weights_gradient

        return weights

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

    def _logistic_function(self, X: np.ndarray) -> np.ndarray:
        """Logistic function.
        `z(x) = (1 + exp(-x))**-1`

        Args:
            X (np.ndarray): Inputs

        Returns:
            np.ndarray: Outputs
        """
        return (1 + np.exp(-X)) ** -1
