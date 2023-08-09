"""
File defining an abstract classifier (for predicting discrete `y` values)
"""

import numpy as np

from cmnemoi_learn.abstract_model import AbstractModel


class AbstractClassifier(AbstractModel):
    """Class defining an abstract classifier (for predicting discrete `y` values).

    Args:
        AbstractModel (AbstractModel): Class defining an abstract model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.5

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns the accuracy of the model for the
        inputs and output given on arguments.

        Args:
            X (np.ndarray): Inputs
            y (np.ndarray): Output

        Returns:
            float: Accuracy
        """
        y_pred = self.predict(X)
        number_of_correctly_classified_samples = 0
        for i, y_true in enumerate(y):
            if y_true == y_pred[i]:
                number_of_correctly_classified_samples += 1

        return number_of_correctly_classified_samples / len(y_pred)
