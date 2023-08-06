"""
File defining an abstract regressor (for predicting continous `y` values)
"""

import numpy as np

from cmnemoi_learn.abstract_model import AbstractModel


class AbstractRegressor(AbstractModel):
    """Class defining an abstract regressor (for predicting continous `y` values).

    Args:
        AbstractModel (AbstractModel): Class defining an abstract model.
    """

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
