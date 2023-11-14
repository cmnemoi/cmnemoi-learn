"""
Module implementing machine learning models for classification tasks.
"""

from ._knn_classifier import KNNClassifier
from ._logistic_regression import LogisticRegression

__all__ = ["KNNClassifier", "LogisticRegression"]
