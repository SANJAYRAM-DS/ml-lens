"""
mllense - A production-grade, educational, extensible machine learning library.
"""
from mllense.math import linalg
from mllense.models import (
    LinearRegression,
    LogisticRegression,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    KMeans,
)

__version__ = "0.1.1"
__all__ = [
    "linalg",
    "LinearRegression",
    "LogisticRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "KMeans",
]
