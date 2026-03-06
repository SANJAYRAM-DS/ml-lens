"""
math4ml.preprocessing.feature_engineering
-----------------------------------------

Feature engineering utilities.

Common exports (examples, adapt as needed from feature_engineering.py):
- polynomial_features
- interaction_terms
- date_time_features
- target_stats (groupby aggregations)
- text_to_features

Each function optionally supports help/what flags:
    result = func(X, help=False, what=False)
    result, help_text, what_text = func(X, help=True, what=True)

EXAMPLE
-------
>>> from math4ml.preprocessing.feature_engineering import polynomial_features
>>> out, h, w = polynomial_features(X, degree=2, help=True, what=True)
"""

from . import feature_engineering as _fe

try:
    polynomial_features = _fe.polynomial_features
except AttributeError:
    pass

try:
    interaction_terms = _fe.interaction_terms
except AttributeError:
    pass

__all__ = ["feature_engineering", "polynomial_features", "interaction_terms"]
