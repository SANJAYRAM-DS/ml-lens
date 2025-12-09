"""
math4ml.preprocessing.scaling
-----------------------------

Scaling and normalization utilities.

Common exports (examples, adapt as needed from scaling.py):
- StandardScaler
- MinMaxScaler
- RobustScaler
- MaxAbsScaler
- Normalizer

Scalers follow a fit/transform API and support help/what flags:
    scaler = StandardScaler()
    scaler.fit(X, help=True)
    Xs, help_text, what_text = scaler.transform(X, help=True, what=True)

EXAMPLE
-------
>>> from math4ml.preprocessing.scaling import StandardScaler
>>> s = StandardScaler()
>>> s.fit(X)
>>> Xs, h, w = s.transform(X, help=True, what=True)
"""

from . import scaling as _scaling

try:
    StandardScaler = _scaling.StandardScaler
except AttributeError:
    pass

try:
    MinMaxScaler = _scaling.MinMaxScaler
except AttributeError:
    pass

__all__ = ["scaling", "StandardScaler", "MinMaxScaler"]
