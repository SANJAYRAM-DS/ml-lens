"""
math4ml.preprocessing.utils
---------------------------

Utility helpers for preprocessing.

Common exports (examples, adapt as needed from utils.py):
- infer_dtype
- handle_missing
- validate_array
- to_numpy
- make_pipeline

Functions may support help/what flags:
    out = handle_missing(X, strategy="mean", help=True, what=True)

EXAMPLE
-------
>>> from math4ml.preprocessing.utils import handle_missing
>>> X_clean, h, w = handle_missing(X, strategy="median", help=True, what=True)
"""

from . import utils as _utils

try:
    infer_dtype = _utils.infer_dtype
except AttributeError:
    pass

try:
    handle_missing = _utils.handle_missing
except AttributeError:
    pass

__all__ = ["utils", "infer_dtype", "handle_missing"]
