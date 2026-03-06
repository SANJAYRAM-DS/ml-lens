"""
math4ml.preprocessing.encoding
------------------------------

Categorical encoding utilities.

Common exports (examples, adapt as needed from encoding.py):
- LabelEncoder
- OneHotEncoder
- OrdinalEncoder
- TargetEncoder
- FrequencyEncoder
- HashingEncoder

Each encoder supports:
    fit(X, y=None, help=False, what=False)
    transform(X, help=False, what=False)
    fit_transform(X, y=None, help=False, what=False)

EXAMPLE
-------
>>> from math4ml.preprocessing.encoding import LabelEncoder
>>> enc = LabelEncoder()
>>> out, h, w = enc.fit_transform(["a", "b", "a"], help=True, what=True)
"""

from . import encoding as _encoding

# Re-export commonly used names if they exist in encoding.py
# (This keeps the public API clean while avoiding circular imports.)
try:
    LabelEncoder = _encoding.LabelEncoder
except AttributeError:
    pass

try:
    OneHotEncoder = _encoding.OneHotEncoder
except AttributeError:
    pass

# Export the module and any detected classes
__all__ = ["encoding", "LabelEncoder", "OneHotEncoder"]
