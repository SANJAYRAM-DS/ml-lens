"""
math4ml.preprocessing
=====================

Preprocessing utilities for cleaning, encoding, scaling and engineering features
before feeding data into machine learning models.

SUBMODULES
----------

1. encoding
   - functions/classes for categorical encoding, label encoding, one-hot encoding,
     ordinal encoding, target encoding, frequency encoding, hashing, etc.

2. feature_engineering
   - feature creation, interaction features, polynomial features, date/time features,
     aggregation helpers, text feature extraction utilities.

3. scaling
   - scaling and normalization helpers:
     - StandardScaler-like (mean, std)
     - MinMaxScaler-like
     - RobustScaler-like
     - MaxAbsScaler-like
     - Unit-norm scaling

4. utils
   - small helpers used across preprocessing, e.g. type inference, validation,
     missing value handlers, pipeline helpers.

CONVENTION
----------
Every function in these modules accepts two optional boolean parameters with defaults:
    help=False  -> return or include step-by-step explanation of how the function works
    what=False  -> return or include a short definition and example

RETURN CONTRACT
---------------
Where applicable, functions return:
    result
or
    result, help_text, what_text
depending on whether help/what are True.

USAGE
-----
>>> import math4ml.preprocessing as pp
>>> help(pp)
>>> help(pp.encoding)
>>> enc = pp.encoding.LabelEncoder()
>>> transformed, help_text, what_text = enc.fit_transform(X, help=True, what=True)

EXPORTED SUBMODULES
-------------------
encoding, feature_engineering, scaling, utils
"""

from . import encoding
from . import feature_engineering
from . import scaling
from . import utils

__all__ = [
    "encoding",
    "feature_engineering",
    "scaling",
    "utils",
]
