"""
math4ml — A lightweight math & ML toolkit.

===========================================================
MODULE MAP
===========================================================

check here for tutorials=> https://github.com/SANJAYRAM-DS/math4ml.tutorials.git

Here i have listed few of them check out every concepts and their availabel functions for correct function lists

Example: print(math4ml.linalg.__doc__)

1. Linear Algebra (math4ml.linalg)
----------------------------------
    create/
        - zeros(shape)
        - ones(shape)
        - eye(n)
        - rand(shape, seed=None)

    ops/
        - add(A, B)
        - subtract(A, B)
        - multiply(A, B)          (elementwise)
        - divide(A, B)
        - scalar_multiply(A, s)
        - transpose(A)
        - matmul(A, B)
        - dot(a, b)
        - outer(a, b)
        - reshape(A, new_shape)
        - flatten(A)
        - stack(A, B, axis=0)
        - concat(arrays, axis)

    decomposition/
        - lu(A)
        - qr(A)
        - svd(A)
        - eigen(A)
        - det(A)
        - rank(A)
        - inverse(A)



2. Statistics (math4ml.stats)
-----------------------------
    descriptive.py
        - mean(x)
        - median(x)
        - mode(x)
        - var(x)
        - std(x)
        - quantile(x, q)
        - iqr(x)

    inferential.py
        - t_test(x, y)
        - z_test(x, mu)
        - chi_square(o, e)
        - anova(groups)
        - correlation(x, y)
        - covariance(x, y)

    probability.py
        - factorial(n)
        - perm(n, r)
        - comb(n, r)
        - pmf(dist, x)
        - pdf(dist, x)
        - cdf(dist, x)
        - bayes(prior, likelihood, evidence)

    tests.py  (categorical + nonparametric)
        - mann_whitney(x, y)
        - wilcoxon(x, y)
        - kruskal(groups)
        - chi_square_test(...)
        - fisher_exact(a, b, c, d)


3. Preprocessing (math4ml.preprocessing)
----------------------------------------
    encoding.py
        - OneHotEncoder
        - LabelEncoder
        - OrdinalEncoder

    scaling/
        - StandardScaler
        - MinMaxScaler
        - MaxAbsScaler
        - RobustScaler

    feature_engineering.py
        - polynomial_features(X, degree)
        - interaction_terms(X)
        - binning(X, bins)
        - normalize_text(text)
        - extract_datetime_features(dt_series)

    utils/
        - train_test_split(X, y, test_size)
        - shuffle_data(X, y)
        - impute_missing(X, strategy="mean")


4. Optimization (math4ml.optim)
-------------------------------
    gradient.py
        - grad(f, x)
        - jacobian(f, x)
        - hessian(f, x)

    algorithms.py
        - gradient_descent(...)
        - stochastic_gradient_descent(...)
        - momentum_optimizer(...)
        - adam_optimizer(...)


5. Machine Learning (math4ml.ml)
--------------------------------
    linear_models/
        - LinearRegression
        - RidgeRegression
        - LassoRegression

    classification/
        - LogisticRegression
        - NaiveBayes
        - KNNClassifier

    metrics/
        - mse(y, y_pred)
        - rmse(y, y_pred)
        - mae(y, y_pred)
        - accuracy(y, y_pred)
        - f1_score(y, y_pred)
        - confusion_matrix(y, y_pred)


===========================================================
Usage
===========================================================

import math4ml

help(math4ml)        # shows full module map
help(math4ml.linalg) # help for a specific submodule
help(math4ml.stats)  # detailed stats documentation

===========================================================
"""

__version__ = "0.1.0"

__all__ = ["linalg", "stats", "preprocessing", "optimization", "ml_models"]
"""
math4ml — A lightweight Mathematics & ML Toolkit.
"""

__version__ = "0.1.0"

from . import linalg
from . import stats
from . import preprocessing
from . import optimization
from . import ml_models
