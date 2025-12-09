"""
===============================================
Statistics Module (math4ml.stats)
===============================================

This module provides descriptive statistics, probability
utilities, hypothesis testing, and inferential tools used
in data analysis and machine learning.

each of it has help and what parameters:
help -> steps involved in the function
what -> a simple explanation of a function with example

just use help = True, what = True
mean, steps, defination = (mean(x), help=True, what=True)
print(steps) -> help
print(defination) -> what

SUBMODULES
----------

1. descriptive
    Basic descriptive statistics.
    - mean(x)
    - median(x)
    - mode(x)
    - data_range(x)
    - iqr(x)
    - mad(x)
    - skewness(x)
    - kurtisis(x)
    - covariance(x, y)
    - correlation(x, y, method)
    - quantiles(x, q)
    - variance(x)
    - std(x)
    - _range(x)
    - summary(x)

2. probability
    Probability helpers, combinatorics, PMF/PDF/CDF utilities.
    - fact(n)
    - nCr(n, r)
    - nPr(n, r)
    - pmf(dist, x)
    - pdf(dist, x)
    - cdf(dist, x)
    - bayes_theorem(prior, likelihood)
    - posterior(prior, likelihood)

3. distributions
    Common probability distributions (PDF, CDF, ppf, rvs, mean, variance, sampling).
    - Normal(μ, σ)
    - Uniform(a, b)
    - Exponential(...)
    - Bernoulli(p)
    - Binomial(n, p)
    - Poisson(λ)
    - Beta(...)
    -Gamma(...)

4. hypothesis
    Classical statistical hypothesis tests.
    - t_test_1sample(data, mu0)
    - t_test_2sample_independent(a, b)
    - t_test_paire(a, b)
    - z_test(a, b)
    - anova_oneway(groups)
    - anova_twoway(values, factor_a, factor_b)
    - shapiro_wilk(data)
    - kolmogorov_smirnov(data)
    - anderson_darling(data)
    - levene_test(*groups)
    - bartlett_test(*groups)
    - fligner_test(*group)

5. inferential
    Interval estimation and ANOVA.
    - confidence_interval_mean(...)
    - ci_mean(data, confidence)
    - ci_proportion(success, trails, confidence)
    - cohen_d(x, y)
    - odds_ratio(a_success, a_total, b_succes, b_total)
    - standard_error(data)
    - margin_of_errors(se, confidence)

6. tests
    Categorical & nonparametric tests.
    - chi_square_goodness_of_fit(...)
    - chi_square_independence(...)
    - proportion_test_1sample(...)
    - proportion_test_2sample(...)
    - friedman_test(...)
    - wilcoxon_signed_rank(...)
    - kruskal_wallis(...)
    - fisher_exact_test(...)

USAGE
-----

import math4ml.stats as st

# descriptive stats
mean_value = st.descriptive.mean([1,2,3])

# probability
ways = st.probability.nCr(5, 2)

# distributions
dist = st.distributions.Normal(0, 1)
p = dist.pdf(0)

# hypothesis testing
result = st.hypothesis.t_test([1,2,3], mu=0)

# inferential
ci = st.inferential.confidence_interval_mean([10,20,30])

# tests
chi = st.tests.chi_square_test([[10,20],[5,15]])

To explore a submodule:
help(math4ml.stats.descriptive)
help(math4ml.stats.probability)
help(math4ml.stats.distributions)
help(math4ml.stats.hypothesis)
help(math4ml.stats.inferential)
help(math4ml.stats.tests)

"""

# Expose submodules to the namespace
from . import descriptive
from . import probability
from . import distributions
from . import hypothesis
from . import inferential
from . import tests

__all__ = [
    "descriptive",
    "probability",
    "distributions",
    "hypothesis",
    "inferential",
    "tests",
]
