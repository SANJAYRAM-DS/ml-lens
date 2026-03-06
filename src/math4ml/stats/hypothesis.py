import numpy as np
from scipy.stats import t as sp_t
from scipy.stats import norm, shapiro, kstest, anderson, levene, bartlett, fligner, f

class TestResult:
    def __init__(self, statistic, pvalue, df=None, interpretation="", explanation="", definition=""):
        self.statistic = statistic
        self.pvalue = pvalue
        self.df = df
        self.interpretation = interpretation
        self.explanation = explanation
        self.definition = definition

def t_test_1sample(data, mu0, help=False, what=False):
    data = np.array(data)
    n = len(data)
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    se = std_data / np.sqrt(n)
    t_stat = (mean_data - mu0) / se
    df = n - 1
    p_value = 2 * (1 - sp_t.cdf(abs(t_stat), df))
    interpretation = "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis"

    explanation = ""
    if help:
        explanation = (
            f"Sample mean = {mean_data}\n"
            f"Hypothesized mean = {mu0}\n"
            f"Std = {std_data}, n = {n}\n"
            f"SE = {se}\n"
            f"t = {t_stat}, df = {df}\n"
            f"p = {p_value}\n"
        )

    definition = ""
    if what:
        definition = (
            "One-sample t-test checks if sample mean differs from hypothesized mean.\n"
        )

    return TestResult(t_stat, p_value, df, interpretation, explanation, definition)

def t_test_2sample_independent(a, b, help=False, what=False):
    a, b = np.array(a), np.array(b)
    n1, n2 = len(a), len(b)
    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2) / se
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    p_value = 2 * (1 - sp_t.cdf(abs(t_stat), df))
    interpretation = "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis"

    explanation = ""
    if help:
        explanation = (
            f"Mean1={mean1}, Mean2={mean2}\n"
            f"Var1={var1}, Var2={var2}\n"
            f"SE={se}\n"
            f"t={t_stat}, df={df}\n"
            f"p={p_value}\n"
        )

    definition = ""
    if what:
        definition = (
            "Independent two-sample t-test checks if two group means differ.\n"
        )

    return TestResult(t_stat, p_value, df, interpretation, explanation, definition)

def t_test_paired(a, b, help=False, what=False):
    a, b = np.array(a), np.array(b)
    d = a - b
    return t_test_1sample(d, 0, help, what)

def z_test(data, mu0, sigma, help=False, what=False):
    data = np.array(data)
    n = len(data)
    mean_data = np.mean(data)
    se = sigma / np.sqrt(n)
    z_stat = (mean_data - mu0) / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    interpretation = "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis"

    explanation = ""
    if help:
        explanation = (
            f"Mean={mean_data}, Hypothesis={mu0}\n"
            f"Sigma={sigma}, n={n}\n"
            f"SE={se}\n"
            f"Z={z_stat}, p={p_value}\n"
        )

    definition = ""
    if what:
        definition = (
            "Z-test checks if sample mean differs from hypothesized mean when sigma is known.\n"
        )

    return TestResult(z_stat, p_value, n-1, interpretation, explanation, definition)

def anova_oneway(groups, help=False, what=False):
    groups = [np.array(g) for g in groups]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    overall_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - overall_mean)**2 for g in groups)
    df_between = k - 1
    ss_within = sum(sum((g - np.mean(g))**2) for g in groups)
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    p_value = 1 - f.cdf(f_stat, df_between, df_within)
    interpretation = "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis"

    explanation = ""
    if help:
        explanation = (
            f"SS_between={ss_between}, df_between={df_between}\n"
            f"SS_within={ss_within}, df_within={df_within}\n"
            f"F={f_stat}, p={p_value}\n"
        )

    definition = ""
    if what:
        definition = (
            "One-way ANOVA tests whether all group means are equal.\n"
        )

    return TestResult(f_stat, p_value, (df_between, df_within), interpretation, explanation, definition)

def anova_twoway(values, factor_a, factor_b, help=False, what=False):
    values = np.array(values)
    factor_a = np.array(factor_a)
    factor_b = np.array(factor_b)

    if not (len(values) == len(factor_a) == len(factor_b)):
        raise ValueError("All inputs must have equal length.")

    overall_mean = np.mean(values)
    levels_a = np.unique(factor_a)
    levels_b = np.unique(factor_b)

    ss_a = sum(np.sum(values[factor_a == lvl] - np.mean(values[factor_a == lvl]))**2 for lvl in levels_a)
    df_a = len(levels_a) - 1

    ss_b = sum(np.sum(values[factor_b == lvl] - np.mean(values[factor_b == lvl]))**2 for lvl in levels_b)
    df_b = len(levels_b) - 1

    ss_within = np.sum((values - overall_mean)**2)
    df_within = len(values) - df_a - df_b - 1

    ms_within = ss_within / df_within
    ms_a = ss_a / df_a
    ms_b = ss_b / df_b

    f_a = ms_a / ms_within
    f_b = ms_b / ms_within

    p_a = 1 - f.cdf(f_a, df_a, df_within)
    p_b = 1 - f.cdf(f_b, df_b, df_within)

    interpretation_a = "Reject H0 for factor A" if p_a < 0.05 else "Fail to reject H0 for factor A"
    interpretation_b = "Reject H0 for factor B" if p_b < 0.05 else "Fail to reject H0 for factor B"

    explanation = ""
    if help:
        explanation = (
            f"F_A={f_a}, p_A={p_a}\n"
            f"F_B={f_b}, p_B={p_b}\n"
        )

    definition = ""
    if what:
        definition = "Two-way ANOVA tests effects of two categorical factors.\n"

    return {
        "factor_A": TestResult(f_a, p_a, df_a, interpretation_a, explanation, definition),
        "factor_B": TestResult(f_b, p_b, df_b, interpretation_b, explanation, definition)
    }

def shapiro_wilk(data, help=False, what=False):
    stat, p = shapiro(data)
    interpretation = "Reject null: not normal" if p < 0.05 else "Fail to reject: normal"
    explanation = "Shapiro-Wilk test\n" if help else ""
    definition = "Tests normality\n" if what else ""
    return TestResult(stat, p, len(data)-1, interpretation, explanation, definition)

def kolmogorov_smirnov(data, help=False, what=False):
    stat, p = kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    interpretation = "Reject null: not normal" if p < 0.05 else "Fail to reject: normal"
    explanation = "K-S test\n" if help else ""
    definition = "Tests distribution vs normal\n" if what else ""
    return TestResult(stat, p, len(data)-1, interpretation, explanation, definition)

def anderson_darling(data, help=False, what=False):
    result = anderson(data)
    stat = result.statistic
    interpretation = [f"{lvl}%: {'Reject H0' if stat > cv else 'Fail to reject'}"
                        for lvl, cv in zip(result.significance_level, result.critical_values)]
    explanation = "Anderson-Darling test\n" if help else ""
    definition = "Normality test\n" if what else ""
    return TestResult(stat, None, None, interpretation, explanation, definition)

def levene_test(*groups, help=False, what=False):
    stat, p = levene(*groups)
    interpretation = "Reject H0: unequal variances" if p < 0.05 else "Fail to reject: equal variances"
    explanation = "Levene test\n" if help else ""
    definition = "Variance homogeneity test\n" if what else ""
    return TestResult(stat, p, len(groups)-1, interpretation, explanation, definition)

def bartlett_test(*groups, help=False, what=False):
    stat, p = bartlett(*groups)
    interpretation = "Reject H0: unequal variances" if p < 0.05 else "Fail to reject: equal variances"
    explanation = "Bartlett test\n" if help else ""
    definition = "Variance test\n" if what else ""
    return TestResult(stat, p, len(groups)-1, interpretation, explanation, definition)

def fligner_test(*groups, help=False, what=False):
    stat, p = fligner(*groups)
    interpretation = "Reject H0: unequal variances" if p < 0.05 else "Fail to reject: equal variances"
    explanation = "Fligner test\n" if help else ""
    definition = "Robust variance test\n" if what else ""
    return TestResult(stat, p, len(groups)-1, interpretation, explanation, definition)