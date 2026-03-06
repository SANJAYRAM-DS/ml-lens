import numpy as np
from scipy.stats import chi2_contingency, chisquare, fisher_exact, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare

class TestResult:
    def __init__(self, statistic, pvalue, df=None, interpretation="", explanation="", definition=""):
        self.statistic = statistic
        self.pvalue = pvalue
        self.df = df
        self.interpretation = interpretation
        self.explanation = explanation
        self.definition = definition

def chi_square_goodness_of_fit(observed, expected=None, help=False, what=False):
    observed = np.array(observed)
    if expected is None:
        expected = np.full_like(observed, np.mean(observed))
    stat, p = chisquare(f_obs=observed, f_exp=expected)
    df = len(observed) - 1
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Chi-square goodness-of-fit:\nObserved = {observed}\nExpected = {expected}\n"
        explanation += f"Statistic = {stat}, df = {df}, p-value = {p}\n"

    definition = ""
    if what:
        definition += "Tests if observed frequencies match expected frequencies.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)


def chi_square_independence(table, help=False, what=False):
    table = np.array(table)
    stat, p, df, expected = chi2_contingency(table)
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Chi-square test of independence:\nObserved table:\n{table}\nExpected table:\n{expected}\n"
        explanation += f"Statistic = {stat}, df = {df}, p-value = {p}\n"

    definition = ""
    if what:
        definition += "Tests if two categorical variables are independent.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)

def proportion_test_1sample(successes, trials, p0, help=False, what=False):
    from statsmodels.stats.proportion import proportions_ztest
    stat, p = proportions_ztest(count=successes, nobs=trials, value=p0)
    df = trials - 1
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Proportion test 1-sample:\nSuccesses={successes}, Trials={trials}, p0={p0}\n"
        explanation += f"Z statistic={stat}, p-value={p}\n"

    definition = ""
    if what:
        definition += "Tests if sample proportion differs from hypothesized proportion.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)


def proportion_test_2sample(s1, n1, s2, n2, help=False, what=False):
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array([s1, s2])
    nobs = np.array([n1, n2])
    stat, p = proportions_ztest(count, nobs)
    df = n1 + n2 - 2
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Proportion test 2-sample:\nSuccess1={s1}, Trials1={n1}\nSuccess2={s2}, Trials2={n2}\n"
        explanation += f"Z statistic={stat}, p-value={p}\n"

    definition = ""
    if what:
        definition += "Tests if two sample proportions differ.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)


def fisher_exact_test(table_2x2, help=False, what=False):
    stat, p = fisher_exact(table_2x2)
    df = None
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Fisher's Exact Test for 2x2 table:\nTable:\n{table_2x2}\n"
        explanation += f"Odds ratio={stat}, p-value={p}\n"

    definition = ""
    if what:
        definition += "Tests independence in a 2x2 contingency table, exact test.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)

def mann_whitney_u(a, b, help=False, what=False):
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    df = len(a) + len(b) - 2
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Mann-Whitney U test:\nSample a={a}\nSample b={b}\n"
        explanation += f"U statistic={stat}, p-value={p}\n"

    definition = ""
    if what:
        definition += "Non-parametric test comparing two independent samples.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)


def wilcoxon_signed_rank(a, b, help=False, what=False):
    stat, p = wilcoxon(a, b)
    df = len(a) - 1
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Wilcoxon signed-rank test:\nSample a={a}\nSample b={b}\n"
        explanation += f"Statistic={stat}, p-value={p}\n"

    definition = ""
    if what:
        definition += "Non-parametric paired test comparing two related samples.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)


def kruskal_wallis(*groups, help=False, what=False):
    stat, p = kruskal(*groups)
    df = len(groups) - 1
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Kruskal-Wallis test:\nGroups={groups}\n"
        explanation += f"Statistic={stat}, p-value={p}\n"

    definition = ""
    if what:
        definition += "Non-parametric test for comparing medians of k independent groups.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)


def friedman_test(*groups, help=False, what=False):
    stat, p = friedmanchisquare(*groups)
    df = len(groups) - 1
    interpretation = "Reject H0" if p < 0.05 else "Fail to reject H0"

    explanation = ""
    if help:
        explanation += f"Friedman test:\nGroups={groups}\n"
        explanation += f"Statistic={stat}, p-value={p}\n"

    definition = ""
    if what:
        definition += "Non-parametric test for repeated measures across k conditions.\n"

    return TestResult(stat, p, df, interpretation, explanation, definition)
