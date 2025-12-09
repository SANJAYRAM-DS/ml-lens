import numpy as np
from scipy.stats import t, norm

def ci_mean(data, confidence=0.95, help=False, what=False):
    data = np.array(data)
    n = len(data)
    mean_val = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = std_err * t.ppf((1 + confidence)/2, df=n-1)
    lower = mean_val - h
    upper = mean_val + h

    explanation = ""
    if help:
        explanation += f"CI for mean with confidence={confidence}\n"
        explanation += f"Mean={mean_val}, Std Error={std_err}, Margin={h}\n"
        explanation += f"CI = ({lower}, {upper})\n"

    definition = ""
    if what:
        definition += "Confidence interval estimates the range in which the population mean likely falls.\n"

    return (lower, upper), explanation, definition


def ci_proportion(successes, trials, confidence=0.95, help=False, what=False):
    p_hat = successes / trials
    z = norm.ppf((1 + confidence)/2)
    margin = z * np.sqrt(p_hat*(1-p_hat)/trials)
    lower = p_hat - margin
    upper = p_hat + margin

    explanation = ""
    if help:
        explanation += f"CI for proportion with confidence={confidence}\n"
        explanation += f"p_hat={p_hat}, z={z}, Margin={margin}\n"
        explanation += f"CI = ({lower}, {upper})\n"

    definition = ""
    if what:
        definition += "Confidence interval estimates the likely range for a population proportion.\n"

    return (lower, upper), explanation, definition
def cohen_d(x, y, help=False, what=False):
    """
    Cohen's d effect size for two samples
    """
    x = np.array(x)
    y = np.array(y)
    nx = len(x)
    ny = len(y)
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    d = (np.mean(x) - np.mean(y)) / pooled_std

    explanation = ""
    if help:
        explanation += f"Cohen's d = (mean_x - mean_y) / pooled_std\n"
        explanation += f"mean_x={np.mean(x)}, mean_y={np.mean(y)}, pooled_std={pooled_std}\n"
        explanation += f"Cohen's d = {d}\n"

    definition = ""
    if what:
        definition += "Cohen's d measures standardized difference between two means.\n"

    return d, explanation, definition

def odds_ratio(a_success, a_total, b_success, b_total, help=False, what=False):
    odds_a = a_success / (a_total - a_success)
    odds_b = b_success / (b_total - b_success)
    or_val = odds_a / odds_b

    explanation = ""
    if help:
        explanation += f"Odds A = {odds_a}, Odds B = {odds_b}\n"
        explanation += f"Odds ratio = Odds A / Odds B = {or_val}\n"

    definition = ""
    if what:
        definition += "Odds ratio measures association between binary outcomes in two groups.\n"

    return or_val, explanation, definition

def standard_error(data, help=False, what=False):
    data = np.array(data)
    se = np.std(data, ddof=1) / np.sqrt(len(data))

    explanation = ""
    if help:
        explanation += f"Standard error = std / sqrt(n) = {se}\n"

    definition = ""
    if what:
        definition += "Standard error estimates variability of the sample mean.\n"

    return se, explanation, definition

def margin_of_error(se, confidence=0.95, n=None, df=None, help=False, what=False):
    if df is not None:
        crit = t.ppf((1+confidence)/2, df=df)
    else:
        crit = norm.ppf((1+confidence)/2)
    me = crit * se

    explanation = ""
    if help:
        explanation += f"Margin of error = critical_value * SE = {me}\n"

    definition = ""
    if what:
        definition += "Margin of error is half-width of the confidence interval.\n"

    return me, explanation, definition