import math
from collections import Counter

def _sorted(data):
    return sorted(data)

def mean(data, help=False, what=False):
    n = len(data)
    s = sum(data)
    result = s / n

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating mean.\n"
        explanation += f"1. Sum of data = {s}\n"
        explanation += f"2. Count of data = {n}\n"
        explanation += f"3. Mean = sum / count = {s} / {n} = {result}\n"

    definition = ""
    if what:
        definition += (
            "Mean (average) is the sum of all values divided by the number of values.\n"
            f"Example: mean({data}) = {result}\n"
        )

    return result, explanation, definition

def median(data, help=False, what=False):
    d = _sorted(data)
    n = len(d)

    if n % 2 == 1:
        result = d[n // 2]
    else:
        result = (d[n//2 - 1] + d[n//2]) / 2

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating median.\n"
        explanation += f"Sorted data = {d}\n"
        if n % 2 == 1:
            explanation += f"Odd count → middle value is {result}\n"
        else:
            explanation += (
                f"Even count → average of {d[n//2 - 1]} and {d[n//2]} = {result}\n"
            )

    definition = ""
    if what:
        definition += (
            "Median is the middle value when data is sorted.\n"
            f"Example: median({data}) = {result}\n"
        )

    return result, explanation, definition

def mode(data, help=False, what=False):
    c = Counter(data)
    max_count = max(c.values())
    result = [x for x, count in c.items() if count == max_count]

    # If only one mode, return number instead of list
    result = result[0] if len(result) == 1 else result

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating mode.\n"
        explanation += f"Frequency = {dict(c)}\n"
        explanation += f"Highest frequency = {max_count}\n"
        explanation += f"Mode(s) = {result}\n"

    definition = ""
    if what:
        definition += (
            "Mode is the most frequently occurring value(s).\n"
            f"Example: mode({data}) = {result}\n"
        )

    return result, explanation, definition


def data_range(data, help=False, what=False):
    result = max(data) - min(data)

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating range.\n"
        explanation += f"Max = {max(data)}, Min = {min(data)}\n"
        explanation += f"Range = max - min = {result}\n"

    definition = ""
    if what:
        definition += (
            "Range is the difference between the largest and smallest value.\n"
            f"Example: range({data}) = {result}\n"
        )

    return result, explanation, definition


def iqr(data, help=False, what=False):
    d = _sorted(data)
    n = len(d)

    q1_index = n // 4
    q3_index = 3 * n // 4

    q1 = d[q1_index]
    q3 = d[q3_index]
    result = q3 - q1

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating IQR.\n"
        explanation += f"Sorted data = {d}\n"
        explanation += f"Q1 ≈ position {q1_index} → {q1}\n"
        explanation += f"Q3 ≈ position {q3_index} → {q3}\n"
        explanation += f"IQR = Q3 - Q1 = {result}\n"

    definition = ""
    if what:
        definition += (
            "IQR (Interquartile Range) measures the spread of the middle 50% of data.\n"
            f"Example: iqr({data}) = {result}\n"
        )

    return result, explanation, definition


def variance(data, help=False, what=False):
    m, _, _ = mean(data)
    n = len(data)

    diffs = [(x - m) ** 2 for x in data]
    result = sum(diffs) / (n - 1)  # sample variance

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating variance.\n"
        explanation += f"Mean = {m}\n"
        explanation += f"(x - mean)^2 values = {diffs}\n"
        explanation += f"Variance = sum / (n - 1)\n"
        explanation += f"= {result}\n"

    definition = ""
    if what:
        definition += (
            "Variance measures how far data values spread out from the mean.\n"
            f"Example: variance({data}) = {result}\n"
        )

    return result, explanation, definition

def std(data, help=False, what=False):
    v, _, _ = variance(data)
    result = math.sqrt(v)

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating std.\n"
        explanation += f"Variance = {v}\n"
        explanation += f"Std = sqrt(variance) = {result}\n"

    definition = ""
    if what:
        definition += (
            "Standard deviation is the square root of variance.\n"
            f"Example: std({data}) = {result}\n"
        )

    return result, explanation, definition

def mad(data, help=False, what=False):
    med, _, _ = median(data)
    abs_devs = [abs(x - med) for x in data]
    result, _, _ = median(abs_devs)

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating MAD.\n"
        explanation += f"Median = {med}\n"
        explanation += f"Absolute deviations = {abs_devs}\n"
        explanation += f"MAD = median(abs deviations) = {result}\n"

    definition = ""
    if what:
        definition += (
            "MAD is a robust measure of variability based on median.\n"
            f"Example: mad({data}) = {result}\n"
        )

    return result, explanation, definition

def skewness(data, help=False, what=False):
    m, _, _ = mean(data)
    s, _, _ = std(data)
    n = len(data)

    numerators = [(x - m) ** 3 for x in data]
    result = (n / ((n - 1) * (n - 2))) * sum(numerators) / (s ** 3)

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating skewness.\n"
        explanation += f"Mean = {m}, Std = {s}\n"
        explanation += f"(x - mean)^3 = {numerators}\n"
        explanation += f"Skewness = {result}\n"

    definition = ""
    if what:
        definition += (
            "Skewness measures asymmetry of the distribution.\n"
            f"Example: skewness({data}) = {result}\n"
        )

    return result, explanation, definition

def kurtosis(data, help=False, what=False):
    m, _, _ = mean(data)
    s, _, _ = std(data)
    n = len(data)

    numerators = [(x - m) ** 4 for x in data]
    result = (n*(n+1)/((n-1)*(n-2)*(n-3))) * (sum(numerators) / (s**4)) - (3*(n-1)**2/((n-2)*(n-3)))

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating kurtosis.\n"
        explanation += f"Mean = {m}, Std = {s}\n"
        explanation += f"(x - mean)^4 = {numerators}\n"
        explanation += f"Kurtosis = {result}\n"

    definition = ""
    if what:
        definition += (
            "Kurtosis measures the 'tailedness' of the distribution.\n"
            f"Example: kurtosis({data}) = {result}\n"
        )

    return result, explanation, definition

def covariance(x, y, help=False, what=False):
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    n = len(x)
    mean_x, _, _ = mean(x)
    mean_y, _, _ = mean(y)

    products = [(x[i] - mean_x) * (y[i] - mean_y) for i in range(n)]
    result = sum(products) / (n - 1)

    explanation = ""
    if help:
        explanation += "Step-by-step: Calculating covariance.\n"
        explanation += f"Mean of x = {mean_x}, Mean of y = {mean_y}\n"
        explanation += f"(x - mean_x)*(y - mean_y) = {products}\n"
        explanation += f"Covariance = sum / (n - 1) = {result}\n"

    definition = ""
    if what:
        definition += (
            "Covariance measures how two variables change together.\n"
            "Positive ⇒ variables increase together.\n"
            "Negative ⇒ inverse relation.\n"
            f"Example: covariance({x}, {y}) = {result}\n"
        )

    return result, explanation, definition

def _rank(data):
    sorted_data = sorted(set(data))
    rank_map = {value: i+1 for i, value in enumerate(sorted_data)}
    return [rank_map[v] for v in data]

def _kendall_tau(x, y):
    n = len(x)
    concord = 0
    discord = 0

    for i in range(n):
        for j in range(i+1, n):
            sign1 = x[i] - x[j]
            sign2 = y[i] - y[j]
            if sign1 * sign2 > 0:
                concord += 1
            elif sign1 * sign2 < 0:
                discord += 1
    return (concord - discord) / (concord + discord)

def correlation(x, y, method="pearson", help=False, what=False):
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    explanation = ""
    definition = ""

    if method == "pearson":
        cov, _, _ = covariance(x, y)
        std_x, _, _ = std(x)
        std_y, _, _ = std(y)
        result = cov / (std_x * std_y)

        if help:
            explanation += "Pearson Correlation:\n"
            explanation += f"Cov(x,y) = {cov}\n"
            explanation += f"Std(x) = {std_x}, Std(y) = {std_y}\n"
            explanation += f"Correlation = cov / (std_x * std_y) = {result}\n"

        if what:
            definition += (
                "Pearson correlation measures linear relationship between two variables.\n"
                "- Range: -1 to +1\n"
                "- +1 strong positive linear\n"
                "- -1 strong negative linear\n"
                f"Example: correlation({x}, {y}) = {result}\n"
            )

        return result, explanation, definition

    elif method == "spearman":
        rx = _rank(x)
        ry = _rank(y)
        result, e, w = correlation(rx, ry, method="pearson")

        if help:
            explanation += "Spearman Rank Correlation:\n"
            explanation += f"Ranks for x: {rx}\nRanks for y: {ry}\n"
            explanation += e

        if what:
            definition += (
                "Spearman correlation measures monotonic relationship using rank order.\n"
                f"Example: spearman({x}, {y}) = {result}\n"
            )

        return result, explanation, definition

    elif method == "kendall":
        result = _kendall_tau(x, y)

        if help:
            explanation += "Kendall Tau Correlation:\n"
            explanation += f"Kendall tau = {result}\n"

        if what:
            definition += (
                "Kendall Tau measures ordinal association between two variables.\n"
                "Counts concordant vs discordant pairs.\n"
                f"Example: kendall({x}, {y}) = {result}\n"
            )

        return result, explanation, definition

    else:
        raise ValueError("method must be: pearson, spearman, kendall")


def quantiles(data, q=[0.25, 0.5, 0.75], help=False, what=False):
    d = sorted(data)
    n = len(d)

    result = {}
    explanation = ""

    for prob in q:
        index = prob * (n - 1)
        low = int(index)
        high = math.ceil(index)

        if low == high:
            value = d[int(index)]
        else:
            # linear interpolation
            value = d[low] + (d[high] - d[low]) * (index - low)

        result[prob] = value

    if help:
        explanation += "Step-by-step quantile calculation:\n"
        explanation += f"Sorted data = {d}\n"
        for prob in q:
            explanation += f"Quantile {prob}: {result[prob]}\n"

    definition = ""
    if what:
        definition += (
            "Quantiles split data into equal-sized intervals.\n"
            "Examples:\n"
            " - 0.25 → first quartile\n"
            " - 0.50 → median\n"
            " - 0.75 → third quartile\n"
            f"Result: {result}\n"
        )

    return result, explanation, definition


def summary(data, help=False, what=False):
    count_val = len(data)
    mean_val, _, _ = mean(data)
    std_val, _, _ = std(data)
    min_val = min(data)
    max_val = max(data)
    quants, _, _ = quantiles(data)

    result = {
        "count": count_val,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "25%": quants[0.25],
        "50%": quants[0.5],
        "75%": quants[0.75],
        "max": max_val
    }

    explanation = ""
    if help:
        explanation += "Step-by-step descriptive summary:\n"
        for k, v in result.items():
            explanation += f"{k}: {v}\n"

    definition = ""
    if what:
        definition += (
            "The summary function gives a statistical overview similar to pandas.describe().\n"
            f"Example: summary({data}) = {result}\n"
        )

    return result, explanation, definition
