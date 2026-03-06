import numpy as np
import math
from math import erf
from scipy.special import erfinv, comb, beta as beta_func, betaincinv
from scipy.special import gamma as gamma_func

class Distribution:
    """Base class for all distributions."""
    pass

class Normal(Distribution):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x, help=False, what=False):
        x = np.array(x, dtype=float)
        coef = 1 / (self.sigma * np.sqrt(2 * np.pi))
        result = coef * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

        explanation = ""
        if help:
            explanation += (
                "Normal PDF formula:\n"
                "f(x) = (1 / (σ√(2π))) * exp(-(x - μ)^2 / (2σ^2))\n"
                f"mu = {self.mu}, sigma = {self.sigma}\n"
            )

        definition = ""
        if what:
            definition += (
                "PDF gives the probability density at each x value.\n"
                "Higher PDF ⇒ value more likely.\n"
                f"Input: {x.tolist()}, Output: {result.tolist()}\n"
            )

        return result, explanation, definition

    def cdf(self, x, help=False, what=False):
        x = np.array(x, dtype=float)
        z = (x - self.mu) / (self.sigma * np.sqrt(2))
        result = 0.5 * (1 + erf(z))

        explanation = ""
        if help:
            explanation += (
                "Normal CDF formula:\n"
                "F(x) = 1/2 * (1 + erf((x - μ) / (σ√2)))\n"
            )

        definition = ""
        if what:
            definition += (
                "CDF gives probability that X ≤ x.\n"
                "Ranges from 0 to 1.\n"
            )

        return result, explanation, definition

    def ppf(self, q, help=False, what=False):
        q = np.array(q, dtype=float)
        result = self.mu + self.sigma * np.sqrt(2) * erfinv(2 * q - 1)

        explanation = ""
        if help:
            explanation += (
                "Normal PPF formula:\n"
                "x = μ + σ * √2 * erfinv(2q - 1)\n"
            )

        definition = ""
        if what:
            definition += (
                "PPF returns the x value for which CDF(x) = q.\n"
                "Used for computing quantiles.\n"
            )

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.normal(self.mu, self.sigma, size)

        explanation = ""
        if help:
            explanation += f"Sampling {size} values from N({self.mu}, {self.sigma}²)\n"

        definition = ""
        if what:
            definition += (
                "Random samples drawn from the distribution.\n"
                f"Example output: {result.tolist()}\n"
            )

        return result, explanation, definition

    def mean(self, help=False, what=False):
        result = self.mu

        explanation = "Mean of normal distribution = μ\n" if help else ""
        definition = "Expected value (center) of distribution.\n" if what else ""

        return result, explanation, definition

    def variance(self, help=False, what=False):
        result = self.sigma ** 2

        explanation = "Variance = σ²\n" if help else ""
        definition = "Spread of the distribution.\n" if what else ""

        return result, explanation, definition

class Uniform(Distribution):
    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

    def pdf(self, x, help=False, what=False):
        x = np.array(x)
        result = np.where((x >= self.a) & (x <= self.b), 1 / (self.b - self.a), 0)

        explanation = ""
        if help:
            explanation += f"Uniform PDF = 1 / (b - a) for x in [{self.a}, {self.b}]\n"

        definition = ""
        if what:
            definition += (
                "Uniform distribution assigns equal density to all values in range.\n"
            )

        return result, explanation, definition

    def cdf(self, x, help=False, what=False):
        x = np.array(x, dtype=float)
        result = np.clip((x - self.a) / (self.b - self.a), 0, 1)

        explanation = ""
        if help:
            explanation += "CDF = (x - a)/(b - a) for x in range.\n"

        definition = ""
        if what:
            definition += "CDF gives probability X ≤ x.\n"

        return result, explanation, definition

    def ppf(self, q, help=False, what=False):
        q = np.array(q, dtype=float)
        result = self.a + q * (self.b - self.a)

        explanation = "Inverse CDF = a + q(b - a)\n" if help else ""
        definition = "Returns quantile.\n" if what else ""

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.uniform(self.a, self.b, size)

        explanation = f"Sampling {size} values from U({self.a}, {self.b})\n" if help else ""
        definition = "Uniform random samples.\n" if what else ""

        return result, explanation, definition

    def mean(self, help=False, what=False):
        result = (self.a + self.b) / 2
        return result, "Mean = (a+b)/2\n" if help else "", "Center of distribution.\n" if what else ""

    def variance(self, help=False, what=False):
        result = ((self.b - self.a) ** 2) / 12
        return result, "Var = (b-a)^2 / 12\n" if help else "", "Spread.\n" if what else ""

class Exponential(Distribution):
    def __init__(self, lam=1.0):  
        self.lam = lam

    def pdf(self, x, help=False, what=False):
        x = np.array(x)
        result = self.lam * np.exp(-self.lam * x) * (x >= 0)

        explanation = "PDF = λ e^{-λx}\n" if help else ""
        definition = "Density decreases exponentially.\n" if what else ""

        return result, explanation, definition

    def cdf(self, x, help=False, what=False):
        x = np.array(x)
        result = (1 - np.exp(-self.lam * x)) * (x >= 0)

        explanation = "CDF = 1 - e^{-λx}\n" if help else ""
        definition = "Probability X ≤ x.\n" if what else ""

        return result, explanation, definition

    def ppf(self, q, help=False, what=False):
        q = np.array(q)
        result = -np.log(1 - q) / self.lam

        explanation = "Inverse CDF = -ln(1-q)/λ\n" if help else ""
        definition = "Quantile function.\n" if what else ""

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.exponential(1 / self.lam, size)

        explanation = "" if not help else f"Sampling exponential(shape={size})\n"
        definition = "Exponential random sample.\n" if what else ""

        return result, explanation, definition

    def mean(self, help=False, what=False):
        return 1 / self.lam, "Mean = 1/λ\n" if help else "", ""

    def variance(self, help=False, what=False):
        return 1 / (self.lam ** 2), "Var = 1/λ²\n" if help else "", ""

class Bernoulli:
    def __init__(self, p=0.5):
        self.p = p

    def pmf(self, k, help=False, what=False):
        k = np.array(k)
        result = np.where((k == 0) | (k == 1), self.p ** k * (1 - self.p) ** (1 - k), 0)

        explanation = ""
        if help:
            explanation += "Bernoulli PMF: P(X=k) = p^k * (1-p)^(1-k)\n"

        definition = ""
        if what:
            definition += "Probability mass function for success/failure experiment.\n"

        return result, explanation, definition

    def cdf(self, k, help=False, what=False):
        k = np.array(k)
        result = np.where(k < 0, 0, np.where(k < 1, 1 - self.p, 1))

        explanation = ""
        if help:
            explanation += "CDF: cumulative probability P(X ≤ k)\n"

        definition = ""
        if what:
            definition += "CDF gives probability that X ≤ k.\n"

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.binomial(1, self.p, size)
        explanation = f"Sample {size} values with p={self.p}\n" if help else ""
        definition = "Random Bernoulli samples.\n" if what else ""
        return result, explanation, definition

    def mean(self, help=False, what=False):
        return self.p, f"Mean = p = {self.p}\n" if help else "", "Expected value of Bernoulli.\n" if what else ""

    def variance(self, help=False, what=False):
        var = self.p * (1 - self.p)
        return var, f"Variance = p*(1-p) = {var}\n" if help else "", "Spread of Bernoulli.\n" if what else ""


class Binomial:
    def __init__(self, n=1, p=0.5):
        self.n = n
        self.p = p

    def pmf(self, k, help=False, what=False):
        k = np.array(k)
        result = comb(self.n, k) * self.p ** k * (1 - self.p) ** (self.n - k)

        explanation = ""
        if help:
            explanation += f"Binomial PMF: P(X=k) = C(n,k)*p^k*(1-p)^(n-k)\n"

        definition = ""
        if what:
            definition += "Probability of k successes in n independent trials.\n"

        return result, explanation, definition

    def cdf(self, k, help=False, what=False):
        k = np.array(k)
        result = np.array([sum(comb(self.n, i) * self.p ** i * (1 - self.p) ** (self.n - i) for i in range(int(np.floor(ki)) + 1)) for ki in k])

        explanation = ""
        if help:
            explanation += "CDF: sum of PMF from 0 to k\n"

        definition = ""
        if what:
            definition += "Probability X ≤ k successes.\n"

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.binomial(self.n, self.p, size)
        explanation = f"Random sample of size {size} from Binomial(n={self.n}, p={self.p})\n" if help else ""
        definition = "Random binomial samples.\n" if what else ""
        return result, explanation, definition

    def mean(self, help=False, what=False):
        m = self.n * self.p
        return m, f"Mean = n*p = {m}\n" if help else "", "Expected value.\n" if what else ""

    def variance(self, help=False, what=False):
        var = self.n * self.p * (1 - self.p)
        return var, f"Variance = n*p*(1-p) = {var}\n" if help else "", "Spread of distribution.\n" if what else ""


class Poisson:
    def __init__(self, lam=1.0):
        self.lam = lam

    def pmf(self, k, help=False, what=False):
        k = np.array(k)
        result = (self.lam ** k) * np.exp(-self.lam) / np.array([math.factorial(int(ki)) for ki in k])

        explanation = ""
        if help:
            explanation += "Poisson PMF: P(X=k) = λ^k * e^-λ / k!\n"

        definition = ""
        if what:
            definition += "Probability of k events in fixed interval.\n"

        return result, explanation, definition

    def cdf(self, k, help=False, what=False):
        k = np.array(k)
        result = np.array([sum((self.lam ** i) * np.exp(-self.lam) / math.factorial(i) for i in range(int(ki) + 1)) for ki in k])

        explanation = ""
        if help:
            explanation += "CDF = sum of PMF from 0 to k\n"

        definition = ""
        if what:
            definition += "Probability X ≤ k.\n"

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.poisson(self.lam, size)
        explanation = f"Random sample size {size} from Poisson(λ={self.lam})\n" if help else ""
        definition = "Random Poisson samples.\n" if what else ""
        return result, explanation, definition

    def mean(self, help=False, what=False):
        return self.lam, f"Mean = λ = {self.lam}\n" if help else "", "Expected value.\n" if what else ""

    def variance(self, help=False, what=False):
        return self.lam, f"Variance = λ = {self.lam}\n" if help else "", "Spread.\n" if what else ""

class Beta:
    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b

    def pdf(self, x, help=False, what=False):
        x = np.array(x)
        result = np.where((x >= 0) & (x <= 1), x ** (self.a - 1) * (1 - x) ** (self.b - 1) / beta_func(self.a, self.b), 0)

        explanation = ""
        if help:
            explanation += "Beta PDF: f(x) = x^(a-1)*(1-x)^(b-1)/B(a,b)\n"

        definition = ""
        if what:
            definition += "Continuous distribution on [0,1], often for probabilities.\n"

        return result, explanation, definition

    def cdf(self, x, help=False, what=False):
        x = np.array(x)
        result = np.array([betaincinv(self.a, self.b, xi) if 0 <= xi <= 1 else 0 for xi in x])

        explanation = ""
        if help:
            explanation += "CDF = integral of PDF from 0 to x\n"

        definition = ""
        if what:
            definition += "Probability X ≤ x.\n"

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.beta(self.a, self.b, size)
        explanation = f"Random sample size {size} from Beta(a={self.a}, b={self.b})\n" if help else ""
        definition = "Beta random samples.\n" if what else ""
        return result, explanation, definition

    def mean(self, help=False, what=False):
        m = self.a / (self.a + self.b)
        return m, f"Mean = a/(a+b) = {m}\n" if help else "", "Expected value.\n" if what else ""

    def variance(self, help=False, what=False):
        var = (self.a * self.b) / ((self.a + self.b) ** 2 * (self.a + self.b + 1))
        return var, f"Variance = ab/((a+b)^2*(a+b+1)) = {var}\n" if help else "", "Spread of distribution.\n" if what else ""

class Gamma:
    def __init__(self, shape=1.0, scale=1.0):
        self.shape = shape
        self.scale = scale

    def pdf(self, x, help=False, what=False):
        x = np.array(x)
        result = np.where(x >= 0, x ** (self.shape - 1) * np.exp(-x / self.scale) / (gamma_func(self.shape) * self.scale ** self.shape), 0)

        explanation = ""
        if help:
            explanation += "Gamma PDF: f(x) = x^(k-1)*exp(-x/θ)/(Γ(k)*θ^k)\n"

        definition = ""
        if what:
            definition += "Continuous distribution, generalization of Exponential.\n"

        return result, explanation, definition

    def cdf(self, x, help=False, what=False):
        x = np.array(x)
        from scipy.stats import gamma as sp_gamma
        result = sp_gamma.cdf(x, self.shape, scale=self.scale)

        explanation = ""
        if help:
            explanation += "CDF = integral of PDF from 0 to x\n"

        definition = ""
        if what:
            definition += "Probability X ≤ x.\n"

        return result, explanation, definition

    def rvs(self, size=1, help=False, what=False):
        result = np.random.gamma(self.shape, self.scale, size)
        explanation = f"Random sample size {size} from Gamma(shape={self.shape}, scale={self.scale})\n" if help else ""
        definition = "Gamma random samples.\n" if what else ""
        return result, explanation, definition

    def mean(self, help=False, what=False):
        m = self.shape * self.scale
        return m, f"Mean = shape*scale = {m}\n" if help else "", "Expected value.\n" if what else ""

    def variance(self, help=False, what=False):
        var = self.shape * self.scale ** 2
        return var, f"Variance = shape*scale^2 = {var}\n" if help else "", "Spread of distribution.\n" if what else ""