import numpy as np
from math import factorial
from scipy.stats import bernoulli, binom, poisson, norm, uniform, expon

def nCr(n, r, help=False, what=False):
    result = factorial(n) // (factorial(r) * factorial(n - r))
    explanation = ""
    if help:
        explanation += f"Combination nCr: choose {r} items from {n}\n"
        explanation += f"nCr = n! / (r!(n-r)!) = {result}\n"
    definition = ""
    if what:
        definition += f"nCr calculates the number of combinations of {n} items taken {r} at a time.\n"
    return result, explanation, definition

def nPr(n, r, help=False, what=False):
    result = factorial(n) // factorial(n - r)
    explanation = ""
    if help:
        explanation += f"Permutation nPr: arrange {r} items from {n}\n"
        explanation += f"nPr = n! / (n-r)! = {result}\n"
    definition = ""
    if what:
        definition += f"nPr calculates the number of ordered arrangements of {r} items from {n}.\n"
    return result, explanation, definition

def fact(n, help=False, what=False):
    result = factorial(n)
    explanation = ""
    if help:
        explanation += f"Factorial: {n}! = {result}\n"
    definition = ""
    if what:
        definition += f"Factorial of n, denoted n!, is the product of all positive integers up to n.\n"
    return result, explanation, definition

def pmf(distribution, x, params={}, help=False, what=False):
    x = np.array(x)
    if distribution.lower() == 'bernoulli':
        p = params.get('p', 0.5)
        result = bernoulli.pmf(x, p)
    elif distribution.lower() == 'binomial':
        n = params.get('n', 1)
        p = params.get('p', 0.5)
        result = binom.pmf(x, n, p)
    elif distribution.lower() == 'poisson':
        mu = params.get('mu', 1)
        result = poisson.pmf(x, mu)
    else:
        raise ValueError("Unsupported discrete distribution for PMF")

    explanation = ""
    if help:
        explanation += f"PMF of {distribution} at x={x} with parameters {params} = {result}\n"

    definition = ""
    if what:
        definition += f"PMF returns probability of each discrete outcome in the distribution.\n"

    return result, explanation, definition


def pdf(distribution, x, params={}, help=False, what=False):
    x = np.array(x)
    if distribution.lower() == 'normal':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        result = norm.pdf(x, loc=mu, scale=sigma)
    elif distribution.lower() == 'uniform':
        a = params.get('a', 0)
        b = params.get('b', 1)
        result = uniform.pdf(x, loc=a, scale=b-a)
    elif distribution.lower() == 'exponential':
        lam = params.get('lam', 1)
        result = expon.pdf(x, scale=1/lam)
    else:
        raise ValueError("Unsupported continuous distribution for PDF")

    explanation = ""
    if help:
        explanation += f"PDF of {distribution} at x={x} with parameters {params} = {result}\n"

    definition = ""
    if what:
        definition += f"PDF returns probability density for continuous distribution.\n"

    return result, explanation, definition


def cdf(distribution, x, params={}, help=False, what=False):
    x = np.array(x)
    if distribution.lower() == 'bernoulli':
        p = params.get('p', 0.5)
        result = bernoulli.cdf(x, p)
    elif distribution.lower() == 'binomial':
        n = params.get('n', 1)
        p = params.get('p', 0.5)
        result = binom.cdf(x, n, p)
    elif distribution.lower() == 'poisson':
        mu = params.get('mu', 1)
        result = poisson.cdf(x, mu)
    elif distribution.lower() == 'normal':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        result = norm.cdf(x, loc=mu, scale=sigma)
    elif distribution.lower() == 'uniform':
        a = params.get('a', 0)
        b = params.get('b', 1)
        result = uniform.cdf(x, loc=a, scale=b-a)
    elif distribution.lower() == 'exponential':
        lam = params.get('lam', 1)
        result = expon.cdf(x, scale=1/lam)
    else:
        raise ValueError("Unsupported distribution for CDF")

    explanation = ""
    if help:
        explanation += f"CDF of {distribution} at x={x} with parameters {params} = {result}\n"

    definition = ""
    if what:
        definition += f"CDF returns cumulative probability up to each x.\n"

    return result, explanation, definition

def bayes_theorem(prior, likelihood, evidence, help=False, what=False):
    posterior = (likelihood * prior) / evidence
    explanation = ""
    if help:
        explanation += f"Bayes Theorem:\nPrior={prior}, Likelihood={likelihood}, Evidence={evidence}\n"
        explanation += f"Posterior = (Likelihood * Prior) / Evidence = {posterior}\n"
    definition = ""
    if what:
        definition += "Bayes theorem computes posterior probability given prior, likelihood, and evidence.\n"
    return posterior, explanation, definition

def posterior(prior, likelihood, help=False, what=False):
    post = likelihood * prior
    post = post / np.sum(post)
    explanation = ""
    if help:
        explanation += f"Posterior probabilities (normalized):\nPrior={prior}\nLikelihood={likelihood}\nPosterior={post}\n"
    definition = ""
    if what:
        definition += "Calculates normalized posterior probabilities given prior and likelihood.\n"
    return post, explanation, definition
