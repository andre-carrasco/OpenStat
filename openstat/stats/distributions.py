"""Catalogue of probability distributions, shared by Calc and by later analyses."""
from __future__ import annotations

import math

from scipy import stats as sps

# label -> factory(p1, p2, p3) returning a frozen scipy distribution
DISTRIBUTIONS = {
    "Normal (mean, standard deviation)": lambda a, b, c: sps.norm(a, b),
    "Uniform (lower, upper)": lambda a, b, c: sps.uniform(a, b - a),
    "Exponential (scale, threshold)": lambda a, b, c: sps.expon(b, a),
    "Lognormal (location, scale)": lambda a, b, c: sps.lognorm(b, scale=math.exp(a)),
    "Weibull (shape, scale)": lambda a, b, c: sps.weibull_min(a, scale=b),
    "Gamma (shape, scale)": lambda a, b, c: sps.gamma(a, scale=b),
    "Beta (first shape, second shape)": lambda a, b, c: sps.beta(a, b),
    "Logistic (location, scale)": lambda a, b, c: sps.logistic(a, b),
    "Cauchy (location, scale)": lambda a, b, c: sps.cauchy(a, b),
    "Laplace (location, scale)": lambda a, b, c: sps.laplace(a, b),
    "Smallest extreme value (location, scale)": lambda a, b, c: sps.gumbel_l(a, b),
    "Largest extreme value (location, scale)": lambda a, b, c: sps.gumbel_r(a, b),
    "Triangular (lower, mode, upper)": lambda a, b, c: sps.triang((b - a) / (c - a), a, c - a),
    "t (degrees of freedom)": lambda a, b, c: sps.t(a),
    "Chi-square (degrees of freedom)": lambda a, b, c: sps.chi2(a),
    "F (numerator df, denominator df)": lambda a, b, c: sps.f(a, b),
    "Binomial (trials, event probability)": lambda a, b, c: sps.binom(int(a), b),
    "Bernoulli (event probability)": lambda a, b, c: sps.bernoulli(a),
    "Poisson (mean)": lambda a, b, c: sps.poisson(a),
    "Geometric (event probability)": lambda a, b, c: sps.geom(a),
    "Negative binomial (events, event probability)": lambda a, b, c: sps.nbinom(int(a), b),
    "Hypergeometric (population, events in population, sample size)":
        lambda a, b, c: sps.hypergeom(int(a), int(b), int(c)),
    "Integer (minimum, maximum)": lambda a, b, c: sps.randint(int(a), int(b) + 1),
}

NAMES = list(DISTRIBUTIONS)


def make(name: str, p1: float = 0.0, p2: float = 1.0, p3: float = 0.0):
    try:
        dist = DISTRIBUTIONS[name](p1, p2, p3)
        valid = not math.isnan(float(dist.ppf(0.5)))  # scipy gives nan for invalid parameters
    except Exception as exc:
        raise ValueError(f"Invalid parameters for {name}") from exc
    if not valid:
        raise ValueError(f"Invalid parameters for {name}")
    return dist


def is_discrete(dist) -> bool:
    return hasattr(dist.dist, "pmf")


def density(dist, x):
    return dist.pmf(x) if is_discrete(dist) else dist.pdf(x)
