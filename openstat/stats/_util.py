"""Numerical helpers shared by analyses and graphs."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[~np.isnan(x)]


def quartiles(x):
    """Q1, median, Q3 by the (n+1)p position rule."""
    return np.quantile(np.asarray(x, dtype=float), [0.25, 0.5, 0.75], method="weibull")


def median_ranks(n: int) -> np.ndarray:
    """Benard's approximation of the median rank plotting positions."""
    return (np.arange(1, n + 1) - 0.3) / (n + 0.4)


def anderson_darling_normal(values) -> tuple[float, float]:
    """Anderson-Darling statistic and p-value for normality with estimated mean and sd.

    P-value from the D'Agostino & Stephens (1986) approximation.
    """
    x = np.sort(clean(values))
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float("nan"), float("nan")
    z = sps.norm.cdf((x - x.mean()) / x.std(ddof=1))
    z = np.clip(z, 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    a2 = -n - np.sum((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1]))) / n
    a = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if a >= 0.6:
        p = np.exp(1.2937 - 5.709 * a + 0.0186 * a ** 2)
    elif a >= 0.34:
        p = np.exp(0.9177 - 4.279 * a - 1.38 * a ** 2)
    elif a > 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * a - 59.938 * a ** 2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * a - 223.73 * a ** 2)
    return float(a2), float(min(max(p, 0.0), 1.0))


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and its t confidence limits."""
    x = clean(values)
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan")
    half = sps.t.ppf(0.5 + confidence / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return mean, mean - half, mean + half
