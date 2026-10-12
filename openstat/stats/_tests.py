"""Shared pieces of hypothesis tests: alternatives, p-values and confidence bounds."""
from __future__ import annotations

import numpy as np

from openstat.core.registry import Param

NOT_EQUAL, LESS, GREATER = "Not equal", "Less than", "Greater than"
ALTERNATIVES = [NOT_EQUAL, LESS, GREATER]
SCIPY_ALT = {NOT_EQUAL: "two-sided", LESS: "less", GREATER: "greater"}

CONFIDENCE = Param("confidence", "Confidence level (%)", "number", default=95.0, minimum=1, maximum=99.999)
ALTERNATIVE = Param("alternative", "Alternative hypothesis", "choice", default=NOT_EQUAL, choices=ALTERNATIVES)
ALPHA = Param("alpha", "Significance level", "number", default=0.05, minimum=0.0001, maximum=0.5)


def p_value(statistic: float, dist, alternative: str = NOT_EQUAL) -> float:
    """P-value of `statistic` under the frozen null distribution `dist` (symmetric for two-sided)."""
    if np.isnan(statistic):
        return float("nan")
    if alternative == LESS:
        return float(dist.cdf(statistic))
    if alternative == GREATER:
        return float(dist.sf(statistic))
    return float(min(1.0, 2 * min(dist.cdf(statistic), dist.sf(statistic))))


def bounds(estimate: float, se: float, dist, confidence: float, alternative: str = NOT_EQUAL):
    """Confidence limits; one-sided alternatives give a single bound (the other is nan)."""
    level = confidence / 100
    nan = float("nan")
    if alternative == LESS:
        return nan, estimate + dist.ppf(level) * se
    if alternative == GREATER:
        return estimate - dist.ppf(level) * se, nan
    half = dist.ppf(0.5 + level / 2) * se
    return estimate - half, estimate + half


def tail_levels(confidence: float, alternative: str = NOT_EQUAL) -> tuple[float | None, float | None]:
    """Lower and upper tail probabilities for a quantile-based interval."""
    alpha = 1 - confidence / 100
    if alternative == LESS:
        return None, 1 - alpha
    if alternative == GREATER:
        return alpha, None
    return alpha / 2, 1 - alpha / 2
