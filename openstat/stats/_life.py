"""Life-data models: maximum likelihood for location-scale families with censoring, and Kaplan-Meier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gamma
from scipy import stats as sps
from statsmodels.tools.numdiff import approx_hess

# name -> (standardized distribution of the location-scale family, model log(time)?, scale fixed at 1?)
FAMILIES = {
    "Weibull": (sps.gumbel_l, True, False),
    "Lognormal": (sps.norm, True, False),
    "Exponential": (sps.gumbel_l, True, True),
    "Loglogistic": (sps.logistic, True, False),
    "Normal": (sps.norm, False, False),
    "Smallest extreme value": (sps.gumbel_l, False, False),
    "Logistic": (sps.logistic, False, False),
}
NAMES = list(FAMILIES)


@dataclass
class LifeFit:
    name: str
    beta: np.ndarray          # location coefficients (one per column of X)
    sigma: float
    cov: np.ndarray           # covariance of (beta..., log sigma); log sigma absent when sigma is fixed
    loglik: float
    n: int
    failures: int
    columns: list[str]

    @property
    def base(self):
        return FAMILIES[self.name][0]

    @property
    def log_time(self) -> bool:
        return FAMILIES[self.name][1]

    @property
    def fixed_scale(self) -> bool:
        return FAMILIES[self.name][2]

    @property
    def aic(self) -> float:
        return 2 * (len(self.beta) + (0 if self.fixed_scale else 1)) - 2 * self.loglik

    def se_beta(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov)[:len(self.beta)])

    def se_sigma(self) -> float:
        return 0.0 if self.fixed_scale else float(self.sigma * np.sqrt(self.cov[-1, -1]))

    def quantile(self, p, x=None, confidence=95.0):
        """Estimate and confidence limits of the p-th quantile of time at covariate row x."""
        x = np.ones(1) if x is None else np.asarray(x, dtype=float)
        z = float(self.base.ppf(p))
        gradient = np.concatenate([x, [] if self.fixed_scale else [z * self.sigma]])
        estimate = float(x @ self.beta + z * self.sigma)
        se = float(np.sqrt(max(gradient @ self.cov @ gradient, 0.0)))
        half = sps.norm.ppf(0.5 + confidence / 200) * se
        lo, hi = estimate - half, estimate + half
        return tuple(np.exp(v) for v in (estimate, lo, hi)) if self.log_time else (estimate, lo, hi)

    def survival(self, t, x=None):
        x = np.ones(1) if x is None else np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(t) if self.log_time else t
        return self.base.sf((y - x @ self.beta) / self.sigma)

    def mean(self) -> float:
        """Mean time to failure for a model without covariates."""
        mu, s = float(self.beta[0]), self.sigma
        if self.name in ("Weibull", "Exponential"):
            return float(np.exp(mu) * gamma(1 + s))
        if self.name == "Lognormal":
            return float(np.exp(mu + s ** 2 / 2))
        if self.name == "Loglogistic":
            return float(np.exp(mu) * np.pi * s / np.sin(np.pi * s)) if s < 1 else float("inf")
        if self.name == "Smallest extreme value":
            return mu - np.euler_gamma * s
        return mu


def fit_life(lower, upper, X=None, name="Weibull", columns=None) -> LifeFit:
    """Maximum likelihood with exact (lower == upper), right (upper = inf), left (lower = -inf or 0)
    and interval censored observations."""
    base, log_time, fixed = FAMILIES[name]
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    n = len(lower)
    X = np.ones((n, 1)) if X is None else np.asarray(X, dtype=float)
    if log_time:
        if (upper <= 0).any() or (lower < 0).any():
            raise ValueError(f"The {name} distribution needs positive times")
        with np.errstate(divide="ignore"):
            lower, upper = np.log(lower), np.log(upper)
    exact = np.isclose(lower, upper)
    if exact.sum() + (~exact & np.isfinite(lower) & np.isfinite(upper)).sum() < 1:
        raise ValueError("At least one failure (uncensored observation) is needed")
    if n <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Not enough observations to fit the model")
    mid = np.where(np.isfinite(lower) & np.isfinite(upper), (lower + upper) / 2,
                   np.where(np.isfinite(lower), lower, upper))
    beta0, *_ = np.linalg.lstsq(X, mid, rcond=None)
    sigma0 = max(float(np.std(mid - X @ beta0)), 1e-3)
    k = X.shape[1]

    def negative_loglik(theta):
        beta = theta[:k]
        sigma = 1.0 if fixed else np.exp(theta[k])
        mu = X @ beta
        total = np.sum(base.logpdf((lower[exact] - mu[exact]) / sigma) - np.log(sigma))
        rest = ~exact
        if rest.any():
            hi = base.cdf((upper[rest] - mu[rest]) / sigma)
            lo = base.cdf((lower[rest] - mu[rest]) / sigma)
            right = np.isinf(upper[rest])
            prob = np.where(right, base.sf((lower[rest] - mu[rest]) / sigma), hi - lo)
            total += np.sum(np.log(np.clip(prob, 1e-300, None)))
        return -total if np.isfinite(total) else 1e300

    start = np.concatenate([beta0, [] if fixed else [np.log(sigma0)]])
    best = optimize.minimize(negative_loglik, start, method="Nelder-Mead",
                             options={"maxiter": 4000 * len(start), "xatol": 1e-9, "fatol": 1e-10})
    polished = optimize.minimize(negative_loglik, best.x, method="BFGS")
    theta = polished.x if polished.fun <= best.fun + 1e-9 and np.all(np.isfinite(polished.x)) else best.x
    try:
        cov = np.linalg.inv(approx_hess(theta, negative_loglik))
    except np.linalg.LinAlgError:
        cov = np.full((len(theta), len(theta)), np.nan)
    sigma = 1.0 if fixed else float(np.exp(theta[k]))
    loglik = -float(negative_loglik(theta))
    if log_time:
        # density of time rather than of log(time), so that all families are comparable
        loglik -= float(lower[exact].sum())
    return LifeFit(name, theta[:k], sigma, cov, loglik, n, int(exact.sum()), list(columns or ["Intercept"]))


def right_censored(time, failed) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    return time, np.where(np.asarray(failed, dtype=bool), time, np.inf)


def kaplan_meier(time, failed, confidence=95.0) -> pd.DataFrame:
    """Product-limit survival estimates with Greenwood standard errors."""
    time, failed = np.asarray(time, dtype=float), np.asarray(failed, dtype=bool)
    z = sps.norm.ppf(0.5 + confidence / 200)
    rows, survival, greenwood = [], 1.0, 0.0
    for t in np.unique(time[failed]):
        at_risk = int((time >= t).sum())
        d = int(((time == t) & failed).sum())
        survival *= 1 - d / at_risk
        if at_risk > d:
            greenwood += d / (at_risk * (at_risk - d))
        se = survival * np.sqrt(greenwood) if at_risk > d else 0.0
        rows.append({"Time": t, "Number at risk": at_risk, "Number failed": d, "Survival probability": survival,
                     "Standard error": se, "CI Lower": max(survival - z * se, 0.0),
                     "CI Upper": min(survival + z * se, 1.0)})
    return pd.DataFrame(rows)


def km_median(table: pd.DataFrame) -> float:
    below = table[table["Survival probability"] <= 0.5]
    return float(below["Time"].iloc[0]) if len(below) else float("nan")


def plotting_positions(time, failed) -> tuple[np.ndarray, np.ndarray]:
    """Failure times and their Herd-Johnson cumulative probabilities for probability plots."""
    time, failed = np.asarray(time, dtype=float), np.asarray(failed, dtype=bool)
    order = np.lexsort((~failed, time))   # failures before censored values at tied times
    time, failed = time[order], failed[order]
    n = len(time)
    reliability, times, probs = 1.0, [], []
    for i in range(n):
        if failed[i]:
            reverse_rank = n - i
            reliability *= reverse_rank / (reverse_rank + 1)
            times.append(time[i])
            probs.append(1 - reliability)
    return np.array(times), np.array(probs)
