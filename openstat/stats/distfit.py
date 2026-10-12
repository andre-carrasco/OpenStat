"""Fitting continuous distributions to complete data, with Anderson-Darling goodness of fit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from openstat.stats._util import anderson_darling_normal, clean


@dataclass
class Fit:
    name: str
    dist: object            # frozen scipy distribution
    params: dict            # reported parameters
    ad: float
    p_value: float          # nan when no p-value approximation is available


def _normal(x):
    return sps.norm(x.mean(), x.std(ddof=1)), {"Mean": x.mean(), "StDev": x.std(ddof=1)}


def _lognormal(x):
    logs = np.log(x)
    return sps.lognorm(logs.std(ddof=1), scale=np.exp(logs.mean())), {"Location": logs.mean(),
                                                                      "Scale": logs.std(ddof=1)}


def _exponential(x):
    return sps.expon(scale=x.mean()), {"Mean": x.mean()}


def _weibull(x):
    shape, _, scale = sps.weibull_min.fit(x, floc=0)
    return sps.weibull_min(shape, scale=scale), {"Shape": shape, "Scale": scale}


def _gamma(x):
    shape, _, scale = sps.gamma.fit(x, floc=0)
    return sps.gamma(shape, scale=scale), {"Shape": shape, "Scale": scale}


def _loglogistic(x):
    loc, scale = sps.logistic.fit(np.log(x))
    return sps.fisk(1 / scale, scale=np.exp(loc)), {"Location": loc, "Scale": scale}


def _logistic(x):
    loc, scale = sps.logistic.fit(x)
    return sps.logistic(loc, scale), {"Location": loc, "Scale": scale}


def _smallest_ev(x):
    loc, scale = sps.gumbel_l.fit(x)
    return sps.gumbel_l(loc, scale), {"Location": loc, "Scale": scale}


def _largest_ev(x):
    loc, scale = sps.gumbel_r.fit(x)
    return sps.gumbel_r(loc, scale), {"Location": loc, "Scale": scale}


# name -> (fitter, requires positive data)
FITTERS = {
    "Normal": (_normal, False),
    "Lognormal": (_lognormal, True),
    "Exponential": (_exponential, True),
    "Weibull": (_weibull, True),
    "Gamma": (_gamma, True),
    "Loglogistic": (_loglogistic, True),
    "Logistic": (_logistic, False),
    "Smallest extreme value": (_smallest_ev, False),
    "Largest extreme value": (_largest_ev, False),
}
NAMES = list(FITTERS)


def anderson_darling(x, dist) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    cdf = np.clip(dist.cdf(x), 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(cdf) + np.log1p(-cdf[::-1]))) / n)


def fit(values, name: str) -> Fit:
    x = clean(values)
    if len(x) < 3:
        raise ValueError("At least three observations are needed")
    fitter, positive = FITTERS[name]
    if positive and (x <= 0).any():
        raise ValueError(f"The {name} distribution needs positive data")
    dist, params = fitter(x)
    p = float("nan")
    if name == "Normal":
        _, p = anderson_darling_normal(x)
    elif name == "Lognormal":
        _, p = anderson_darling_normal(np.log(x))
    return Fit(name, dist, {k: float(v) for k, v in params.items()}, anderson_darling(x, dist), p)


def fit_all(values) -> list[Fit]:
    """All distributions that can be fitted to the data, best Anderson-Darling first."""
    fits = []
    for name in NAMES:
        try:
            fits.append(fit(values, name))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    return sorted(fits, key=lambda f: f.ad)
