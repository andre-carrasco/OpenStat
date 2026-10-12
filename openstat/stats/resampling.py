"""Resampling: bootstrap confidence intervals and randomization tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._tests import ALTERNATIVE, CONFIDENCE, GREATER, LESS, NOT_EQUAL
from openstat.stats._util import clean
from openstat.stats.basic import _frame, _symbol

MENU = ("Calc", "Resampling")
BLUE, RED = "#5b9bd5", "#c00000"
RESAMPLES = Param("resamples", "Number of resamples", "int", default=1000, minimum=100, maximum=1_000_000)
SEED = Param("seed", "Base for random numbers (0 = random)", "int", default=0, minimum=0)
CHUNK = 2_000_000   # values drawn at a time

STATISTICS = {
    "Mean": lambda x: x.mean(axis=1), "Median": lambda x: np.median(x, axis=1),
    "StDev": lambda x: x.std(axis=1, ddof=1), "Variance": lambda x: x.var(axis=1, ddof=1),
    "Sum": lambda x: x.sum(axis=1),
}


def _sample(x: np.ndarray, name: str, minimum: int = 2) -> np.ndarray:
    if len(x) < minimum:
        raise ValueError(f"{name}: not enough data")
    return x


def bootstrap(x: np.ndarray, statistic, resamples: int, rng) -> np.ndarray:
    """`statistic` of `resamples` samples drawn from x with replacement."""
    out, step = [], max(1, CHUNK // len(x))
    for start in range(0, resamples, step):
        rows = min(step, resamples - start)
        out.append(statistic(x[rng.integers(0, len(x), size=(rows, len(x)))]))
    return np.concatenate(out)


def permuted_differences(x1: np.ndarray, x2: np.ndarray, resamples: int, rng) -> np.ndarray:
    """Difference of the two sample means after shuffling the observations between the samples."""
    pooled = np.concatenate([x1, x2])
    out, step = [], max(1, CHUNK // len(pooled))
    for start in range(0, resamples, step):
        rows = min(step, resamples - start)
        shuffled = rng.permuted(np.tile(pooled, (rows, 1)), axis=1)
        out.append(shuffled[:, :len(x1)].mean(axis=1) - shuffled[:, len(x1):].mean(axis=1))
    return np.concatenate(out)


def percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence / 100) / 2
    lo, hi = np.quantile(values, [tail, 1 - tail])
    return float(lo), float(hi)


def tail_proportion(values: np.ndarray, observed: float, center: float, alternative: str) -> float:
    """Share of the resampled statistics at least as extreme as the observed one."""
    if alternative == GREATER:
        return float((values >= observed).mean())
    if alternative == LESS:
        return float((values <= observed).mean())
    return float((np.abs(values - center) >= abs(observed - center)).mean())


def _histogram(values, marks, title: str, xlabel: str):
    fig = new_figure()
    ax = fig.add_subplot()
    ax.hist(values, bins=40, color=BLUE, edgecolor="white")
    for mark in marks:
        ax.axvline(mark, color=RED, linestyle="--")
    ax.set_xlabel(xlabel), ax.set_ylabel(tr("Frequency")), ax.set_title(title)
    return fig


def _rng(seed: int):
    return np.random.default_rng(seed or None)


@analysis("resample.bootstrap1", menu=MENU, title="Bootstrapping for 1-Sample Function",
          params=[Param("columns", "Variables", "columns"),
                  Param("statistic", "Statistic", "choice", default="Mean", choices=list(STATISTICS)),
                  RESAMPLES, CONFIDENCE, SEED])
def bootstrap_one_sample(ws, columns, statistic="Mean", resamples=1000, confidence=95.0, seed=0):
    rng = _rng(seed)
    res = Result("Bootstrapping for 1-Sample Function")
    rows, figures = {}, []
    for name in columns:
        x = _sample(clean(ws.df[name]), name)
        values = bootstrap(x, STATISTICS[statistic], resamples, rng)
        lo, hi = percentile_interval(values, confidence)
        rows[name] = {"N": len(x), statistic: float(STATISTICS[statistic](x[None, :])[0]),
                      "Bootstrap average": float(values.mean()), "Bootstrap StDev": float(values.std(ddof=1)),
                      "CI Lower": lo, "CI Upper": hi}
        figures.append(_histogram(values, (lo, hi), f"{tr('Bootstrap distribution')}: {name}", tr(statistic)))
    res.text(f"{tr('Number of resamples')}: {resamples}    {confidence:g}% {tr('percentile interval')}")
    res.table(_frame(rows))
    for fig in figures:
        res.figure(fig)
    return res


@analysis("resample.bootstrap2", menu=MENU, title="Bootstrapping for 2-Sample Means",
          params=[Param("sample1", "Sample 1", "column"), Param("sample2", "Sample 2", "column"),
                  RESAMPLES, CONFIDENCE, SEED])
def bootstrap_two_sample(ws, sample1, sample2, resamples=1000, confidence=95.0, seed=0):
    rng = _rng(seed)
    x1, x2 = _sample(clean(ws.df[sample1]), sample1), _sample(clean(ws.df[sample2]), sample2)
    values = bootstrap(x1, STATISTICS["Mean"], resamples, rng) - bootstrap(x2, STATISTICS["Mean"], resamples, rng)
    lo, hi = percentile_interval(values, confidence)
    res = Result("Bootstrapping for 2-Sample Means")
    res.text(f"{tr('Number of resamples')}: {resamples}    {confidence:g}% {tr('percentile interval')}")
    res.table(pd.DataFrame({"N": [len(x1), len(x2)], "Mean": [x1.mean(), x2.mean()],
                            "StDev": [x1.std(ddof=1), x2.std(ddof=1)]},
                           index=pd.Index([sample1, sample2], name="Sample")))
    res.table(pd.DataFrame([{"Difference": float(x1.mean() - x2.mean()), "Bootstrap average": float(values.mean()),
                             "Bootstrap StDev": float(values.std(ddof=1)), "CI Lower": lo, "CI Upper": hi}]),
              title="Difference of means", index=False)
    return res.figure(_histogram(values, (lo, hi), tr("Bootstrap distribution"), tr("Difference of means")))


@analysis("resample.random1", menu=MENU, title="Randomization Test for 1-Sample Mean",
          params=[Param("columns", "Variables", "columns"),
                  Param("mu0", "Hypothesized mean", "number", default=0.0), ALTERNATIVE, RESAMPLES, SEED])
def randomization_one_sample(ws, columns, mu0=0.0, alternative=NOT_EQUAL, resamples=1000, seed=0):
    rng = _rng(seed)
    res = Result("Randomization Test for 1-Sample Mean")
    rows, figures = {}, []
    for name in columns:
        x = _sample(clean(ws.df[name]), name)
        # resampling the data shifted to the hypothesized mean gives the distribution under H0
        values = bootstrap(x - x.mean() + mu0, STATISTICS["Mean"], resamples, rng)
        rows[name] = {"N": len(x), "Mean": float(x.mean()), "StDev": float(x.std(ddof=1)),
                      "Average of resamples": float(values.mean()),
                      "P-Value": tail_proportion(values, x.mean(), mu0, alternative)}
        figures.append(_histogram(values, (x.mean(),), f"{tr('Randomization distribution')}: {name}", tr("Mean")))
    res.text(f"H0: μ = {mu0:g}    H1: μ {_symbol(alternative)} {mu0:g}    {tr('Number of resamples')}: {resamples}")
    res.table(_frame(rows))
    for fig in figures:
        res.figure(fig)
    return res


@analysis("resample.random2", menu=MENU, title="Randomization Test for 2-Sample Means",
          params=[Param("sample1", "Sample 1", "column"), Param("sample2", "Sample 2", "column"),
                  ALTERNATIVE, RESAMPLES, SEED])
def randomization_two_sample(ws, sample1, sample2, alternative=NOT_EQUAL, resamples=1000, seed=0):
    x1, x2 = _sample(clean(ws.df[sample1]), sample1, 1), _sample(clean(ws.df[sample2]), sample2, 1)
    observed = float(x1.mean() - x2.mean())
    values = permuted_differences(x1, x2, resamples, _rng(seed))
    res = Result("Randomization Test for 2-Sample Means")
    res.text(f"H0: μ1 − μ2 = 0    H1: μ1 − μ2 {_symbol(alternative)} 0    {tr('Number of resamples')}: {resamples}")
    res.table(pd.DataFrame({"N": [len(x1), len(x2)], "Mean": [x1.mean(), x2.mean()]},
                           index=pd.Index([sample1, sample2], name="Sample")))
    res.table(pd.DataFrame([{"Difference": observed, "Average of resamples": float(values.mean()),
                             "P-Value": tail_proportion(values, observed, 0.0, alternative)}]),
              title="Test", index=False)
    return res.figure(_histogram(values, (observed,), tr("Randomization distribution"), tr("Difference of means")))
