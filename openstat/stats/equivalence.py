"""Equivalence tests by two one-sided t-tests (TOST)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.i18n import tr
from openstat.stats._tests import ALPHA
from openstat.stats._util import clean

MENU = ("Stat", "Equivalence Tests")
LOWER = Param("lower", "Lower equivalence limit", "number", default=-1.0)
UPPER = Param("upper", "Upper equivalence limit", "number", default=1.0)


def tost(estimate: float, se: float, df: float, lower: float, upper: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests of lower < estimate < upper, with the (1 - 2 alpha) interval."""
    if lower >= upper:
        raise ValueError("The lower limit must be less than the upper limit")
    dist = sps.t(df)
    t_lower = (estimate - lower) / se
    t_upper = (estimate - upper) / se
    p_lower, p_upper = float(dist.sf(t_lower)), float(dist.cdf(t_upper))
    half = dist.ppf(1 - alpha) * se
    return {"estimate": estimate, "se": se, "df": df, "ci": (estimate - half, estimate + half),
            "t_lower": t_lower, "p_lower": p_lower, "t_upper": t_upper, "p_upper": p_upper,
            "p": max(p_lower, p_upper), "equivalent": max(p_lower, p_upper) < alpha}


def _report(title: str, label: str, out: dict, lower: float, upper: float, alpha: float) -> Result:
    res = Result(title)
    res.text(f"{tr('Equivalence interval')}: ({lower:g}, {upper:g})    α = {alpha:g}")
    res.table(pd.DataFrame([{"Difference": out["estimate"], "SE": out["se"], "DF": out["df"],
                             "CI Lower": out["ci"][0], "CI Upper": out["ci"][1]}],
                           index=pd.Index([label], name="")), title=f"{100 * (1 - 2 * alpha):g}% CI")
    res.table(pd.DataFrame({"T-Value": [out["t_lower"], out["t_upper"]],
                            "P-Value": [out["p_lower"], out["p_upper"]]},
                           index=pd.Index([f"H0: Δ ≤ {lower:g}", f"H0: Δ ≥ {upper:g}"], name="Null hypothesis")),
              title="Test")
    res.text(tr("Can claim equivalence.") if out["equivalent"] else tr("Cannot claim equivalence."))
    return res


@analysis("equiv.one_sample", menu=MENU, title="Equivalence Test: 1-Sample",
          params=[Param("column", "Sample", "column"), Param("target", "Target", "number", default=0.0),
                  LOWER, UPPER, ALPHA])
def one_sample(ws, column, target=0.0, lower=-1.0, upper=1.0, alpha=0.05):
    x = clean(ws.df[column])
    n = len(x)
    if n < 2:
        raise ValueError("At least two observations are needed")
    out = tost(float(x.mean() - target), float(x.std(ddof=1) / np.sqrt(n)), n - 1, lower, upper, alpha)
    return _report("Equivalence Test: 1-Sample", f"{column} − {target:g}", out, lower, upper, alpha)


@analysis("equiv.two_sample", menu=MENU, title="Equivalence Test: 2-Sample",
          params=[Param("test", "Test sample", "column"), Param("reference", "Reference sample", "column"),
                  LOWER, UPPER, Param("pooled", "Assume equal variances", "bool", default=False), ALPHA])
def two_sample(ws, test, reference, lower=-1.0, upper=1.0, pooled=False, alpha=0.05):
    x1, x2 = clean(ws.df[test]), clean(ws.df[reference])
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample needs at least two observations")
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    if pooled:
        df = n1 + n2 - 2
        se = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df * (1 / n1 + 1 / n2))
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    out = tost(float(x1.mean() - x2.mean()), float(se), float(df), lower, upper, alpha)
    return _report("Equivalence Test: 2-Sample", f"{test} − {reference}", out, lower, upper, alpha)


@analysis("equiv.paired", menu=MENU, title="Equivalence Test: Paired",
          params=[Param("test", "Test sample", "column"), Param("reference", "Reference sample", "column"),
                  LOWER, UPPER, ALPHA])
def paired(ws, test, reference, lower=-1.0, upper=1.0, alpha=0.05):
    frame = ws.df[[test, reference]].dropna().astype(float)
    d = (frame[test] - frame[reference]).to_numpy()
    n = len(d)
    if n < 2:
        raise ValueError("At least two complete pairs are needed")
    out = tost(float(d.mean()), float(d.std(ddof=1) / np.sqrt(n)), n - 1, lower, upper, alpha)
    return _report("Equivalence Test: Paired", f"{test} − {reference}", out, lower, upper, alpha)


def crossover_effect(period1, period2, sequence) -> tuple[float, float, int]:
    """Treatment effect (first-sequence period-1 treatment minus the other) in a 2x2 crossover."""
    frame = pd.DataFrame({"p1": period1, "p2": period2, "seq": sequence}).dropna()
    labels = frame["seq"].map(format_value)
    levels = sorted(labels.unique())
    if len(levels) != 2:
        raise ValueError("The sequence column must have exactly two values")
    d = (frame["p1"].astype(float) - frame["p2"].astype(float)).to_numpy()
    d1, d2 = d[(labels == levels[0]).to_numpy()], d[(labels == levels[1]).to_numpy()]
    n1, n2 = len(d1), len(d2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sequence needs at least two subjects")
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * d1.var(ddof=1) + (n2 - 1) * d2.var(ddof=1)) / df
    return float((d1.mean() - d2.mean()) / 2), float(0.5 * np.sqrt(sp2 * (1 / n1 + 1 / n2))), df


@analysis("equiv.crossover", menu=MENU, title="Equivalence Test: 2x2 Crossover",
          params=[Param("period1", "Period 1 response", "column"), Param("period2", "Period 2 response", "column"),
                  Param("sequence", "Sequence", "column", numeric=False), LOWER, UPPER, ALPHA])
def crossover(ws, period1, period2, sequence, lower=-1.0, upper=1.0, alpha=0.05):
    effect, se, df = crossover_effect(ws.df[period1], ws.df[period2], ws.df[sequence])
    out = tost(effect, se, df, lower, upper, alpha)
    return _report("Equivalence Test: 2x2 Crossover", tr("Treatment effect"), out, lower, upper, alpha)
