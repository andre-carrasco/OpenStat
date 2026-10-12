"""Nonparametric tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.i18n import tr
from openstat.stats._tests import ALTERNATIVE, CONFIDENCE, NOT_EQUAL, SCIPY_ALT
from openstat.stats._util import clean
from openstat.stats.basic import _frame, _symbol, median_ci

MENU = ("Stat", "Nonparametrics")
VARIABLES = Param("columns", "Variables", "columns")
TEST_MEDIAN = Param("median0", "Hypothesized median", "number", default=0.0)
RESPONSE = Param("response", "Response", "column")
FACTOR = Param("factor", "Factor", "column", numeric=False)


@analysis("nonpar.sign", menu=MENU, title="1-Sample Sign",
          params=[VARIABLES, TEST_MEDIAN, CONFIDENCE, ALTERNATIVE])
def sign_test(ws, columns, median0=0.0, confidence=95.0, alternative=NOT_EQUAL):
    rows = {}
    for name in columns:
        x = clean(ws.df[name])
        below, equal, above = int((x < median0).sum()), int((x == median0).sum()), int((x > median0).sum())
        n = below + above
        p = float(sps.binomtest(above, n, 0.5, alternative=SCIPY_ALT[alternative]).pvalue) if n else np.nan
        lo, hi = median_ci(x, confidence)
        rows[name] = {"N": len(x), "Below": below, "Equal": equal, "Above": above, "Median": float(np.median(x)),
                      "CI Lower": lo, "CI Upper": hi, "P-Value": p}
    res = Result("1-Sample Sign")
    res.text(f"H0: η = {median0:g}    H1: η {_symbol(alternative)} {median0:g}    ≥{confidence:g}% CI")
    return res.table(_frame(rows))


def walsh_averages(x: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(x))
    return (x[i] + x[j]) / 2


@analysis("nonpar.wilcoxon", menu=MENU, title="1-Sample Wilcoxon", params=[VARIABLES, TEST_MEDIAN, ALTERNATIVE])
def wilcoxon(ws, columns, median0=0.0, alternative=NOT_EQUAL):
    rows = {}
    for name in columns:
        x = clean(ws.df[name])
        d = x - median0
        d = d[d != 0]
        if len(d) < 1:
            raise ValueError(f"{name}: all values equal the hypothesized median")
        ranks = sps.rankdata(np.abs(d))
        w = float(ranks[d > 0].sum())
        p = float(sps.wilcoxon(d, alternative=SCIPY_ALT[alternative], correction=True,
                               method="exact" if len(d) <= 25 and len(np.unique(np.abs(d))) == len(d) else "approx").pvalue)
        rows[name] = {"N": len(x), "N for test": len(d), "Wilcoxon statistic": w,
                      "Estimated median": float(np.median(walsh_averages(x))), "P-Value": p}
    res = Result("1-Sample Wilcoxon")
    res.text(f"H0: η = {median0:g}    H1: η {_symbol(alternative)} {median0:g}")
    return res.table(_frame(rows))


@analysis("nonpar.mann_whitney", menu=MENU, title="Mann-Whitney",
          params=[Param("sample1", "Sample 1", "column"), Param("sample2", "Sample 2", "column"), ALTERNATIVE])
def mann_whitney(ws, sample1, sample2, alternative=NOT_EQUAL):
    x1, x2 = clean(ws.df[sample1]), clean(ws.df[sample2])
    if not len(x1) or not len(x2):
        raise ValueError("Both samples need data")
    test = sps.mannwhitneyu(x1, x2, alternative=SCIPY_ALT[alternative], method="asymptotic", use_continuity=True)
    w = float(sps.rankdata(np.concatenate([x1, x2]))[:len(x1)].sum())
    shift = float(np.median(np.subtract.outer(x1, x2)))
    res = Result("Mann-Whitney")
    res.text(f"H0: η1 − η2 = 0    H1: η1 − η2 {_symbol(alternative)} 0")
    res.table(pd.DataFrame({"N": [len(x1), len(x2)], "Median": [np.median(x1), np.median(x2)]},
                           index=pd.Index([sample1, sample2], name="Sample")))
    return res.table(pd.DataFrame([{"Difference": shift, "W-Value": w, "U-Value": float(test.statistic),
                                    "P-Value": float(test.pvalue)}]), title="Test", index=False)


def _by_factor(ws, response, factor):
    frame = ws.df[[response, factor]].dropna()
    labels = frame[factor].map(format_value)
    levels = sorted(labels.unique())
    if len(levels) < 2:
        raise ValueError("The factor needs at least two levels")
    return frame[response].to_numpy(dtype=float), labels.to_numpy(), levels


@analysis("nonpar.kruskal", menu=MENU, title="Kruskal-Wallis", params=[RESPONSE, FACTOR])
def kruskal_wallis(ws, response, factor):
    y, labels, levels = _by_factor(ws, response, factor)
    n = len(y)
    ranks = sps.rankdata(y)
    rows = {}
    for level in levels:
        mask = labels == level
        nj = int(mask.sum())
        mean_rank = ranks[mask].mean()
        z = (mean_rank - (n + 1) / 2) / np.sqrt((n + 1) * (n / nj - 1) / 12)
        rows[level] = {"N": nj, "Median": float(np.median(y[mask])), "Mean rank": float(mean_rank), "Z-Value": z}
    h_adj, p_adj = sps.kruskal(*[y[labels == level] for level in levels])
    _, counts = np.unique(y, return_counts=True)
    correction = 1 - (counts ** 3 - counts).sum() / (n ** 3 - n)
    h = h_adj * correction
    df = len(levels) - 1
    res = Result("Kruskal-Wallis")
    res.table(_frame(rows, factor))
    return res.table(pd.DataFrame({"DF": [df, df], "H-Value": [h, h_adj],
                                   "P-Value": [sps.chi2.sf(h, df), p_adj]},
                                  index=pd.Index(["Not adjusted for ties", "Adjusted for ties"], name="Method")),
                     title="Test")


@analysis("nonpar.mood", menu=MENU, title="Mood's Median Test", params=[RESPONSE, FACTOR])
def mood_median(ws, response, factor):
    y, labels, levels = _by_factor(ws, response, factor)
    grand = float(np.median(y))
    rows = {level: {"Median": float(np.median(y[labels == level])),
                    "N <= overall median": int((y[labels == level] <= grand).sum()),
                    "N > overall median": int((y[labels == level] > grand).sum())} for level in levels}
    table = np.array([[r["N > overall median"] for r in rows.values()],
                      [r["N <= overall median"] for r in rows.values()]])
    if (table.sum(axis=1) == 0).any():
        raise ValueError("All observations fall on one side of the overall median")
    chi2, p, df, _ = sps.chi2_contingency(table, correction=False)
    res = Result("Mood's Median Test")
    res.text(f"{tr('Overall median')} = {grand:g}")
    res.table(_frame(rows, factor))
    return res.table(pd.DataFrame([{"DF": df, "Chi-Square": float(chi2), "P-Value": float(p)}]),
                     title="Test", index=False)


@analysis("nonpar.friedman", menu=MENU, title="Friedman",
          params=[RESPONSE, Param("treatment", "Treatment", "column", numeric=False),
                  Param("block", "Blocks", "column", numeric=False)])
def friedman(ws, response, treatment, block):
    frame = ws.df[[response, treatment, block]].dropna()
    wide = frame.pivot_table(index=frame[block].map(format_value), columns=frame[treatment].map(format_value),
                             values=response, aggfunc="mean").dropna()
    b, k = wide.shape
    if b < 2 or k < 3:
        raise ValueError("Friedman's test needs at least three treatments and two complete blocks")
    stat, p = sps.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    ranks = wide.rank(axis=1)
    res = Result("Friedman")
    res.table(pd.DataFrame({"N": b, "Median": wide.median(), "Sum of ranks": ranks.sum()},
                           index=pd.Index(wide.columns, name=treatment)))
    return res.table(pd.DataFrame([{"DF": k - 1, "Chi-Square": float(stat), "P-Value": float(p)}]),
                     title="Test (adjusted for ties)", index=False)


@analysis("nonpar.runs", menu=MENU, title="Runs Test",
          params=[VARIABLES, Param("use_k", "Use the value below instead of the mean", "bool", default=False),
                  Param("k", "Comparison value", "number", default=0.0)])
def runs_test(ws, columns, use_k=False, k=0.0):
    rows = {}
    for name in columns:
        x = clean(ws.df[name])
        centre = k if use_k else float(x.mean())
        above = x > centre
        a, b = int(above.sum()), int((~above).sum())
        runs = int(1 + (above[1:] != above[:-1]).sum()) if len(x) else 0
        n = a + b
        if a == 0 or b == 0 or n < 2:
            rows[name] = {"N": n, "K": centre, "Observed runs": runs, "Expected runs": np.nan,
                          "Above K": a, "At or below K": b, "P-Value": np.nan}
            continue
        expected = 2 * a * b / n + 1
        variance = 2 * a * b * (2 * a * b - n) / (n ** 2 * (n - 1))
        z = (runs - expected) / np.sqrt(variance) if variance > 0 else np.nan
        rows[name] = {"N": n, "K": centre, "Observed runs": runs, "Expected runs": expected, "Above K": a,
                      "At or below K": b, "P-Value": float(2 * sps.norm.sf(abs(z)))}
    res = Result("Runs Test")
    res.text(tr("H0: the order of the data is random"))
    return res.table(_frame(rows))

