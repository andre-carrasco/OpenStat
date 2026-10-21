"""Basic statistics: descriptive summaries and one- and two-sample inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._tests import (ALTERNATIVE, CONFIDENCE, GREATER, LESS, NOT_EQUAL, SCIPY_ALT, bounds, p_value,
                                   tail_levels)
from openstat.stats._util import anderson_darling_normal, by_levels, clean, mean_ci, quartiles

MENU = ("Stat", "Basic Statistics")
VARIABLES = Param("columns", "Variables", "columns")
SAMPLE1 = Param("sample1", "Sample 1", "column")
SAMPLE2 = Param("sample2", "Sample 2", "column")
TEST_MEAN = Param("mu0", "Hypothesized mean", "number", default=0.0)


def describe_values(values) -> dict:
    """Summary statistics of one sample. Quartiles use the (n+1)p position rule."""
    x = np.asarray(values, dtype=float)
    missing = int(np.isnan(x).sum())
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        nan = float("nan")
        return {"N": 0, "N*": missing, "Mean": nan, "SE Mean": nan, "StDev": nan, "Minimum": nan,
                "Q1": nan, "Median": nan, "Q3": nan, "Maximum": nan}
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    q1, med, q3 = quartiles(x)
    return {"N": n, "N*": missing, "Mean": float(x.mean()), "SE Mean": sd / np.sqrt(n), "StDev": sd,
            "Minimum": float(x.min()), "Q1": float(q1), "Median": float(med), "Q3": float(q3),
            "Maximum": float(x.max())}


def more_values(values) -> dict:
    x = clean(values)
    n = len(x)
    nan = float("nan")
    if n < 2:
        return {"Variance": nan, "CoefVar": nan, "Range": nan, "IQR": nan, "Sum": float(x.sum()),
                "Skewness": nan, "Kurtosis": nan}
    q1, _, q3 = quartiles(x)
    mean = x.mean()
    return {"Variance": float(x.var(ddof=1)),
            "CoefVar": float(100 * x.std(ddof=1) / mean) if mean else nan,
            "Range": float(x.max() - x.min()), "IQR": float(q3 - q1), "Sum": float(x.sum()),
            "Skewness": float(sps.skew(x, bias=False)) if n > 2 else nan,
            "Kurtosis": float(sps.kurtosis(x, bias=False)) if n > 3 else nan}


@analysis("basic.describe", menu=MENU, title="Descriptive Statistics",
          params=[VARIABLES,
                  Param("by", "By variable", "column", numeric=False, optional=True),
                  Param("more", "Additional statistics", "bool", default=False)])
def describe(ws, columns, by=None, more=False):
    res = Result("Descriptive Statistics")
    rows, extra, labels = [], [], []
    for name in columns:
        parts = [(name, ws.df[name])]
        if by:
            parts = [(f"{name} ({by} = {format_value(level)})", frame[name]) for level, frame in by_levels(ws, by)]
        for label, values in parts:
            rows.append(describe_values(values))
            extra.append(more_values(values))
            labels.append(label)
    index = pd.Index(labels, name="Variable")
    res.table(pd.DataFrame(rows, index=index))
    if more:
        res.table(pd.DataFrame(extra, index=index))
    return res


@analysis("basic.store_describe", menu=MENU, title="Store Descriptive Statistics",
          params=[VARIABLES, Param("by", "By variable", "column", numeric=False, optional=True)])
def store_describe(ws, columns, by=None):
    rows = []
    for name in columns:
        groups = [(v, frame[name]) for v, frame in by_levels(ws, by)] if by else [(None, ws.df[name])]
        for level, values in groups:
            row = {"Variable": name}
            if by:
                row["By" if by == "Variable" else by] = level
            rows.append({**row, **describe_values(values), **more_values(values)})
    frame = pd.DataFrame(rows)
    res = Result("Store Descriptive Statistics")
    res.worksheet(f"{ws.name} ({tr('statistics')})", frame)
    res.text(f"{tr('New worksheet')}: {len(frame)} x {frame.shape[1]}")
    return res


def median_ci(values, confidence: float = 95.0) -> tuple[float, float]:
    """Distribution-free interval for the median from order statistics (coverage >= nominal)."""
    x = np.sort(clean(values))
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    alpha = 1 - confidence / 100
    k = int(sps.binom.ppf(alpha / 2, n, 0.5))
    if sps.binom.cdf(k, n, 0.5) > alpha / 2:
        k -= 1
    k = max(k, 0)
    return float(x[k]), float(x[n - k - 1])


@analysis("basic.graphical_summary", menu=MENU, title="Graphical Summary", params=[VARIABLES, CONFIDENCE])
def graphical_summary(ws, columns, confidence=95.0):
    from openstat.graphs.basic import BLUE, RED, box_stats

    res = Result("Graphical Summary")
    for name in columns:
        x = clean(ws.df[name])
        n = len(x)
        if n < 2:
            continue
        d, m = describe_values(x), more_values(x)
        ad, p = anderson_darling_normal(x)
        mean, lo, hi = mean_ci(x, confidence / 100)
        med_lo, med_hi = median_ci(x, confidence)
        alpha = 1 - confidence / 100
        sd = d["StDev"]
        sd_lo = sd * np.sqrt((n - 1) / sps.chi2.ppf(1 - alpha / 2, n - 1))
        sd_hi = sd * np.sqrt((n - 1) / sps.chi2.ppf(alpha / 2, n - 1))

        fig = new_figure(6.5, 4.5)
        top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[4, 1])
        top.hist(x, bins="auto", density=True, color=BLUE, edgecolor="white")
        if sd > 0:
            grid = np.linspace(x.min() - sd, x.max() + sd, 200)
            top.plot(grid, sps.norm.pdf(grid, mean, sd), color=RED)
        top.set_title(f"{tr('Graphical Summary')}: {name}")
        bottom.bxp([box_stats(x, "")], orientation="horizontal", patch_artist=True,
                   boxprops={"facecolor": BLUE}, medianprops={"color": "black"})
        bottom.set_yticks([])
        res.figure(fig)

        res.heading(name)
        res.table(pd.DataFrame({"Value": {
            "Anderson-Darling A²": ad, "P-Value": p, "Mean": mean, "StDev": sd, "Variance": m["Variance"],
            "Skewness": m["Skewness"], "Kurtosis": m["Kurtosis"], "N": n, "Minimum": d["Minimum"],
            "Q1": d["Q1"], "Median": d["Median"], "Q3": d["Q3"], "Maximum": d["Maximum"]}}))
        res.table(pd.DataFrame({"Lower": [lo, med_lo, sd_lo], "Upper": [hi, med_hi, sd_hi]},
                               index=pd.Index(["Mean", "Median", "StDev"], name=f"{confidence:g}% CI")))
    return res


# ----- means -----

def _one_sample(x, se_dist, se, mu0, confidence, alternative):
    mean = float(np.mean(x))
    stat = (mean - mu0) / se if se > 0 else float("nan")
    lo, hi = bounds(mean, se, se_dist, confidence, alternative)
    return mean, stat, p_value(stat, se_dist, alternative), lo, hi


@analysis("basic.z1", menu=MENU, title="1-Sample Z",
          params=[VARIABLES, Param("sigma", "Known standard deviation", "number", default=1.0, minimum=1e-12),
                  TEST_MEAN, CONFIDENCE, ALTERNATIVE])
def one_sample_z(ws, columns, sigma=1.0, mu0=0.0, confidence=95.0, alternative=NOT_EQUAL):
    rows = {}
    for name in columns:
        x = clean(ws.df[name])
        se = sigma / np.sqrt(len(x))
        mean, z, p, lo, hi = _one_sample(x, sps.norm(), se, mu0, confidence, alternative)
        rows[name] = {"N": len(x), "Mean": mean, "StDev": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                      "SE Mean": se, "CI Lower": lo, "CI Upper": hi, "Z-Value": z, "P-Value": p}
    res = Result("1-Sample Z")
    res.text(f"H0: μ = {mu0:g}    H1: μ {_symbol(alternative)} {mu0:g}    σ = {sigma:g}    {confidence:g}% CI")
    return res.table(_frame(rows))


@analysis("basic.t1", menu=MENU, title="1-Sample t", params=[VARIABLES, TEST_MEAN, CONFIDENCE, ALTERNATIVE])
def one_sample_t(ws, columns, mu0=0.0, confidence=95.0, alternative=NOT_EQUAL):
    rows = {}
    for name in columns:
        x = clean(ws.df[name])
        n = len(x)
        if n < 2:
            raise ValueError(f"{name}: at least two observations are needed")
        sd = float(np.std(x, ddof=1))
        se = sd / np.sqrt(n)
        mean, t, p, lo, hi = _one_sample(x, sps.t(n - 1), se, mu0, confidence, alternative)
        rows[name] = {"N": n, "Mean": mean, "StDev": sd, "SE Mean": se, "CI Lower": lo, "CI Upper": hi,
                      "T-Value": t, "P-Value": p}
    res = Result("1-Sample t")
    res.text(f"H0: μ = {mu0:g}    H1: μ {_symbol(alternative)} {mu0:g}    {confidence:g}% CI")
    return res.table(_frame(rows))


def _symbol(alternative: str) -> str:
    return {NOT_EQUAL: "≠", LESS: "<", GREATER: ">"}[alternative]


def _frame(rows: dict, index_name: str = "Variable") -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = index_name
    return frame


def two_sample_t_values(x1, x2, pooled=False, delta0=0.0, confidence=95.0, alternative=NOT_EQUAL) -> dict:
    x1, x2 = clean(x1), clean(x2)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample needs at least two observations")
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    diff = float(x1.mean() - x2.mean())
    if pooled:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    dist = sps.t(df)
    t = (diff - delta0) / se
    lo, hi = bounds(diff, se, dist, confidence, alternative)
    return {"Difference": diff, "SE": float(se), "CI Lower": lo, "CI Upper": hi, "T-Value": float(t),
            "DF": float(df), "P-Value": p_value(t, dist, alternative)}


@analysis("basic.t2", menu=MENU, title="2-Sample t",
          params=[SAMPLE1, SAMPLE2,
                  Param("pooled", "Assume equal variances", "bool", default=False),
                  Param("delta0", "Hypothesized difference", "number", default=0.0), CONFIDENCE, ALTERNATIVE])
def two_sample_t(ws, sample1, sample2, pooled=False, delta0=0.0, confidence=95.0, alternative=NOT_EQUAL):
    res = Result("2-Sample t")
    res.text(f"H0: μ1 − μ2 = {delta0:g}    H1: μ1 − μ2 {_symbol(alternative)} {delta0:g}    {confidence:g}% CI")
    res.table(pd.DataFrame([describe_values(ws.df[c]) for c in (sample1, sample2)],
                           index=pd.Index([sample1, sample2], name="Sample"))[["N", "Mean", "StDev", "SE Mean"]])
    values = two_sample_t_values(ws.df[sample1], ws.df[sample2], pooled, delta0, confidence, alternative)
    return res.table(pd.DataFrame([values], index=pd.Index([f"{sample1} − {sample2}"], name="Difference")),
                     title="Test")


@analysis("basic.paired_t", menu=MENU, title="Paired t",
          params=[SAMPLE1, SAMPLE2, Param("delta0", "Hypothesized difference", "number", default=0.0),
                  CONFIDENCE, ALTERNATIVE])
def paired_t(ws, sample1, sample2, delta0=0.0, confidence=95.0, alternative=NOT_EQUAL):
    frame = ws.df[[sample1, sample2]].dropna().astype(float)
    d = (frame[sample1] - frame[sample2]).to_numpy()
    n = len(d)
    if n < 2:
        raise ValueError("At least two complete pairs are needed")
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    mean, t, p, lo, hi = _one_sample(d, sps.t(n - 1), se, delta0, confidence, alternative)
    res = Result("Paired t")
    res.text(f"H0: μd = {delta0:g}    H1: μd {_symbol(alternative)} {delta0:g}    {confidence:g}% CI")
    rows = {c: {k: describe_values(frame[c])[k] for k in ("N", "Mean", "StDev", "SE Mean")}
            for c in (sample1, sample2)}
    rows["Difference"] = {"N": n, "Mean": mean, "StDev": sd, "SE Mean": se}
    res.table(_frame(rows, "Sample"))
    return res.table(pd.DataFrame([{"CI Lower": lo, "CI Upper": hi, "T-Value": t, "P-Value": p}],
                                  index=pd.Index(["Difference"], name="")), title="Test", index=False)


# ----- proportions and rates -----

def _events(ws, column, events, trials) -> tuple[int, int]:
    """Events and trials from a two-level column (event = highest level) or from summarized counts."""
    if column:
        col = ws.df[column].dropna()
        levels = sorted(col.unique())
        if len(levels) > 2:
            raise ValueError(f"{column}: the column must have two distinct values")
        return int((col == levels[-1]).sum()), len(col)
    if trials < 1 or not 0 <= events <= trials:
        raise ValueError("Events must be between 0 and the number of trials")
    return int(events), int(trials)


def clopper_pearson(k: int, n: int, confidence=95.0, alternative=NOT_EQUAL):
    lo_tail, hi_tail = tail_levels(confidence, alternative)
    lo = float("nan") if lo_tail is None else (0.0 if k == 0 else float(sps.beta.ppf(lo_tail, k, n - k + 1)))
    hi = float("nan") if hi_tail is None else (1.0 if k == n else float(sps.beta.ppf(hi_tail, k + 1, n - k)))
    return lo, hi


PROPORTION_METHODS = ["Exact", "Normal approximation"]


@analysis("basic.p1", menu=MENU, title="1 Proportion",
          params=[Param("column", "Sample column (two values)", "column", numeric=False, optional=True),
                  Param("events", "Number of events", "int", default=0, minimum=0),
                  Param("trials", "Number of trials", "int", default=0, minimum=0),
                  Param("p0", "Hypothesized proportion", "number", default=0.5, minimum=0.0001, maximum=0.9999),
                  Param("method", "Method", "choice", default="Exact", choices=PROPORTION_METHODS),
                  CONFIDENCE, ALTERNATIVE])
def one_proportion(ws, column=None, events=0, trials=0, p0=0.5, method="Exact", confidence=95.0,
                   alternative=NOT_EQUAL):
    k, n = _events(ws, column, events, trials)
    p_hat = k / n
    row = {"X": k, "N": n, "Sample p": p_hat}
    if method == "Exact":
        lo, hi = clopper_pearson(k, n, confidence, alternative)
        p = float(sps.binomtest(k, n, p0, alternative=SCIPY_ALT[alternative]).pvalue)
        row.update({"CI Lower": lo, "CI Upper": hi, "P-Value": p})
    else:
        lo, hi = bounds(p_hat, np.sqrt(p_hat * (1 - p_hat) / n), sps.norm(), confidence, alternative)
        z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
        row.update({"CI Lower": lo, "CI Upper": hi, "Z-Value": z, "P-Value": p_value(z, sps.norm(), alternative)})
    res = Result("1 Proportion")
    res.text(f"H0: p = {p0:g}    H1: p {_symbol(alternative)} {p0:g}    {tr(method)}    {confidence:g}% CI")
    return res.table(pd.DataFrame([row]), index=False, digits=6)


@analysis("basic.p2", menu=MENU, title="2 Proportions",
          params=[Param("events1", "Events in sample 1", "int", default=0, minimum=0),
                  Param("trials1", "Trials in sample 1", "int", default=0, minimum=0),
                  Param("events2", "Events in sample 2", "int", default=0, minimum=0),
                  Param("trials2", "Trials in sample 2", "int", default=0, minimum=0),
                  CONFIDENCE, ALTERNATIVE])
def two_proportions(ws, events1, trials1, events2, trials2, confidence=95.0, alternative=NOT_EQUAL):
    k1, n1 = _events(ws, None, events1, trials1)
    k2, n2 = _events(ws, None, events2, trials2)
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    lo, hi = bounds(diff, np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2), sps.norm(), confidence, alternative)
    pooled = (k1 + k2) / (n1 + n2)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = diff / se0 if se0 > 0 else float("nan")
    fisher = sps.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative=SCIPY_ALT[alternative]).pvalue
    res = Result("2 Proportions")
    res.text(f"H0: p1 − p2 = 0    H1: p1 − p2 {_symbol(alternative)} 0    {confidence:g}% CI")
    res.table(pd.DataFrame({"X": [k1, k2], "N": [n1, n2], "Sample p": [p1, p2]},
                           index=pd.Index(["1", "2"], name="Sample")), digits=6)
    return res.table(pd.DataFrame([{"Difference": diff, "CI Lower": lo, "CI Upper": hi, "Z-Value": z,
                                    "P-Value": p_value(z, sps.norm(), alternative),
                                    "Fisher's exact P-Value": float(fisher)}]), title="Test", index=False, digits=6)


def poisson_exact_p(total: int, expected: float, alternative: str) -> float:
    if alternative == LESS:
        return float(sps.poisson.cdf(total, expected))
    if alternative == GREATER:
        return float(sps.poisson.sf(total - 1, expected))
    upper = int(max(total, expected) * 4 + 50)
    pmf = sps.poisson.pmf(np.arange(upper + 1), expected)
    return float(min(1.0, pmf[pmf <= sps.poisson.pmf(total, expected) * (1 + 1e-7)].sum()))


def poisson_rate_ci(total: int, n: float, confidence=95.0, alternative=NOT_EQUAL):
    lo_tail, hi_tail = tail_levels(confidence, alternative)
    lo = float("nan") if lo_tail is None else (0.0 if total == 0 else sps.chi2.ppf(lo_tail, 2 * total) / (2 * n))
    hi = float("nan") if hi_tail is None else sps.chi2.ppf(hi_tail, 2 * total + 2) / (2 * n)
    return float(lo), float(hi)


@analysis("basic.poisson1", menu=MENU, title="1-Sample Poisson Rate",
          params=[Param("column", "Counts", "column"),
                  Param("rate0", "Hypothesized rate", "number", default=1.0, minimum=1e-12),
                  CONFIDENCE, ALTERNATIVE])
def one_sample_poisson(ws, column, rate0=1.0, confidence=95.0, alternative=NOT_EQUAL):
    x = clean(ws.df[column])
    if (x < 0).any() or (x != np.round(x)).any():
        raise ValueError("Counts must be non-negative integers")
    total, n = int(x.sum()), len(x)
    lo, hi = poisson_rate_ci(total, n, confidence, alternative)
    res = Result("1-Sample Poisson Rate")
    res.text(f"H0: λ = {rate0:g}    H1: λ {_symbol(alternative)} {rate0:g}    {confidence:g}% CI")
    return res.table(pd.DataFrame([{"Total occurrences": total, "N": n, "Rate": total / n, "CI Lower": lo,
                                    "CI Upper": hi, "P-Value": poisson_exact_p(total, rate0 * n, alternative)}],
                                  index=pd.Index([column], name="Variable")))


@analysis("basic.poisson2", menu=MENU, title="2-Sample Poisson Rate",
          params=[SAMPLE1, SAMPLE2, CONFIDENCE, ALTERNATIVE])
def two_sample_poisson(ws, sample1, sample2, confidence=95.0, alternative=NOT_EQUAL):
    x1, x2 = clean(ws.df[sample1]), clean(ws.df[sample2])
    t1, t2, n1, n2 = x1.sum(), x2.sum(), len(x1), len(x2)
    r1, r2 = t1 / n1, t2 / n2
    diff = r1 - r2
    lo, hi = bounds(diff, np.sqrt(r1 / n1 + r2 / n2), sps.norm(), confidence, alternative)
    pooled = (t1 + t2) / (n1 + n2)
    se0 = np.sqrt(pooled * (1 / n1 + 1 / n2))
    z = diff / se0 if se0 > 0 else float("nan")
    res = Result("2-Sample Poisson Rate")
    res.text(f"H0: λ1 − λ2 = 0    H1: λ1 − λ2 {_symbol(alternative)} 0    {confidence:g}% CI")
    res.table(pd.DataFrame({"Total occurrences": [t1, t2], "N": [n1, n2], "Rate": [r1, r2]},
                           index=pd.Index([sample1, sample2], name="Sample")))
    return res.table(pd.DataFrame([{"Difference": diff, "CI Lower": lo, "CI Upper": hi, "Z-Value": z,
                                    "P-Value": p_value(z, sps.norm(), alternative)}]), title="Test", index=False)


# ----- variances -----

@analysis("basic.var1", menu=MENU, title="1 Variance",
          params=[VARIABLES, Param("sigma0", "Hypothesized standard deviation", "number", default=1.0, minimum=1e-12),
                  CONFIDENCE, ALTERNATIVE])
def one_variance(ws, columns, sigma0=1.0, confidence=95.0, alternative=NOT_EQUAL):
    rows = {}
    for name in columns:
        x = clean(ws.df[name])
        n = len(x)
        if n < 2:
            raise ValueError(f"{name}: at least two observations are needed")
        var = float(x.var(ddof=1))
        chi2 = (n - 1) * var / sigma0 ** 2
        dist = sps.chi2(n - 1)
        if alternative == NOT_EQUAL:
            p = min(1.0, 2 * min(dist.cdf(chi2), dist.sf(chi2)))
        else:
            p = dist.cdf(chi2) if alternative == LESS else dist.sf(chi2)
        lo_tail, hi_tail = tail_levels(confidence, alternative)
        lo = np.nan if lo_tail is None else np.sqrt((n - 1) * var / dist.ppf(1 - lo_tail))
        hi = np.nan if hi_tail is None else np.sqrt((n - 1) * var / dist.ppf(1 - hi_tail))
        rows[name] = {"N": n, "StDev": np.sqrt(var), "Variance": var, "CI Lower (StDev)": lo,
                      "CI Upper (StDev)": hi, "Chi-Square": chi2, "DF": n - 1, "P-Value": float(p)}
    res = Result("1 Variance")
    res.text(f"H0: σ = {sigma0:g}    H1: σ {_symbol(alternative)} {sigma0:g}    {confidence:g}% CI")
    return res.table(_frame(rows))


@analysis("basic.var2", menu=MENU, title="2 Variances", params=[SAMPLE1, SAMPLE2, CONFIDENCE, ALTERNATIVE])
def two_variances(ws, sample1, sample2, confidence=95.0, alternative=NOT_EQUAL):
    x1, x2 = clean(ws.df[sample1]), clean(ws.df[sample2])
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample needs at least two observations")
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    f = v1 / v2
    dist = sps.f(n1 - 1, n2 - 1)
    if alternative == NOT_EQUAL:
        p = min(1.0, 2 * min(dist.cdf(f), dist.sf(f)))
    else:
        p = dist.cdf(f) if alternative == LESS else dist.sf(f)
    lo_tail, hi_tail = tail_levels(confidence, alternative)
    lo = np.nan if lo_tail is None else np.sqrt(f / dist.ppf(1 - lo_tail))
    hi = np.nan if hi_tail is None else np.sqrt(f / dist.ppf(1 - hi_tail))
    levene = sps.levene(x1, x2, center="median")
    res = Result("2 Variances")
    res.text(f"H0: σ1 / σ2 = 1    H1: σ1 / σ2 {_symbol(alternative)} 1    {confidence:g}% CI")
    res.table(pd.DataFrame({"N": [n1, n2], "StDev": np.sqrt([v1, v2]), "Variance": [v1, v2]},
                           index=pd.Index([sample1, sample2], name="Sample")))
    res.table(pd.DataFrame([{"Ratio of StDevs": np.sqrt(f), "CI Lower": lo, "CI Upper": hi}]),
              title="Ratio of standard deviations", index=False)
    return res.table(pd.DataFrame({"Statistic": [f, levene.statistic], "DF1": [n1 - 1, 1],
                                   "DF2": [n2 - 1, n1 + n2 - 2], "P-Value": [float(p), levene.pvalue]},
                                  index=pd.Index(["F", "Levene"], name="Method")), title="Test")


# ----- association -----

CORRELATION_METHODS = ["Pearson", "Spearman"]


@analysis("basic.correlation", menu=MENU, title="Correlation",
          params=[VARIABLES, Param("method", "Method", "choice", default="Pearson", choices=CORRELATION_METHODS),
                  CONFIDENCE])
def correlation(ws, columns, method="Pearson", confidence=95.0):
    if len(columns) < 2:
        raise ValueError("Select at least two columns")
    rows = {}
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            frame = ws.df[[a, b]].dropna().astype(float)
            n = len(frame)
            if n < 3:
                rows[f"{a} – {b}"] = {"N": n, "Correlation": np.nan, "CI Lower": np.nan, "CI Upper": np.nan,
                                      "P-Value": np.nan}
                continue
            func = sps.pearsonr if method == "Pearson" else sps.spearmanr
            r, p = func(frame[a], frame[b])
            lo = hi = np.nan
            if n > 3 and abs(r) < 1:
                # Fisher z interval; the Spearman version uses the Bonett-Wright standard error
                se = 1 / np.sqrt(n - 3) if method == "Pearson" else np.sqrt((1 + r ** 2 / 2) / (n - 3))
                half = sps.norm.ppf(0.5 + confidence / 200) * se
                lo, hi = np.tanh(np.arctanh(r) - half), np.tanh(np.arctanh(r) + half)
            rows[f"{a} – {b}"] = {"N": n, "Correlation": float(r), "CI Lower": lo, "CI Upper": hi,
                                  "P-Value": float(p)}
    res = Result("Correlation")
    res.text(f"{tr(method)}    {confidence:g}% CI")
    res.table(ws.df[list(columns)].astype(float).corr(method=method.lower()), title="Correlations")
    return res.table(_frame(rows, "Pair"), title="Pairwise correlations")


@analysis("basic.covariance", menu=MENU, title="Covariance", params=[VARIABLES])
def covariance(ws, columns):
    return Result("Covariance").table(ws.df[list(columns)].astype(float).cov())


# ----- distribution checks -----

def ryan_joiner(values) -> tuple[float, str]:
    """Ryan-Joiner correlation statistic and an approximate p-value range.

    Critical values from Ryan & Joiner (1976), "Normal probability plots and tests for normality".
    """
    x = np.sort(clean(values))
    n = len(x)
    if n < 4 or x.std() == 0:
        return float("nan"), "*"
    scores = sps.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    r = float(np.corrcoef(x, scores)[0, 1])
    root = np.sqrt(n)
    cv10 = 1.0071 - 0.1371 / root - 0.3682 / n + 0.7780 / n ** 2
    cv05 = 1.0063 - 0.1288 / root - 0.6118 / n + 1.3505 / n ** 2
    cv01 = 0.9963 - 0.0211 / root - 1.4106 / n + 3.1791 / n ** 2
    if r > cv10:
        return r, ">0.100"
    if r < cv01:
        return r, "<0.010"
    if r >= cv05:
        return r, f"{0.05 + 0.05 * (r - cv05) / (cv10 - cv05):.3f}"
    return r, f"{0.01 + 0.04 * (r - cv01) / (cv05 - cv01):.3f}"


NORMALITY_TESTS = ["Anderson-Darling", "Ryan-Joiner", "Kolmogorov-Smirnov"]


@analysis("basic.normality", menu=MENU, title="Normality Test",
          params=[VARIABLES, Param("test", "Test", "choice", default=NORMALITY_TESTS[0], choices=NORMALITY_TESTS)])
def normality_test(ws, columns, test=NORMALITY_TESTS[0]):
    from statsmodels.stats.diagnostic import lilliefors

    from openstat.graphs.basic import probability_plot_figure

    res = Result("Normality Test")
    rows = {}
    for name in columns:
        x = clean(ws.df[name])
        row = {"N": len(x), "Mean": float(x.mean()) if len(x) else np.nan,
               "StDev": float(x.std(ddof=1)) if len(x) > 1 else np.nan}
        if test == NORMALITY_TESTS[0]:
            row["AD"], row["P-Value"] = anderson_darling_normal(x)
        elif test == NORMALITY_TESTS[1]:
            row["RJ"], row["P-Value"] = ryan_joiner(x)
        else:
            ks, p = lilliefors(x, dist="norm", pvalmethod="table") if len(x) > 3 else (np.nan, np.nan)
            row["KS"], row["P-Value"] = float(ks), float(p)
        rows[name] = row
        res.figure(probability_plot_figure(x, name))
    res.blocks.insert(0, ("table", (_frame(rows), test, 4, True)))
    return res


def grubbs(values, alpha=0.05) -> dict:
    """Grubbs' test for one outlier (two-sided)."""
    x = clean(values)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        raise ValueError("Grubbs' test needs at least three non-identical observations")
    mean, sd = x.mean(), x.std(ddof=1)
    i = int(np.argmax(np.abs(x - mean)))
    g = abs(x[i] - mean) / sd
    denom = (n - 1) ** 2 - n * g ** 2
    if denom <= 0:
        p = 0.0
    else:
        t = np.sqrt(n * (n - 2) * g ** 2 / denom)
        p = min(1.0, 2 * n * sps.t.sf(t, n - 2))
    return {"N": n, "Mean": float(mean), "StDev": float(sd), "Minimum": float(x.min()), "Maximum": float(x.max()),
            "Outlier": float(x[i]), "G": float(g), "P-Value": float(p)}


def dixon_tail(q: float, n: int) -> float:
    """P(r10 > q) for the largest of n normal observations.

    Given the smallest value x and the largest z, r10 exceeds q when the other n - 2 values all lie
    below z - q(z - x); that probability is integrated over the joint density of x and z.
    """
    def integrand(z, x):
        inside = sps.norm.cdf(z - q * (z - x)) - sps.norm.cdf(x)
        return sps.norm.pdf(x) * sps.norm.pdf(z) * inside ** (n - 2)
    value, _ = integrate.dblquad(integrand, -8, 8, lambda x: x, lambda x: 8)
    return float(min(max(n * (n - 1) * value, 0.0), 1.0))


DIXON_MAX_N = 100


def dixon(values) -> dict:
    """Dixon's Q test (ratio r10) for one outlier at either end (two-sided)."""
    x = np.sort(clean(values))
    n = len(x)
    if n < 3 or n > DIXON_MAX_N or x[-1] == x[0]:
        raise ValueError("Dixon's test needs 3 to 100 non-identical observations")
    spread = x[-1] - x[0]
    low, high = (x[1] - x[0]) / spread, (x[-1] - x[-2]) / spread
    q = max(low, high)
    return {"N": n, "Minimum": float(x[0]), "Maximum": float(x[-1]), "Outlier": float(x[-1] if high >= low else x[0]),
            "Q": float(q), "P-Value": min(1.0, 2 * dixon_tail(q, n))}


OUTLIER_TESTS = ["Grubbs", "Dixon's Q (r10)"]


@analysis("basic.outlier", menu=MENU, title="Outlier Test",
          params=[VARIABLES, Param("alpha", "Significance level", "number", default=0.05, minimum=0.0001, maximum=0.5),
                  Param("test", "Test", "choice", default=OUTLIER_TESTS[0], choices=OUTLIER_TESTS)])
def outlier_test(ws, columns, alpha=0.05, test=OUTLIER_TESTS[0]):
    if test == OUTLIER_TESTS[0]:
        rows = {name: grubbs(ws.df[name], alpha) for name in columns}
    else:
        rows = {name: dixon(ws.df[name]) for name in columns}
    res = Result("Outlier Test")
    res.text(f"{tr('Grubbs test' if test == OUTLIER_TESTS[0] else 'Dixon Q test')}    α = {alpha:g}")
    res.table(_frame(rows))
    for name, row in rows.items():
        found = row["P-Value"] < alpha
        res.text(f"{name}: " + (f"{tr('outlier at')} {row['Outlier']:g}" if found else tr("no outlier at this significance level")))
    return res


@analysis("basic.gof_poisson", menu=MENU, title="Goodness-of-Fit Test for Poisson",
          params=[Param("column", "Counts", "column")])
def gof_poisson(ws, column):
    x = clean(ws.df[column])
    if len(x) < 2 or (x < 0).any() or (x != np.round(x)).any():
        raise ValueError("Counts must be non-negative integers")
    x = x.astype(int)
    n, mean, top = len(x), x.mean(), int(x.max())
    if top < 2:
        raise ValueError("At least three categories are needed")
    observed = np.array([(x == k).sum() for k in range(top)] + [(x >= top).sum()], dtype=float)
    probs = np.append(sps.poisson.pmf(np.arange(top), mean), sps.poisson.sf(top - 1, mean))
    expected = n * probs
    contribution = (observed - expected) ** 2 / expected
    df = len(observed) - 2
    chi2 = float(contribution.sum())
    res = Result("Goodness-of-Fit Test for Poisson")
    res.text(f"{tr('Poisson mean')}: {mean:.5g}    N = {n}")
    labels = [str(k) for k in range(top)] + [f">={top}"]
    res.table(pd.DataFrame({"Poisson probability": probs, "Observed": observed, "Expected": expected,
                            "Contribution to Chi-Square": contribution}, index=pd.Index(labels, name=column)))
    res.table(pd.DataFrame([{"N": n, "DF": df, "Chi-Square": chi2, "P-Value": float(sps.chi2.sf(chi2, df))}]),
              title="Chi-Square Test", index=False)
    if (expected < 5).any():
        res.text(f"{int((expected < 5).sum())} {tr('cells with expected counts less than 5')}")
    return res

