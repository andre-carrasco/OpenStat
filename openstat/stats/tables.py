"""Tables: tallies, cross tabulation and chi-square tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.inter_rater import cohens_kappa

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.i18n import tr
from openstat.stats._tests import CONFIDENCE

MENU = ("Stat", "Tables")


@analysis("tables.tally", menu=MENU, title="Tally Individual Variables",
          params=[Param("columns", "Variables", "columns", numeric=False)])
def tally(ws, columns):
    res = Result("Tally Individual Variables")
    for name in columns:
        col = ws.df[name]
        counts = col.dropna().map(format_value).value_counts(sort=False)
        try:
            counts = counts.loc[sorted(counts.index, key=float)]
        except ValueError:
            counts = counts.sort_index()
        table = pd.DataFrame({"Count": counts, "Percent": 100 * counts / counts.sum()})
        table["CumCnt"] = table["Count"].cumsum()
        table["CumPct"] = table["Percent"].cumsum()
        table.index.name = name
        res.table(table, digits=2)
        res.text(f"N = {int(counts.sum())}    N* = {int(col.isna().sum())}")
    return res


def chi_square_table(observed: pd.DataFrame, res: Result) -> dict:
    """Add expected counts and association tests for a contingency table to `res`."""
    obs = observed.to_numpy(dtype=float)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        raise ValueError("The table needs at least two rows and two columns")
    if (obs < 0).any() or obs.sum() == 0:
        raise ValueError("Counts must be non-negative and not all zero")
    total = obs.sum()
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / total
    if (expected == 0).any():
        raise ValueError("A row or column has a total of zero")
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    pearson = float(((obs - expected) ** 2 / expected).sum())
    mask = obs > 0
    likelihood = float(2 * (obs[mask] * np.log(obs[mask] / expected[mask])).sum())
    res.table(pd.DataFrame(expected, index=observed.index, columns=observed.columns), title="Expected counts",
              digits=2)
    tests = pd.DataFrame({"Chi-Square": [pearson, likelihood], "DF": [df, df],
                          "P-Value": [sps.chi2.sf(pearson, df), sps.chi2.sf(likelihood, df)]},
                         index=pd.Index(["Pearson", "Likelihood ratio"], name="Test"))
    res.table(tests, title="Chi-Square Test")
    out = {"pearson": pearson, "likelihood": likelihood, "df": df, "p": float(sps.chi2.sf(pearson, df))}
    cramers_v = float(np.sqrt(pearson / (total * (min(obs.shape) - 1))))
    extra = {"Cramér's V": cramers_v}
    if obs.shape == (2, 2):
        extra["Fisher's exact P-Value"] = float(sps.fisher_exact(obs.astype(int)).pvalue)
        out["fisher"] = extra["Fisher's exact P-Value"]
    res.table(pd.DataFrame([extra]), index=False)
    low = int((expected < 5).sum())
    if low:
        res.text(f"{low} {tr('cells with expected counts less than 5')}")
    return out


def ordered(labels) -> list[str]:
    """Category labels in numeric order when they are all numbers, otherwise alphabetical."""
    try:
        return sorted(labels, key=float)
    except ValueError:
        return sorted(labels)


def crosstab(first: pd.Series, second: pd.Series, orders: tuple | None = None) -> pd.DataFrame:
    """Counts of two columns, with the categories of each in order (`orders`: the row and column labels, in order)."""
    table = pd.crosstab(first.map(format_value), second.map(format_value))
    rows, cols = orders or (ordered(table.index), ordered(table.columns))
    return table.loc[rows, cols]


def ordinal_measures(obs: np.ndarray) -> dict:
    """Concordant and discordant pairs of a table with ordered rows and columns, and the measures
    of ordinal association built on them."""
    concordant = discordant = 0.0
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            concordant += obs[i, j] * obs[i + 1:, j + 1:].sum()
            discordant += obs[i, j] * obs[i + 1:, :j].sum()
    n = obs.sum()
    pairs = n * (n - 1) / 2
    untied_rows = pairs - (obs.sum(axis=1) * (obs.sum(axis=1) - 1) / 2).sum()   # pairs in different rows
    untied_cols = pairs - (obs.sum(axis=0) * (obs.sum(axis=0) - 1) / 2).sum()
    excess = concordant - discordant
    nan = float("nan")
    return {"Concordant pairs": concordant, "Discordant pairs": discordant,
            "Goodman-Kruskal gamma": excess / (concordant + discordant) if concordant + discordant else nan,
            "Kendall's tau-b": excess / np.sqrt(untied_rows * untied_cols) if untied_rows * untied_cols else nan,
            "Somers' D (columns dependent)": excess / untied_rows if untied_rows else nan,
            "Somers' D (rows dependent)": excess / untied_cols if untied_cols else nan}


@analysis("tables.crosstab", menu=MENU, title="Cross Tabulation and Chi-Square",
          params=[Param("rows", "Rows", "column", numeric=False),
                  Param("cols", "Columns", "column", numeric=False),
                  Param("measures", "Measures for ordinal categories and agreement (kappa)", "bool", default=False)])
def cross_tabulation(ws, rows, cols, measures=False):
    frame = ws.df[[rows, cols]].dropna()
    observed = crosstab(frame[rows], frame[cols], (ws.levels(rows, frame[rows].map(format_value)),
                                                   ws.levels(cols, frame[cols].map(format_value))))
    res = Result("Cross Tabulation and Chi-Square")
    res.text(f"{tr('Rows')}: {rows}    {tr('Columns')}: {cols}")
    shown = observed.copy()
    shown[tr("All")] = shown.sum(axis=1)
    shown.loc[tr("All")] = shown.sum(axis=0)
    shown.index.name = rows
    res.table(shown, title="Observed counts")
    chi_square_table(observed, res)
    if measures:
        res.table(pd.DataFrame([ordinal_measures(observed.to_numpy(dtype=float))]),
                  title="Measures of association for ordinal categories", index=False)
        if set(observed.index) == set(observed.columns):
            kappa = cohens_kappa(observed.loc[:, list(observed.index)].to_numpy())
            res.table(pd.DataFrame([{"Kappa": kappa.kappa, "SE Kappa": kappa.std_kappa0, "Z": kappa.z_value,
                                     "P-Value": kappa.pvalue_two_sided}]), title="Cohen's kappa", index=False)
    return res


def _two_levels(col: pd.Series, name: str) -> pd.Series:
    labels = col.map(format_value)
    if labels.nunique() != 2:
        raise ValueError(f"{name}: the column must have exactly two categories")
    return labels


@analysis("tables.mcnemar", menu=MENU, title="McNemar's Test",
          params=[Param("first", "First measurement", "column", numeric=False),
                  Param("second", "Second measurement", "column", numeric=False), CONFIDENCE])
def mcnemar(ws, first, second, confidence=95.0):
    frame = ws.df[[first, second]].dropna()
    a, b = _two_levels(frame[first], first), _two_levels(frame[second], second)
    levels_a, levels_b = ordered(a.unique()), ordered(b.unique())
    table = pd.crosstab(a, b).reindex(index=levels_a, columns=levels_b, fill_value=0)
    table.index.name, table.columns.name = first, second
    n = int(table.to_numpy().sum())
    changed_up, changed_down = int(table.iloc[0, 1]), int(table.iloc[1, 0])   # the discordant pairs
    discordant = changed_up + changed_down
    difference = (changed_up - changed_down) / n
    se = np.sqrt(discordant - (changed_up - changed_down) ** 2 / n) / n
    half = sps.norm.ppf(0.5 + confidence / 200) * se
    p = float(sps.binomtest(changed_up, discordant, 0.5).pvalue) if discordant else 1.0
    res = Result("McNemar's Test")
    if levels_a != levels_b:
        pairing = ", ".join(f"{x} = {y}" for x, y in zip(levels_a, levels_b))
        res.text(f"{tr('The categories are paired in order')}: {pairing}")
    res.table(table, title="Observed counts")
    res.text(f"{tr('Difference')}: p({second} = {levels_b[1]}) − p({first} = {levels_a[1]})")
    res.table(pd.DataFrame([{"N": n, "Discordant pairs": discordant, "Difference": difference,
                             "CI Lower": difference - half, "CI Upper": difference + half,
                             "Exact P-Value": p}]), index=False)
    return res


@analysis("tables.cmh", menu=MENU, title="Cochran-Mantel-Haenszel Test",
          params=[Param("rows", "Rows", "column", numeric=False),
                  Param("cols", "Columns", "column", numeric=False),
                  Param("layers", "Layers", "column", numeric=False), CONFIDENCE])
def cochran_mantel_haenszel(ws, rows, cols, layers, confidence=95.0):
    frame = ws.df[[rows, cols, layers]].dropna()
    a, b = _two_levels(frame[rows], rows), _two_levels(frame[cols], cols)
    levels_a, levels_b = ordered(a.unique()), ordered(b.unique())
    strata = frame[layers].map(format_value)
    tables, odds = [], {}
    for layer in ordered(strata.unique()):
        inside = (strata == layer).to_numpy()
        table = pd.crosstab(a[inside], b[inside]).reindex(index=levels_a, columns=levels_b, fill_value=0).to_numpy()
        if table.sum() < 2:
            continue
        tables.append(table)
        cross = table[0, 1] * table[1, 0]
        odds[layer] = {"N": int(table.sum()), "Odds ratio": table[0, 0] * table[1, 1] / cross if cross else np.nan}
    if len(tables) < 2:
        raise ValueError("At least two layers with data are needed")
    stratified = StratifiedTable(tables)
    test = stratified.test_null_odds(correction=True)
    lo, hi = stratified.oddsratio_pooled_confint(alpha=1 - confidence / 100)
    homogeneity = stratified.test_equal_odds()
    res = Result("Cochran-Mantel-Haenszel Test")
    res.text(f"{tr('Rows')}: {rows} ({', '.join(levels_a)})    {tr('Columns')}: {cols} ({', '.join(levels_b)})    "
             f"{tr('Layers')}: {layers}")
    res.table(_named(pd.DataFrame.from_dict(odds, orient="index"), layers), title="Odds ratio by layer")
    res.table(pd.DataFrame([{"Common odds ratio": stratified.oddsratio_pooled, "CI Lower": lo, "CI Upper": hi,
                             "CMH Chi-Square": float(test.statistic), "DF": 1, "P-Value": float(test.pvalue)}]),
              title="Test of conditional independence", index=False)
    res.table(pd.DataFrame([{"Chi-Square": float(homogeneity.statistic), "DF": len(tables) - 1,
                             "P-Value": float(homogeneity.pvalue)}]),
              title="Breslow-Day test of homogeneous odds ratios", index=False)
    return res


def _named(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame.index.name = name
    return frame


TABLE_STATISTICS = {"Mean": "mean", "Median": "median", "StDev": "std", "Sum": "sum", "Minimum": "min",
                    "Maximum": "max", "N": "count"}


@analysis("tables.describe", menu=MENU, title="Table of Descriptive Statistics",
          params=[Param("rows", "Rows", "column", numeric=False),
                  Param("variable", "Summarized variable", "column"),
                  Param("cols", "Columns", "column", numeric=False, optional=True),
                  Param("statistic", "Statistic", "choice", default="Mean", choices=list(TABLE_STATISTICS))])
def describe_table(ws, rows, variable, cols=None, statistic="Mean"):
    keys = [rows] + ([cols] if cols else [])
    frame = ws.df[keys + [variable]].dropna(subset=keys)
    labels = [frame[k].map(format_value) for k in keys]
    grouped = frame[variable].astype(float).groupby(labels).agg(TABLE_STATISTICS[statistic])
    if cols:
        table = grouped.unstack()
        table = table.loc[ordered(table.index), ordered(table.columns)]
    else:
        table = grouped.loc[ordered(grouped.index)].to_frame(statistic)
    res = Result("Table of Descriptive Statistics")
    res.text(f"{tr(statistic)}: {variable}    {tr('Rows')}: {rows}" + (f"    {tr('Columns')}: {cols}" if cols else ""))
    return res.table(_named(table, rows))


@analysis("tables.chisq_association", menu=MENU, title="Chi-Square Test for Association",
          params=[Param("columns", "Columns containing the table", "columns"),
                  Param("labels", "Row labels", "column", numeric=False, optional=True)])
def chi_square_association(ws, columns, labels=None):
    frame = ws.df[list(columns) + ([labels] if labels else [])].dropna()
    observed = frame[list(columns)].astype(float)
    observed.index = frame[labels].map(format_value) if labels else [str(i + 1) for i in range(len(frame))]
    observed.index.name = labels or tr("Rows")
    res = Result("Chi-Square Test for Association")
    res.table(observed, title="Observed counts", digits=0)
    chi_square_table(observed, res)
    return res


@analysis("tables.chisq_gof", menu=MENU, title="Chi-Square Goodness-of-Fit Test",
          params=[Param("observed", "Observed counts", "column"),
                  Param("proportions", "Expected proportions (blank = equal)", "column", optional=True),
                  Param("labels", "Category names", "column", numeric=False, optional=True)])
def chi_square_gof(ws, observed, proportions=None, labels=None):
    cols = [observed] + [c for c in (proportions, labels) if c]
    frame = ws.df[cols].dropna()
    obs = frame[observed].to_numpy(dtype=float)
    k = len(obs)
    if k < 2:
        raise ValueError("At least two categories are needed")
    props = frame[proportions].to_numpy(dtype=float) if proportions else np.full(k, 1 / k)
    if (props <= 0).any():
        raise ValueError("Expected proportions must be positive")
    props = props / props.sum()
    expected = obs.sum() * props
    contribution = (obs - expected) ** 2 / expected
    chi2 = float(contribution.sum())
    index = pd.Index(frame[labels].map(format_value) if labels else [str(i + 1) for i in range(k)],
                     name=labels or "Category")
    res = Result("Chi-Square Goodness-of-Fit Test")
    res.table(pd.DataFrame({"Observed": obs, "Test proportion": props, "Expected": expected,
                            "Contribution to Chi-Square": contribution}, index=index))
    res.table(pd.DataFrame([{"N": int(obs.sum()), "DF": k - 1, "Chi-Square": chi2,
                             "P-Value": float(sps.chi2.sf(chi2, k - 1))}]), title="Chi-Square Test", index=False)
    return res
