"""Tables: tallies, cross tabulation and chi-square tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.i18n import tr

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


@analysis("tables.crosstab", menu=MENU, title="Cross Tabulation and Chi-Square",
          params=[Param("rows", "Rows", "column", numeric=False),
                  Param("cols", "Columns", "column", numeric=False)])
def cross_tabulation(ws, rows, cols):
    frame = ws.df[[rows, cols]].dropna()
    observed = pd.crosstab(frame[rows].map(format_value), frame[cols].map(format_value))
    res = Result("Cross Tabulation and Chi-Square")
    res.text(f"{tr('Rows')}: {rows}    {tr('Columns')}: {cols}")
    shown = observed.copy()
    shown[tr("All")] = shown.sum(axis=1)
    shown.loc[tr("All")] = shown.sum(axis=0)
    shown.index.name = rows
    res.table(shown, title="Observed counts")
    chi_square_table(observed, res)
    return res


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
