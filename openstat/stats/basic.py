"""Basic statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openstat.core.registry import Param, analysis
from openstat.core.result import Result


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
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75], method="weibull")
    return {"N": n, "N*": missing, "Mean": float(x.mean()), "SE Mean": sd / np.sqrt(n), "StDev": sd,
            "Minimum": float(x.min()), "Q1": float(q1), "Median": float(med), "Q3": float(q3),
            "Maximum": float(x.max())}


@analysis("basic.describe", menu=("Stat", "Basic Statistics"), title="Descriptive Statistics",
          params=[Param("columns", "Variables", "columns"),
                  Param("by", "By variable", "column", numeric=False, optional=True)])
def describe(ws, columns, by=None):
    res = Result("Descriptive Statistics")
    rows, labels = [], []
    for name in columns:
        if by:
            for level, group in ws.df.groupby(by, sort=True)[name]:
                rows.append(describe_values(group))
                labels.append(f"{name} ({by} = {level})")
        else:
            rows.append(describe_values(ws.df[name]))
            labels.append(name)
    table = pd.DataFrame(rows, index=pd.Index(labels, name="Variable"))
    res.table(table, digits=4)
    return res
