"""Calc menu: calculator, column/row statistics, random data and probability distributions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openstat.core.expression import evaluate
from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.i18n import tr
from openstat.stats import distributions as dists

MENU = ("Calc",)

DIST_PARAMS = [
    Param("distribution", "Distribution", "choice", default=dists.NAMES[0], choices=dists.NAMES),
    Param("p1", "Parameter 1", "number", default=0.0),
    Param("p2", "Parameter 2", "number", default=1.0),
    Param("p3", "Parameter 3", "number", default=0.0),
]


@analysis("calc.calculator", menu=MENU, title="Calculator",
          params=[Param("store", "Store result in column", "text", default="Result"),
                  Param("expression", "Expression", "text")])
def calculator(ws, store, expression):
    value = evaluate(ws, expression)
    res = Result("Calculator")
    res.text(f"{store} = {expression}")
    if np.ndim(value) == 0:
        res.text(f"{tr('Value')}: {value}")
        values = [value]
    else:
        values = np.asarray(value)
        values = values.astype(float).tolist() if values.dtype == bool else values.tolist()
    res.store(store.strip() or "Result", values)
    return res


COLUMN_STATS = {
    "Sum": lambda x: x.sum(), "Mean": lambda x: x.mean(), "StDev": lambda x: x.std(ddof=1),
    "Variance": lambda x: x.var(ddof=1), "Minimum": lambda x: x.min(), "Maximum": lambda x: x.max(),
    "Range": lambda x: x.max() - x.min(), "Median": lambda x: x.median(),
    "Sum of squares": lambda x: (x ** 2).sum(), "N": lambda x: float(x.count()),
}


@analysis("calc.column_stats", menu=MENU, title="Column Statistics",
          params=[Param("columns", "Variables", "columns")])
def column_statistics(ws, columns):
    frame = ws.df[list(columns)].astype(float)
    table = pd.DataFrame({name: frame.apply(func) for name, func in COLUMN_STATS.items()})
    table["N*"] = frame.isna().sum().astype(float)
    table.index.name = "Variable"
    return Result("Column Statistics").table(table)


@analysis("calc.row_stats", menu=MENU, title="Row Statistics",
          params=[Param("columns", "Variables", "columns"),
                  Param("statistic", "Statistic", "choice", default="Mean", choices=list(COLUMN_STATS))])
def row_statistics(ws, columns, statistic="Mean"):
    frame = ws.df[list(columns)].astype(float)
    values = frame.apply(COLUMN_STATS[statistic], axis=1)
    name = f"Row{statistic.replace(' ', '')}"
    res = Result("Row Statistics").store(name, values.tolist())
    res.text(f"{tr(statistic)}: {', '.join(columns)}")
    res.text(f"{tr('Stored in columns')}: {name}")
    return res


@analysis("calc.patterned", menu=MENU, title="Make Patterned Data",
          params=[Param("first", "From first value", "number", default=1.0),
                  Param("last", "To last value", "number", default=10.0),
                  Param("step", "In steps of", "number", default=1.0),
                  Param("repeat_each", "Times to list each value", "int", default=1, minimum=1),
                  Param("repeat_all", "Times to list the sequence", "int", default=1, minimum=1)])
def patterned(ws, first=1.0, last=10.0, step=1.0, repeat_each=1, repeat_all=1):
    if step == 0 or (last - first) * step < 0:
        raise ValueError("The step does not lead from the first value to the last value")
    count = int(np.floor((last - first) / step + 1e-9)) + 1
    seq = first + step * np.arange(count)
    seq = np.tile(np.repeat(seq, repeat_each), repeat_all)
    res = Result("Make Patterned Data").store("Sequence", seq.tolist())
    res.text(f"{tr('Stored in columns')}: Sequence ({len(seq)})")
    return res


@analysis("calc.random", menu=MENU, title="Random Data",
          params=[Param("rows", "Number of rows", "int", default=100, minimum=1, maximum=10_000_000),
                  *DIST_PARAMS,
                  Param("seed", "Base for random numbers (0 = random)", "int", default=0, minimum=0)])
def random_data(ws, rows=100, distribution=dists.NAMES[0], p1=0.0, p2=1.0, p3=0.0, seed=0):
    dist = dists.make(distribution, p1, p2, p3)
    rng = np.random.default_rng(seed or None)
    values = dist.rvs(size=rows, random_state=rng)
    res = Result("Random Data").store("Random", np.asarray(values, dtype=float).tolist())
    res.text(f"{tr(distribution)}: {p1:g}, {p2:g}, {p3:g}")
    res.text(f"{tr('Stored in columns')}: Random ({rows})")
    return res


FUNCTIONS = ["Probability density", "Cumulative probability", "Inverse cumulative probability"]


@analysis("calc.probability", menu=MENU, title="Probability Distributions",
          params=[Param("function", "Function", "choice", default=FUNCTIONS[1], choices=FUNCTIONS),
                  *DIST_PARAMS,
                  Param("column", "Input column", "column", optional=True),
                  Param("constant", "Input constant", "number", default=0.0)])
def probability(ws, function=FUNCTIONS[1], distribution=dists.NAMES[0], p1=0.0, p2=1.0, p3=0.0,
                column=None, constant=0.0):
    dist = dists.make(distribution, p1, p2, p3)
    x = ws.df[column].to_numpy(dtype=float) if column else np.array([float(constant)])
    if function == FUNCTIONS[0]:
        y, label = dists.density(dist, x), "f(x)"
    elif function == FUNCTIONS[1]:
        y, label = dist.cdf(x), "P(X <= x)"
    else:
        if np.nanmin(x) < 0 or np.nanmax(x) > 1:
            raise ValueError("Probabilities must be between 0 and 1")
        y, label = dist.ppf(x), "x"
    res = Result("Probability Distributions")
    res.text(f"{tr(function)} - {tr(distribution)}: {p1:g}, {p2:g}, {p3:g}")
    first = "P(X <= x)" if function == FUNCTIONS[2] else "x"
    table = pd.DataFrame({first: x, label: y})
    res.table(table.head(200), digits=6, index=False)
    if column:
        res.store(f"{'Inv' if function == FUNCTIONS[2] else 'Prob'}_{column}", np.asarray(y, dtype=float).tolist())
    return res
