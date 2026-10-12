"""Data menu: reshaping and recoding worksheet data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openstat.core.expression import evaluate
from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.i18n import tr

MENU = ("Data",)


def _stored_note(res: Result) -> Result:
    if res.stored:
        res.text(f"{tr('Stored in columns')}: {', '.join(res.stored)}")
    for name, df in res.worksheets:
        res.text(f"{tr('New worksheet')}: {name} ({len(df)} x {df.shape[1]})")
    return res


@analysis("data.sort", menu=MENU, title="Sort",
          params=[Param("by", "Sort by", "columns", numeric=False),
                  Param("descending", "Descending", "bool", default=False)])
def sort(ws, by, descending=False):
    res = Result("Sort")
    res.worksheet(f"{ws.name} (sort)", ws.df.sort_values(list(by), ascending=not descending,
                                                         na_position="last", kind="stable"))
    return _stored_note(res)


@analysis("data.subset", menu=MENU, title="Subset Worksheet",
          params=[Param("condition", "Condition (e.g. C1 > 10)", "text")])
def subset(ws, condition):
    mask = np.asarray(evaluate(ws, condition))
    if mask.shape != (ws.nrows,):
        raise ValueError("The condition must give one value per row")
    res = Result("Subset Worksheet")
    res.worksheet(f"{ws.name} (subset)", ws.df[pd.Series(mask).fillna(False).astype(bool).to_numpy()])
    return _stored_note(res)


@analysis("data.stack", menu=MENU, title="Stack Columns",
          params=[Param("columns", "Columns to stack", "columns", numeric=False)])
def stack(ws, columns):
    values, labels = [], []
    for name in columns:
        col = ws.column(name)
        values.extend(col.tolist())
        labels.extend([name] * len(col))
    res = Result("Stack Columns")
    res.store("Stacked", values).store("Subscripts", labels)
    return _stored_note(res)


@analysis("data.unstack", menu=MENU, title="Unstack Columns",
          params=[Param("column", "Column to unstack", "column", numeric=False),
                  Param("by", "Subscripts", "column", numeric=False)])
def unstack(ws, column, by):
    res = Result("Unstack Columns")
    for level, group in ws.df.groupby(by, sort=True)[column]:
        res.store(f"{column}_{format_value(level)}", group.dropna().tolist())
    return _stored_note(res)


@analysis("data.transpose", menu=MENU, title="Transpose Columns",
          params=[Param("columns", "Columns", "columns", numeric=False)])
def transpose(ws, columns):
    frame = ws.df[list(columns)].T
    frame.columns = [f"C{i + 2}" for i in range(frame.shape[1])]
    frame.insert(0, "Labels", list(columns))
    res = Result("Transpose Columns")
    res.worksheet(f"{ws.name} (transpose)", frame.reset_index(drop=True))
    return _stored_note(res)


@analysis("data.rank", menu=MENU, title="Rank",
          params=[Param("column", "Column", "column")])
def rank(ws, column):
    res = Result("Rank")
    res.store(f"Rank_{column}", ws.df[column].rank(method="average").tolist())
    return _stored_note(res)


STANDARDIZE = ["Subtract mean and divide by standard deviation", "Subtract mean",
               "Divide by standard deviation", "Scale to range 0-1"]


@analysis("data.standardize", menu=MENU, title="Standardize",
          params=[Param("columns", "Columns", "columns"),
                  Param("method", "Method", "choice", default=STANDARDIZE[0], choices=STANDARDIZE)])
def standardize(ws, columns, method=STANDARDIZE[0]):
    res = Result("Standardize")
    for name in columns:
        x = ws.df[name].astype(float)
        if method == STANDARDIZE[0]:
            out = (x - x.mean()) / x.std(ddof=1)
        elif method == STANDARDIZE[1]:
            out = x - x.mean()
        elif method == STANDARDIZE[2]:
            out = x / x.std(ddof=1)
        else:
            out = (x - x.min()) / (x.max() - x.min())
        res.store(f"Std_{name}", out.tolist())
    return _stored_note(res)


def parse_recode(mapping: str):
    """Rules 'old=new; lo:hi=new' -> list of (kind, old, new)."""
    rules = []
    for part in mapping.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid recode rule: {part}")
        old, new = (s.strip() for s in part.rsplit("=", 1))
        if ":" in old:
            lo, hi = (float(s.replace(",", ".")) for s in old.split(":", 1))
            rules.append(("range", (lo, hi), new))
        else:
            rules.append(("value", old, new))
    if not rules:
        raise ValueError("No recode rules given")
    return rules


@analysis("data.recode", menu=MENU, title="Recode",
          params=[Param("column", "Column", "column", numeric=False),
                  Param("mapping", "Rules (old=new; low:high=new)", "text")])
def recode(ws, column, mapping):
    rules = parse_recode(mapping)
    out = []
    for value in ws.df[column].tolist():
        if pd.isna(value):
            out.append(None)
            continue
        text, new = format_value(value), value
        for kind, old, target in rules:
            if kind == "value" and text == old:
                new = target
                break
            if kind == "range" and isinstance(value, (int, float)) and old[0] <= value <= old[1]:
                new = target
                break
        out.append(new)
    from openstat.core.worksheet import parse_cell
    res = Result("Recode")
    res.store(f"Coded_{column}", [parse_cell(v) if isinstance(v, str) else v for v in out])
    return _stored_note(res)
