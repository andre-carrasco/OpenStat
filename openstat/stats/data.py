"""Data menu: reshaping and recoding worksheet data."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from openstat.core.expression import evaluate
from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value, parse_cell, parse_date
from openstat.i18n import tr
from openstat.stats._util import by_levels

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
    res = Result("Subset Worksheet")
    res.worksheet(f"{ws.name} (subset)", ws.df[_row_mask(ws, condition)])
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
    for level, rows in by_levels(ws, by):
        group = rows[column]
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
    res = Result("Recode")
    res.store(f"Coded_{column}", [parse_cell(v) if isinstance(v, str) else v for v in out])
    return _stored_note(res)


def _unique_names(names: list[str]) -> list[str]:
    """`names` with repeats renamed name_2, name_3, ..."""
    seen, out = set(), []
    for name in names:
        new, k = name, 1
        while new in seen:
            k += 1
            new = f"{name}_{k}"
        seen.add(new)
        out.append(new)
    return out


def _row_mask(ws, condition: str) -> np.ndarray:
    mask = np.asarray(evaluate(ws, condition))
    if mask.shape != (ws.nrows,):
        raise ValueError("The condition must give one value per row")
    return pd.Series(mask).fillna(False).astype(bool).to_numpy()


JOINS = ["All rows of both worksheets", "Only rows found in both worksheets",
         "All rows of this worksheet", "All rows of the other worksheet"]
_HOW = dict(zip(JOINS, ["outer", "inner", "left", "right"]))


@analysis("data.merge", menu=MENU, title="Merge Worksheets",
          params=[Param("other", "Worksheet to merge with", "worksheet"),
                  Param("by", "Match rows by (none = side by side)", "columns", numeric=False, optional=True),
                  Param("rows", "Rows to keep", "choice", default=JOINS[0], choices=JOINS)])
def merge(ws, other, by=(), rows=JOINS[0]):
    by = list(by)
    missing = [c for c in by if c not in other.df.columns]
    if missing:
        raise ValueError(f"Column not found in the other worksheet: {', '.join(missing)}")
    right = other.df[[c for c in other.df.columns if c not in by]].copy()
    right.columns = _unique_names(list(ws.df.columns) + list(right.columns))[ws.ncols:]
    if by:
        frame = ws.df.merge(pd.concat([other.df[by], right], axis=1), on=by, how=_HOW[rows], sort=False)
    else:
        frame = pd.concat([ws.df, right], axis=1)
    res = Result("Merge Worksheets")
    res.worksheet(f"{ws.name} + {other.name}", frame)
    return _stored_note(res)


MAX_SPLIT = 50


@analysis("data.split", menu=MENU, title="Split Worksheet",
          params=[Param("by", "By variable", "column", numeric=False)])
def split(ws, by):
    groups = by_levels(ws, by)
    if len(groups) > MAX_SPLIT:
        raise ValueError("Too many groups to split the worksheet by")
    res = Result("Split Worksheet")
    for level, frame in groups:
        res.worksheet(f"{ws.name} ({by} = {format_value(level)})", frame)
    return _stored_note(res)


@analysis("data.copy_columns", menu=MENU, title="Copy Columns to New Worksheet",
          params=[Param("columns", "Columns", "columns", numeric=False),
                  Param("condition", "Rows to include (condition, empty = all)", "text")])
def copy_columns(ws, columns, condition=""):
    frame = ws.df[list(columns)]
    if condition.strip():
        frame = frame[_row_mask(ws, condition)]
    res = Result("Copy Columns to New Worksheet")
    res.worksheet(f"{ws.name} (copy)", frame)
    return _stored_note(res)


@analysis("data.concatenate", menu=MENU, title="Concatenate",
          params=[Param("columns", "Columns", "columns", numeric=False),
                  Param("separator", "Separator", "text")])
def concatenate(ws, columns, separator=""):
    values = [separator.join(format_value(v) for v in row if not pd.isna(v)) or None
              for row in ws.df[list(columns)].itertuples(index=False)]
    res = Result("Concatenate")
    res.store("Concat", values)
    return _stored_note(res)


def parse_rows(spec: str, nrows: int) -> list[int]:
    """Row numbers '1, 3:5' (counted from 1) -> sorted positions counted from 0."""
    rows = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            first, last = (int(s) for s in part.split(":", 1)) if ":" in part else (int(part), int(part))
        except ValueError:
            raise ValueError(f"Invalid row number: {part}") from None
        if first < 1 or last < first:
            raise ValueError(f"Invalid row number: {part}")
        rows.update(range(first - 1, min(last, nrows)))
    if not rows:
        raise ValueError("No rows given")
    return sorted(rows)


@analysis("data.delete_rows", menu=MENU, title="Delete Rows",
          params=[Param("rows", "Rows to delete (e.g. 1, 3:5)", "text")])
def delete_rows(ws, rows):
    drop = parse_rows(rows, ws.nrows)
    res = Result("Delete Rows")
    res.text(f"{tr('Rows deleted')}: {len(drop)}")
    res.worksheet(f"{ws.name} (rows deleted)", ws.df.drop(index=drop))
    return _stored_note(res)


# The day number of a date counts from this origin, as in spreadsheets
DATE_ORIGIN = pd.Timestamp("1899-12-30")
TYPES = ["Numeric", "Text", "Date/time"]


@analysis("data.change_type", menu=MENU, title="Change Data Type",
          params=[Param("column", "Column", "column", numeric=False),
                  Param("to", "New type", "choice", default=TYPES[1], choices=TYPES)])
def change_type(ws, column, to=TYPES[1]):
    values = [None if pd.isna(v) else v for v in ws.df[column].tolist()]
    out = []
    for value in values:
        if value is None or to == TYPES[1]:
            out.append(None if value is None else format_value(value))
            continue
        parsed = parse_cell(value) if isinstance(value, str) else value
        if to == TYPES[0]:
            if isinstance(parsed, datetime):
                parsed = (pd.Timestamp(parsed) - DATE_ORIGIN) / pd.Timedelta(days=1)
            out.append(float(parsed) if isinstance(parsed, (int, float)) else None)
        else:
            if isinstance(parsed, (int, float)):
                parsed = DATE_ORIGIN + pd.Timedelta(days=float(parsed))
            out.append(parsed if isinstance(parsed, datetime) else None)
    failed = sum(o is None for o in out) - sum(v is None for v in values)
    res = Result("Change Data Type")
    res.text(f"{column} -> {tr(to)}")
    if failed:
        res.text(f"{tr('Values that could not be converted, set to missing')}: {failed}")
    res.store(f"{column}_{dict(zip(TYPES, ['num', 'text', 'date']))[to]}", out)
    return _stored_note(res)


def as_dates(ws, column: str) -> pd.Series:
    """The column as timestamps; a text column is read as dates."""
    if ws.is_datetime(column):
        return ws.df[column]
    if ws.is_numeric(column):
        raise ValueError("The column does not contain dates")
    dates = pd.Series(pd.to_datetime([None if pd.isna(v) else parse_date(v) for v in ws.df[column]]))
    if dates.notna().sum() == 0 or dates.notna().sum() != ws.df[column].notna().sum():
        raise ValueError("The column does not contain dates")
    return dates


DATE_PARTS = {
    "Year": lambda d: d.dt.year, "Quarter": lambda d: d.dt.quarter, "Month": lambda d: d.dt.month,
    "Week of the year": lambda d: d.dt.isocalendar().week, "Day of the month": lambda d: d.dt.day,
    "Day of the week (1 = Monday)": lambda d: d.dt.dayofweek + 1, "Day of the year": lambda d: d.dt.dayofyear,
    "Hour": lambda d: d.dt.hour, "Minute": lambda d: d.dt.minute, "Second": lambda d: d.dt.second,
}


@analysis("data.extract_date", menu=MENU, title="Extract from Date/Time",
          params=[Param("column", "Date/time column", "column", numeric=False),
                  Param("part", "Extract", "choice", default="Year", choices=list(DATE_PARTS))])
def extract_date(ws, column, part="Year"):
    values = pd.Series(DATE_PARTS[part](as_dates(ws, column))).astype(float)
    res = Result("Extract from Date/Time")
    res.store(f"{column}_{part.split(' (')[0].replace(' ', '')}", values.tolist())
    return _stored_note(res)
