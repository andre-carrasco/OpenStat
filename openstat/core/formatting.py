"""Conditional formatting: rules that highlight the cells of a column."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openstat.core.worksheet import Worksheet, format_value, parse_cell

GREATER, LESS, EQUAL, NOT_EQUAL, MISSING, OUTLIER = (
    "Greater than", "Less than", "Equal to", "Not equal to", "Missing", "Outlier (1.5 x IQR beyond the quartiles)")
RULES = [GREATER, LESS, EQUAL, NOT_EQUAL, MISSING, OUTLIER]
NEEDS_VALUE = (GREATER, LESS, EQUAL, NOT_EQUAL)
COLORS = {"Red": "#ffc7ce", "Yellow": "#ffeb9c", "Green": "#c6efce", "Blue": "#bdd7ee"}


def make_rule(kind: str, value: str = "", color: str = "Red") -> dict:
    """A rule as stored in the worksheet; the value is kept as typed."""
    if kind not in RULES or color not in COLORS:
        raise ValueError("Unknown formatting rule")
    if kind in NEEDS_VALUE and not str(value).strip():
        raise ValueError("Enter a value for the rule")
    return {"rule": kind, "value": str(value).strip(), "color": color}


def rule_mask(column: pd.Series, rule: dict) -> np.ndarray:
    """True for the cells of `column` that the rule highlights."""
    kind = rule["rule"]
    missing = column.isna().to_numpy()
    if kind == MISSING:
        return missing
    if kind == OUTLIER:
        if not pd.api.types.is_numeric_dtype(column):
            return np.zeros(len(column), dtype=bool)
        x = column.to_numpy(dtype=float)
        if (~missing).sum() < 4:
            return np.zeros(len(column), dtype=bool)
        q1, q3 = np.nanquantile(x, [0.25, 0.75], method="weibull")
        with np.errstate(invalid="ignore"):
            return (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))
    target = parse_cell(rule["value"])
    if kind in (EQUAL, NOT_EQUAL):
        same = column.map(format_value).to_numpy() == format_value(target)
        return (same if kind == EQUAL else ~same) & ~missing
    values = column.tolist()
    out = np.zeros(len(column), dtype=bool)
    for i, value in enumerate(values):
        if missing[i] or isinstance(value, str) != isinstance(target, str):
            continue   # a number is never greater or less than a text
        try:
            out[i] = value > target if kind == GREATER else value < target
        except TypeError:
            continue   # e.g. a date against a number
    return out


def column_colors(ws: Worksheet, name: str) -> list[str | None]:
    """Background color of each cell of a column; where rules overlap the last one wins."""
    colors: list[str | None] = [None] * ws.nrows
    for rule in ws.formats.get(name, []):
        for i in np.flatnonzero(rule_mask(ws.df[name], rule)):
            colors[i] = COLORS[rule["color"]]
    return colors
