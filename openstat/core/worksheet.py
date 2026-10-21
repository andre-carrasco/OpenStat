"""Worksheet: a table of named columns, each numeric, text or date/time, with missing values."""
from __future__ import annotations

import copy
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd

from openstat.i18n import language

_TIME = r"(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$"
_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})" + _TIME)
_SLASH_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})" + _TIME)


def parse_date(text) -> pd.Timestamp | None:
    """A timestamp from 'yyyy-mm-dd' or a slash date, each with an optional time; None when it is neither.

    Slash dates are day first in pt-BR and month first in English.
    """
    text = str(text).strip()
    match = _ISO_DATE.match(text)
    if match:
        year, month, day = (int(g) for g in match.groups()[:3])
    else:
        match = _SLASH_DATE.match(text)
        if not match:
            return None
        first, second, year = (int(g) for g in match.groups()[:3])
        day, month = (first, second) if language() == "pt_BR" else (second, first)
    hour, minute, second = (int(g or 0) for g in match.groups()[3:])
    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    except ValueError:
        return None


def parse_cell(text):
    """Convert typed text into None (blank), float, timestamp or str."""
    if text is None:
        return None
    text = str(text).strip()
    if text == "" or text == "*":
        return None
    candidate = text
    if "," in candidate and "." not in candidate:
        candidate = candidate.replace(",", ".")
    try:
        return float(candidate)
    except ValueError:
        date = parse_date(text)
        return text if date is None else date


def format_value(value) -> str:
    if value is None or value is pd.NaT:
        return ""
    if isinstance(value, datetime):
        value = pd.Timestamp(value)
        return value.strftime("%Y-%m-%d" if value == value.normalize() else "%Y-%m-%d %H:%M:%S")
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return str(value)


def coerce_values(values: list) -> pd.Series:
    """A float column when every present value is a number, a date/time column when every one is a
    timestamp, otherwise a text column."""
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, datetime) for v in present):
        return pd.Series(pd.to_datetime(values)).astype("datetime64[ns]")
    if all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in present):
        return pd.Series([np.nan if v is None else float(v) for v in values], dtype=float)
    return pd.Series([None if v is None else format_value(v) for v in values], dtype=object)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Worksheet form: string column names, numeric columns as float, date/time columns without a
    time zone, everything else as text or None."""
    df = df.reset_index(drop=True).copy()
    df.columns = [str(c) for c in df.columns]
    for name in df.columns:
        col = df[name]
        if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            df[name] = col.astype(float)
        elif pd.api.types.is_datetime64_any_dtype(col):
            df[name] = col.dt.tz_localize(None).astype("datetime64[ns]")
        else:
            df[name] = coerce_values([None if pd.isna(v) else v for v in col.tolist()]).to_numpy()
    return df


def changed_parts(before: dict, after: dict) -> tuple[dict, dict]:
    """The two states with `data` cut down to the columns that differ."""
    names = set(before["data"]) | set(after["data"])
    keep = {c for c in names if (c not in before["data"] or c not in after["data"]
                                 or not before["data"][c].equals(after["data"][c]))}
    return ({**before, "data": {c: v for c, v in before["data"].items() if c in keep}},
            {**after, "data": {c: v for c, v in after["data"].items() if c in keep}})


class Worksheet:
    def __init__(self, name: str = "Worksheet 1", df: pd.DataFrame | None = None):
        self.name = name
        self.df = normalize_frame(pd.DataFrame() if df is None else df)
        self.formulas: dict[str, str] = {}          # column name -> expression that fills it
        self.formula_errors: dict[str, str] = {}    # formulas that failed in the last recalculation
        self.formats: dict[str, list[dict]] = {}    # column name -> conditional formatting rules
        self.descriptions: dict[str, str] = {}      # column name -> free-text description
        self.value_orders: dict[str, str | list[str]] = {}   # column name -> "occurrence" or the user's level order

    @property
    def nrows(self) -> int:
        return len(self.df)

    @property
    def ncols(self) -> int:
        return self.df.shape[1]

    def column_names(self) -> list[str]:
        return list(self.df.columns)

    def is_numeric(self, name: str) -> bool:
        return pd.api.types.is_numeric_dtype(self.df[name])

    def is_datetime(self, name: str) -> bool:
        return pd.api.types.is_datetime64_any_dtype(self.df[name])

    def numeric_columns(self) -> list[str]:
        return [c for c in self.df.columns if self.is_numeric(c)]

    def _unique_name(self, base: str) -> str:
        name, k = base, 1
        while name in self.df.columns:
            k += 1
            name = f"{base}_{k}"
        return name

    def ensure_size(self, nrows: int, ncols: int) -> None:
        if nrows > self.nrows:
            self.df = self.df.reindex(range(nrows))
        while self.ncols < ncols:
            name = self._unique_name(f"C{self.ncols + 1}")
            self.df[name] = np.full(self.nrows, np.nan)

    def get_text(self, row: int, col: int) -> str:
        if row >= self.nrows or col >= self.ncols:
            return ""
        return format_value(self.df.iat[row, col])

    def set_cell(self, row: int, col: int, text) -> None:
        value = parse_cell(text)
        if value is None and (row >= self.nrows or col >= self.ncols):
            return
        self.ensure_size(row + 1, col + 1)
        name = self.df.columns[col]
        current = [None if pd.isna(v) else v for v in self.df[name].tolist()]
        # a date typed into a text column stays as it was typed
        current[row] = str(text).strip() if isinstance(value, datetime) else value
        # Re-parse text so a column goes back to numeric or date once its last text cell is gone
        column = coerce_values([parse_cell(v) if isinstance(v, str) else v for v in current])
        self.df[name] = coerce_values(current) if column.dtype == object else column
        self.recalculate()

    def rename_column(self, col: int, new_name: str) -> None:
        new_name = new_name.strip()
        old = self.df.columns[col]
        if not new_name or new_name == old:
            return
        if new_name in self.df.columns:
            raise ValueError(f"Column name already in use: {new_name}")
        self.df = self.df.rename(columns={old: new_name})
        self.formulas = {new_name if k == old else k: v for k, v in self.formulas.items()}
        self.formats = {new_name if k == old else k: v for k, v in self.formats.items()}
        self.descriptions = {new_name if k == old else k: v for k, v in self.descriptions.items()}
        self.value_orders = {new_name if k == old else k: v for k, v in self.value_orders.items()}
        self.recalculate()

    def snapshot(self) -> dict:
        """Copy of everything an edit can change, for undo."""
        return {
            "columns": self.column_names(), "nrows": self.nrows,
            "data": {c: self.df[c].copy() for c in self.df.columns},
            "formulas": copy.deepcopy(self.formulas), "formats": copy.deepcopy(self.formats),
            "descriptions": copy.deepcopy(self.descriptions), "value_orders": copy.deepcopy(self.value_orders),
        }

    def restore(self, state: dict) -> None:
        """Rebuild the worksheet from a snapshot; columns missing from state["data"] keep their current values."""
        n = state["nrows"]
        self.df = pd.DataFrame(
            {c: (state["data"][c] if c in state["data"] else self.df[c]).to_numpy() for c in state["columns"]},
            index=range(n))
        for key in ("formulas", "formats", "descriptions", "value_orders"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.recalculate()

    def add_column(self, name: str, values) -> str:
        """Append a column (padding rows as needed) and return the name actually used."""
        values = list(values)
        name = self._unique_name(name)
        if len(values) > self.nrows:
            self.df = self.df.reindex(range(len(values)))
        padded = values + [None] * (self.nrows - len(values))
        self.df[name] = coerce_values([None if pd.isna(v) else v for v in padded]).values
        self.recalculate()
        return name

    def set_formula(self, name: str, expression: str) -> None:
        """Fill column `name` (created when missing) from a formula; an empty expression detaches it."""
        from openstat.core.expression import evaluate, referenced_columns

        name, expression = name.strip(), expression.strip()
        if not expression:
            self.formulas.pop(name, None)
            self.formula_errors.pop(name, None)
            return
        if not name:
            raise ValueError("Enter the name of the column")
        if name in referenced_columns(self, expression):
            raise ValueError("A formula cannot refer to its own column")
        evaluate(self, expression)   # a formula that cannot be evaluated is rejected here
        if name not in self.df.columns:
            self.df[name] = np.full(self.nrows, np.nan)
        self.formulas[name] = expression
        self.recalculate()

    def recalculate(self) -> None:
        """Refill the formula columns."""
        from openstat.core.expression import evaluate

        self.formulas = {k: v for k, v in self.formulas.items() if k in self.df.columns}
        self.formula_errors = {}
        for _ in self.formulas:   # repeated so that formulas may depend on one another in any order
            self.formula_errors = {}
            for name, expression in self.formulas.items():
                try:
                    value = evaluate(self, expression)
                except Exception as exc:   # e.g. a column it used was renamed; the old values stay
                    self.formula_errors[name] = str(exc)
                    continue
                values = np.full(self.nrows, value).tolist() if np.ndim(value) == 0 else np.asarray(value).tolist()
                if len(values) != self.nrows:
                    self.formula_errors[name] = "The formula must give one value per row"
                    continue
                values = [None if pd.isna(v) else float(v) if isinstance(v, bool) else v for v in values]
                self.df[name] = coerce_values(values).to_numpy()

    def data(self, columns: list[str], dropna: bool = True) -> pd.DataFrame:
        """Selected columns, optionally keeping only complete rows."""
        frame = self.df[list(columns)]
        return frame.dropna() if dropna else frame.copy()

    def column(self, name: str) -> pd.Series:
        """One column without its missing values."""
        return self.df[name].dropna()
