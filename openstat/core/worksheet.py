"""Worksheet: a table of named columns, each numeric or text, with missing values."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def parse_cell(text):
    """Convert typed text into None (blank), float or str."""
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
        return text


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if value is pd.NaT:
        return ""
    return str(value)


class Worksheet:
    def __init__(self, name: str = "Worksheet 1", df: pd.DataFrame | None = None):
        self.name = name
        if df is None:
            df = pd.DataFrame()
        df = df.reset_index(drop=True).copy()
        df.columns = [str(c) for c in df.columns]
        self.df = df

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
        # Re-parse text so a column goes back to numeric once its last text cell is gone
        values = [None if pd.isna(v) else parse_cell(v) if isinstance(v, str) else v
                  for v in self.df[name].tolist()]
        values[row] = value
        self.df[name] = self._coerce(values)

    @staticmethod
    def _coerce(values: list) -> pd.Series:
        present = [v for v in values if v is not None]
        if all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in present):
            return pd.Series([np.nan if v is None else float(v) for v in values], dtype=float)
        return pd.Series([None if v is None else format_value(v) for v in values], dtype=object)

    def rename_column(self, col: int, new_name: str) -> None:
        new_name = new_name.strip()
        old = self.df.columns[col]
        if not new_name or new_name == old:
            return
        if new_name in self.df.columns:
            raise ValueError(f"Column name already in use: {new_name}")
        self.df = self.df.rename(columns={old: new_name})

    def add_column(self, name: str, values) -> str:
        """Append a column (padding rows as needed) and return the name actually used."""
        values = list(values)
        name = self._unique_name(name)
        if len(values) > self.nrows:
            self.df = self.df.reindex(range(len(values)))
        padded = values + [None] * (self.nrows - len(values))
        self.df[name] = self._coerce([None if pd.isna(v) else v for v in padded]).values
        return name

    def data(self, columns: list[str], dropna: bool = True) -> pd.DataFrame:
        """Selected columns, optionally keeping only complete rows."""
        frame = self.df[list(columns)]
        return frame.dropna() if dropna else frame.copy()

    def column(self, name: str) -> pd.Series:
        """One column without its missing values."""
        return self.df[name].dropna()
