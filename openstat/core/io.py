"""Import and export of worksheet data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from openstat.core.worksheet import Worksheet

IMPORT_FILTER = "Data (*.csv *.txt *.tsv *.xlsx *.xls);;All files (*)"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for name in df.columns:
        col = df[name]
        if pd.api.types.is_bool_dtype(col):
            df[name] = col.astype(float)
        elif pd.api.types.is_numeric_dtype(col):
            df[name] = col.astype(float)
        else:
            df[name] = col.map(lambda v: None if pd.isna(v) else str(v)).astype(object)
    return df


def import_file(path: str | Path) -> list[Worksheet]:
    """Read a data file; Excel workbooks yield one worksheet per sheet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        return [Worksheet(str(name), _normalize(df)) for name, df in sheets.items()]
    # sep=None lets pandas sniff comma, semicolon or tab
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    # Semicolon-separated files usually use decimal commas
    if any(df[c].dtype == object for c in df.columns):
        for name in df.columns:
            if df[name].dtype == object:
                converted = pd.to_numeric(df[name].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                if converted.notna().sum() == df[name].notna().sum():
                    df[name] = converted
    return [Worksheet(path.stem, _normalize(df))]


def export_csv(ws: Worksheet, path: str | Path) -> None:
    ws.df.to_csv(path, index=False)
