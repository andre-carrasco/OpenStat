"""Import and export of worksheet data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from openstat.core.worksheet import Worksheet, parse_date

IMPORT_FILTER = "Data (*.csv *.txt *.tsv *.xlsx *.xls);;All files (*)"


def import_file(path: str | Path) -> list[Worksheet]:
    """Read a data file; Excel workbooks yield one worksheet per sheet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        return [Worksheet(str(name), df) for name, df in sheets.items()]
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
                    continue
                dates = pd.to_datetime([None if pd.isna(v) else parse_date(v) for v in df[name]])
                if df[name].notna().any() and dates.notna().sum() == df[name].notna().sum():
                    df[name] = dates
    return [Worksheet(path.stem, df)]


def export_csv(ws: Worksheet, path: str | Path) -> None:
    ws.df.to_csv(path, index=False)
