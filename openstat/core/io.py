"""Import and export of worksheet data."""
from __future__ import annotations

import warnings
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


SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")


def import_sql(connection: str, query: str) -> Worksheet:
    """Run a query and return its rows: `connection` is the path of an SQLite file or an ODBC
    connection string (which needs the optional pyodbc package)."""
    connection, query = connection.strip(), query.strip()
    if not connection or not query:
        raise ValueError("Enter the connection and the query")
    if connection.lower().endswith(SQLITE_SUFFIXES):
        import sqlite3

        if not Path(connection).is_file():
            raise ValueError(f"File not found: {connection}")
        link = sqlite3.connect(f"{Path(connection).resolve().as_uri()}?mode=ro", uri=True)   # read only
    else:
        try:
            import pyodbc
        except ImportError:
            raise ValueError("ODBC connections need the pyodbc package (pip install pyodbc)") from None
        link = pyodbc.connect(connection, readonly=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # pandas prefers SQLAlchemy but works with DBAPI
            frame = pd.read_sql_query(query, link)
    finally:
        link.close()
    return Worksheet("Query", frame)


def export_csv(ws: Worksheet, path: str | Path) -> None:
    ws.df.to_csv(path, index=False)
