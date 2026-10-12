"""Design matrices for linear models: continuous predictors, coded factors and interactions."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from openstat.core.worksheet import format_value

CONSTANT = "Constant"


@dataclass
class Design:
    X: pd.DataFrame                      # model matrix, first column is the constant
    terms: list[tuple[str, list[str]]]   # (term name, columns of X belonging to it)
    rows: np.ndarray                     # worksheet row positions used (complete cases)
    nrows: int                           # worksheet row count, for storing aligned columns
    levels: dict[str, list[str]] = field(default_factory=dict)

    def aligned(self, values) -> list:
        """Spread per-observation values back onto worksheet rows (missing elsewhere)."""
        out = np.full(self.nrows, np.nan)
        out[self.rows] = np.asarray(values, dtype=float)
        return out.tolist()


def complete_rows(ws, columns) -> np.ndarray:
    return np.flatnonzero(ws.df[list(columns)].notna().all(axis=1).to_numpy())


def code_factor(labels: pd.Series, name: str, coding: str) -> tuple[pd.DataFrame, list[str]]:
    """Indicator columns for a factor.

    'reference' drops the first level (0/1 coding); 'effect' drops the last level and codes it -1,
    so that each term can be tested with the other terms held at their average level.
    """
    levels = sorted(labels.unique())
    if len(levels) < 2:
        raise ValueError(f"{name}: a factor needs at least two levels")
    if coding == "reference":
        cols = {f"{name}_{lv}": (labels == lv).astype(float) for lv in levels[1:]}
    else:
        last = labels == levels[-1]
        cols = {f"{name}_{lv}": (labels == lv).astype(float) - last.astype(float) for lv in levels[:-1]}
    return pd.DataFrame(cols, index=labels.index), levels


def build_design(ws, responses, continuous=(), categorical=(), interactions=False, coding="reference",
                 extra_columns=()) -> Design:
    continuous, categorical = list(continuous or []), list(categorical or [])
    overlap = set(continuous) & set(categorical)
    if overlap:
        raise ValueError(f"A column cannot be both continuous and categorical: {', '.join(sorted(overlap))}")
    if set(responses) & (set(continuous) | set(categorical)):
        raise ValueError("The response cannot also be a predictor")
    if not continuous and not categorical:
        raise ValueError("Select at least one predictor")
    used = list(responses) + continuous + categorical + list(extra_columns)
    rows = complete_rows(ws, used)
    frame = ws.df.iloc[rows]
    X = pd.DataFrame({CONSTANT: np.ones(len(rows))}, index=frame.index)
    terms: list[tuple[str, list[str]]] = []
    blocks: dict[str, pd.DataFrame] = {}
    levels: dict[str, list[str]] = {}
    for name in continuous:
        blocks[name] = pd.DataFrame({name: frame[name].astype(float)})
    for name in categorical:
        blocks[name], levels[name] = code_factor(frame[name].map(format_value), name, coding)
    for name, block in blocks.items():
        X = pd.concat([X, block], axis=1)
        terms.append((name, list(block.columns)))
    if interactions:
        for a, b in combinations(blocks, 2):
            cols = {f"{ca}*{cb}": blocks[a][ca] * blocks[b][cb] for ca in blocks[a] for cb in blocks[b]}
            block = pd.DataFrame(cols, index=frame.index)
            X = pd.concat([X, block], axis=1)
            terms.append((f"{a}*{b}", list(block.columns)))
    if len(rows) <= X.shape[1]:
        raise ValueError("Not enough observations to fit the model")
    return Design(X, terms, rows, ws.nrows, levels)
