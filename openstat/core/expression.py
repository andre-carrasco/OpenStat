"""Formula evaluator for the calculator: column arithmetic without arbitrary code execution."""
from __future__ import annotations

import ast
import operator

import numpy as np
import pandas as pd
from scipy import stats as sps

from openstat.core.worksheet import Worksheet


class ExpressionError(ValueError):
    pass


def _num(x):
    return np.asarray(x, dtype=float)


def _lag(x, k=1):
    return pd.Series(_num(x)).shift(int(k)).to_numpy()


FUNCTIONS = {
    "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "ln": np.log, "log": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "round": lambda x, d=0: np.round(_num(x), int(d)), "floor": np.floor, "ceil": np.ceil,
    "sign": np.sign,
    "mean": lambda x: float(np.nanmean(_num(x))),
    "median": lambda x: float(np.nanmedian(_num(x))),
    "sum": lambda x: float(np.nansum(_num(x))),
    "stdev": lambda x: float(np.nanstd(_num(x), ddof=1)),
    "var": lambda x: float(np.nanvar(_num(x), ddof=1)),
    "min": lambda x: float(np.nanmin(_num(x))),
    "max": lambda x: float(np.nanmax(_num(x))),
    "count": lambda x: float(np.sum(~np.isnan(_num(x)))),
    "cumsum": lambda x: np.nancumsum(_num(x)),
    "lag": _lag,
    "rank": lambda x: pd.Series(_num(x)).rank().to_numpy(),
    "where": lambda cond, a, b: np.where(np.asarray(cond, dtype=bool), a, b),
    "normcdf": lambda x, m=0.0, s=1.0: sps.norm.cdf(_num(x), m, s),
    "norminv": lambda p, m=0.0, s=1.0: sps.norm.ppf(_num(p), m, s),
}

CONSTANTS = {"pi": np.pi, "e": np.e}

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.BitXor: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}
_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _resolve(ws: Worksheet, name: str):
    names = ws.column_names()
    if name in names:
        return ws.df[name].to_numpy()
    lowered = {n.lower(): n for n in names}
    if name.lower() in lowered:
        return ws.df[lowered[name.lower()]].to_numpy()
    if name[:1] in "Cc" and name[1:].isdigit() and 1 <= int(name[1:]) <= len(names):
        return ws.df.iloc[:, int(name[1:]) - 1].to_numpy()
    if name.lower() in CONSTANTS:
        return CONSTANTS[name.lower()]
    raise ExpressionError(f"Unknown name: {name}")


def _eval(node, ws):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool):
            return node.value
        raise ExpressionError("Unsupported constant")
    if isinstance(node, ast.Name):
        return _resolve(ws, node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left, ws), _eval(node.right, ws))
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, ws)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.Not):
            return np.logical_not(value)
    if isinstance(node, ast.BoolOp):
        values = [np.asarray(_eval(v, ws), dtype=bool) for v in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = values[0]
        for v in values[1:]:
            out = combine(out, v)
        return out
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ws)
        out = None
        for op, right_node in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE:
                raise ExpressionError("Unsupported comparison")
            right = _eval(right_node, ws)
            part = _COMPARE[type(op)](left, right)
            out = part if out is None else np.logical_and(out, part)
            left = right
        return out
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = FUNCTIONS.get(node.func.id.lower())
        if func is None:
            raise ExpressionError(f"Unknown function: {node.func.id}")
        return func(*[_eval(a, ws) for a in node.args])
    raise ExpressionError("Unsupported expression")


def evaluate(ws: Worksheet, expression: str):
    """Evaluate `expression` against the worksheet; returns an array or a scalar."""
    expression = expression.strip()
    if not expression:
        raise ExpressionError("Empty expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression: {exc.msg}") from exc
    with np.errstate(all="ignore"):
        result = _eval(tree.body, ws)
    if isinstance(result, np.ndarray) and result.dtype == bool:
        return result
    return result
