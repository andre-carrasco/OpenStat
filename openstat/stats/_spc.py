"""Statistical process control helpers: unbiasing constants, sigma estimates, run rules and chart drawing."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import integrate, special
from scipy import stats as sps

from openstat.core.registry import Param
from openstat.core.worksheet import format_value
from openstat.i18n import tr

BLUE, RED, GREEN, GRAY = "#5b9bd5", "#c00000", "#00803c", "#7f7f7f"

MEASUREMENTS = Param("column", "Measurements", "column")
SUBGROUP_SIZE = Param("subgroup_size", "Subgroup size", "int", default=5, minimum=1, maximum=10000)
SUBGROUP_COLUMN = Param("subgroup_col", "Subgroup IDs (instead of size)", "column", numeric=False, optional=True)
ALL_TESTS = Param("all_tests", "Perform all tests for special causes", "bool", default=False)


def c4(n: int) -> float:
    """E[s] / sigma for a normal sample of size n."""
    if n < 2:
        return float("nan")
    return float(np.sqrt(2 / (n - 1)) * np.exp(special.gammaln(n / 2) - special.gammaln((n - 1) / 2)))


@lru_cache(maxsize=None)
def d2(n: int) -> float:
    """E[range] / sigma for a normal sample of size n."""
    if n < 2:
        return float("nan")
    value, _ = integrate.quad(lambda x: 1 - sps.norm.cdf(x) ** n - sps.norm.sf(x) ** n, -10, 10, limit=200)
    return float(value)


@lru_cache(maxsize=None)
def d3(n: int) -> float:
    """SD[range] / sigma for a normal sample of size n."""
    if n < 2:
        return float("nan")

    def integrand(x, y):
        fy, fx = sps.norm.cdf(y), sps.norm.cdf(x)
        return 1 - fy ** n - (1 - fx) ** n + (fy - fx) ** n

    second, _ = integrate.dblquad(integrand, -9, 9, -9, lambda y: y, epsabs=1e-9, epsrel=1e-9)
    return float(np.sqrt(max(2 * second - d2(n) ** 2, 0.0)))


def make_subgroups(ws, column: str, size: int = 5, subgroup_col: str | None = None) -> list[np.ndarray]:
    """Split a column into subgroups by an ID column or by consecutive blocks of `size` rows."""
    if subgroup_col:
        frame = ws.df[[column, subgroup_col]].dropna()
        labels = frame[subgroup_col].map(format_value)
        order = list(dict.fromkeys(labels))  # order of first appearance
        return [frame[column][labels == level].to_numpy(dtype=float) for level in order]
    x = ws.df[column].dropna().to_numpy(dtype=float)
    count = len(x) // size
    if count < 1:
        raise ValueError("There is not enough data for one subgroup")
    return [x[i * size:(i + 1) * size] for i in range(count)]


def sigma_rbar(groups: list[np.ndarray]) -> float:
    estimates = [np.ptp(g) / d2(len(g)) for g in groups if len(g) > 1]
    if not estimates:
        raise ValueError("Subgroups need at least two observations; use an individuals chart")
    return float(np.mean(estimates))


def sigma_sbar(groups: list[np.ndarray]) -> float:
    estimates = [g.std(ddof=1) / c4(len(g)) for g in groups if len(g) > 1]
    if not estimates:
        raise ValueError("Subgroups need at least two observations; use an individuals chart")
    return float(np.mean(estimates))


def sigma_pooled(groups: list[np.ndarray]) -> float:
    """Pooled standard deviation with the c4 unbiasing constant."""
    df = sum(len(g) - 1 for g in groups)
    if df < 1:
        raise ValueError("Subgroups need at least two observations; use an individuals chart")
    pooled = np.sqrt(sum(((g - g.mean()) ** 2).sum() for g in groups) / df)
    return float(pooled / c4(df + 1))


def sigma_moving_range(x, span: int = 2) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < span:
        raise ValueError("There is not enough data to estimate the variation")
    ranges = np.array([np.ptp(x[i:i + span]) for i in range(len(x) - span + 1)])
    return float(ranges.mean() / d2(span))


def within_sigma(groups: list[np.ndarray]) -> float:
    """Short-term sigma: pooled SD for subgroups, average moving range for individuals."""
    if all(len(g) == 1 for g in groups):
        return sigma_moving_range(np.concatenate(groups))
    return sigma_pooled(groups)


def special_cause_tests(z, tests=(1,)) -> dict[int, list[int]]:
    """Indices failing the eight classical run rules, on standardized values z = (x - center) / sigma."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    out: dict[int, list[int]] = {}

    def run(flags: np.ndarray, window: int, needed: int) -> list[int]:
        hits = []
        for i in range(window - 1, n):
            if flags[i - window + 1:i + 1].sum() >= needed:
                hits.append(i)
        return hits

    side = np.sign(z)
    if 1 in tests:
        out[1] = list(np.flatnonzero(np.abs(z) > 3))
    if 2 in tests:
        out[2] = sorted(set(run(side > 0, 9, 9)) | set(run(side < 0, 9, 9)))
    if 3 in tests:
        diff = np.sign(np.diff(z))
        out[3] = [i for i in range(5, n) if abs(diff[i - 5:i].sum()) == 5]  # six points = five steps
    if 4 in tests:
        diff = np.sign(np.diff(z))
        out[4] = [i for i in range(13, n)
                  if np.all(diff[i - 13:i] != 0) and np.all(diff[i - 13:i - 1] * diff[i - 12:i] < 0)]
    if 5 in tests:
        out[5] = sorted({i for i in set(run(z > 2, 3, 2)) if z[i] > 2} | {i for i in set(run(z < -2, 3, 2)) if z[i] < -2})
    if 6 in tests:
        out[6] = sorted({i for i in set(run(z > 1, 5, 4)) if z[i] > 1} | {i for i in set(run(z < -1, 5, 4)) if z[i] < -1})
    if 7 in tests:
        out[7] = run(np.abs(z) < 1, 15, 15)
    if 8 in tests:
        out[8] = run(np.abs(z) > 1, 8, 8)
    return {k: [int(i) for i in v] for k, v in out.items() if len(v)}


TEST_DESCRIPTIONS = {
    1: "1 point more than 3 standard deviations from the center line",
    2: "9 points in a row on the same side of the center line",
    3: "6 points in a row, all increasing or all decreasing",
    4: "14 points in a row, alternating up and down",
    5: "2 out of 3 points more than 2 standard deviations from the center line (same side)",
    6: "4 out of 5 points more than 1 standard deviation from the center line (same side)",
    7: "15 points in a row within 1 standard deviation of the center line",
    8: "8 points in a row more than 1 standard deviation from the center line",
}


def draw_chart(ax, values, center, lcl, ucl, ylabel: str, flagged=(), labels=None) -> list[int]:
    """Plot a control chart panel; returns the indices outside the limits or flagged by run rules."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    x = np.arange(1, n + 1)
    center, lcl, ucl = (np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in (center, lcl, ucl))
    outside = (values > ucl + 1e-12) | (values < lcl - 1e-12)
    bad = sorted(set(np.flatnonzero(outside).tolist()) | {int(i) for i in flagged})
    ax.plot(x, values, "-o", color=BLUE, markersize=4, linewidth=1)
    if bad:
        ax.plot(x[bad], values[bad], "s", color=RED, markersize=5)
    ax.step(x, center, where="mid", color=GREEN, linewidth=1)
    ax.step(x, ucl, where="mid", color=RED, linewidth=1)
    ax.step(x, lcl, where="mid", color=RED, linewidth=1)
    for name, line in (("UCL", ucl), ("CL", center), ("LCL", lcl)):
        if not np.isnan(line[-1]):
            ax.annotate(f"{name}={line[-1]:.4g}", (1.005, line[-1]), xycoords=("axes fraction", "data"),
                        fontsize=7, va="center")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, n + 0.5)
    if labels is not None and n <= 30:
        ax.set_xticks(x, labels, rotation=45, fontsize=7)
    return bad


def limits_table(rows: dict) -> pd.DataFrame:
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Chart"
    return table


def report_signals(res, name: str, signals: dict[int, list[int]], outside: list[int]) -> None:
    """Describe out-of-control points (1-based positions) in the output."""
    if not signals and not outside:
        return
    res.heading(f"{tr('Test results for')} {name}")
    if outside and 1 not in signals:
        res.text(f"{tr('Points outside the control limits')}: {', '.join(str(i + 1) for i in outside)}")
    for test, points in signals.items():
        res.text(f"{tr('TEST')} {test}. {tr(TEST_DESCRIPTIONS[test])}: {', '.join(str(i + 1) for i in points)}")
