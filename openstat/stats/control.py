"""Control charts: variables, attributes, time-weighted, multivariate and rare-event charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats import _spc
from openstat.stats._spc import (ALL_TESTS, MEASUREMENTS, SUBGROUP_COLUMN, SUBGROUP_SIZE, c4, d2, d3, draw_chart,
                                 limits_table, make_subgroups, report_signals, special_cause_tests)
from openstat.stats._util import clean

MENU = ("Stat", "Control Charts")
VARIABLES_MENU = MENU + ("Variables Charts",)
ATTRIBUTES_MENU = MENU + ("Attributes Charts",)
WEIGHTED_MENU = MENU + ("Time-Weighted Charts",)
OTHER_MENU = MENU + ("Multivariate and Rare Event Charts",)

ALL = tuple(range(1, 9))


def _two_panel(title: str):
    fig = new_figure(8.0, 6.0)
    top, bottom = fig.subplots(2, 1, sharex=True)
    fig.suptitle(title)
    bottom.set_xlabel(tr("Sample"))
    return fig, top, bottom


def _one_panel(title: str, xlabel: str = "Sample"):
    fig = new_figure(8.0, 4.0)
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_xlabel(tr(xlabel))
    return fig, ax


def xbar_limits(groups, sigma) -> dict:
    sizes = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    center = float(np.concatenate(groups).mean())
    half = 3 * sigma / np.sqrt(sizes)
    return {"values": means, "center": center, "lcl": center - half, "ucl": center + half,
            "z": (means - center) / (sigma / np.sqrt(sizes))}


def range_limits(groups, sigma) -> dict:
    sizes = [len(g) for g in groups]
    center = np.array([d2(n) * sigma for n in sizes])
    spread = np.array([3 * d3(n) * sigma for n in sizes])
    return {"values": np.array([np.ptp(g) for g in groups]), "center": center,
            "lcl": np.maximum(center - spread, 0.0), "ucl": center + spread}


def sd_limits(groups, sigma) -> dict:
    sizes = [len(g) for g in groups]
    center = np.array([c4(n) * sigma for n in sizes])
    spread = np.array([3 * sigma * np.sqrt(1 - c4(n) ** 2) for n in sizes])
    return {"values": np.array([g.std(ddof=1) for g in groups]), "center": center,
            "lcl": np.maximum(center - spread, 0.0), "ucl": center + spread}


def _last(v):
    return float(np.atleast_1d(v)[-1])


def _subgroup_chart(ws, column, subgroup_size, subgroup_col, all_tests, kind: str) -> Result:
    groups = make_subgroups(ws, column, subgroup_size, subgroup_col)
    if any(len(g) < 2 for g in groups):
        raise ValueError("Subgroups need at least two observations; use an individuals chart")
    sigma = _spc.sigma_rbar(groups) if kind == "R" else _spc.sigma_sbar(groups)
    xbar = xbar_limits(groups, sigma)
    spread = range_limits(groups, sigma) if kind == "R" else sd_limits(groups, sigma)
    title = f"Xbar-{kind} Chart"
    signals = special_cause_tests(xbar["z"], ALL if all_tests else (1,))
    fig, top, bottom = _two_panel(f"{tr(title)}: {column}")
    flagged = sorted({i for v in signals.values() for i in v})
    out_mean = draw_chart(top, xbar["values"], xbar["center"], xbar["lcl"], xbar["ucl"], tr("Sample mean"), flagged)
    spread_label = tr("Sample range") if kind == "R" else tr("Sample StDev")
    out_spread = draw_chart(bottom, spread["values"], spread["center"], spread["lcl"], spread["ucl"], spread_label)
    res = Result(title)
    res.text(f"{tr('Subgroups')}: {len(groups)}    {tr('Estimated standard deviation')}: {sigma:.6g}")
    res.table(limits_table({
        "Xbar": {"LCL": _last(xbar["lcl"]), "CL": xbar["center"], "UCL": _last(xbar["ucl"])},
        kind: {"LCL": _last(spread["lcl"]), "CL": _last(spread["center"]), "UCL": _last(spread["ucl"])}}),
        title="Control limits (last subgroup)")
    res.figure(fig)
    report_signals(res, "Xbar", signals, out_mean)
    report_signals(res, kind, {}, out_spread)
    return res


SUBGROUP_PARAMS = [MEASUREMENTS, SUBGROUP_SIZE, SUBGROUP_COLUMN, ALL_TESTS]


@analysis("cc.xbar_r", menu=VARIABLES_MENU, title="Xbar-R Chart", params=SUBGROUP_PARAMS)
def xbar_r(ws, column, subgroup_size=5, subgroup_col=None, all_tests=False):
    return _subgroup_chart(ws, column, subgroup_size, subgroup_col, all_tests, "R")


@analysis("cc.xbar_s", menu=VARIABLES_MENU, title="Xbar-S Chart", params=SUBGROUP_PARAMS)
def xbar_s(ws, column, subgroup_size=5, subgroup_col=None, all_tests=False):
    return _subgroup_chart(ws, column, subgroup_size, subgroup_col, all_tests, "S")


def _single_chart(ws, column, subgroup_size, subgroup_col, all_tests, kind: str) -> Result:
    """One of the charts that usually come in pairs: subgroup means, ranges or standard deviations."""
    groups = make_subgroups(ws, column, subgroup_size, subgroup_col)
    if any(len(g) < 2 for g in groups):
        raise ValueError("Subgroups need at least two observations; use an individuals chart")
    title = f"{kind} Chart"
    if kind == "Xbar":
        sigma = _spc.sigma_pooled(groups)
        lim = xbar_limits(groups, sigma)
        signals = special_cause_tests(lim["z"], ALL if all_tests else (1,))
        label = tr("Sample mean")
    else:
        sigma = _spc.sigma_rbar(groups) if kind == "R" else _spc.sigma_sbar(groups)
        lim = range_limits(groups, sigma) if kind == "R" else sd_limits(groups, sigma)
        signals = {}
        label = tr("Sample range") if kind == "R" else tr("Sample StDev")
    fig, ax = _one_panel(f"{tr(title)}: {column}")
    flagged = sorted({i for v in signals.values() for i in v})
    outside = draw_chart(ax, lim["values"], lim["center"], lim["lcl"], lim["ucl"], label, flagged)
    res = Result(title)
    res.text(f"{tr('Subgroups')}: {len(groups)}    {tr('Estimated standard deviation')}: {sigma:.6g}")
    res.table(limits_table({kind: {"LCL": _last(lim["lcl"]), "CL": _last(lim["center"]), "UCL": _last(lim["ucl"])}}),
              title="Control limits (last subgroup)")
    res.figure(fig)
    report_signals(res, kind, signals, outside)
    return res


@analysis("cc.xbar", menu=VARIABLES_MENU, title="Xbar Chart", params=SUBGROUP_PARAMS)
def xbar_chart(ws, column, subgroup_size=5, subgroup_col=None, all_tests=False):
    return _single_chart(ws, column, subgroup_size, subgroup_col, all_tests, "Xbar")


@analysis("cc.r", menu=VARIABLES_MENU, title="R Chart", params=SUBGROUP_PARAMS[:3])
def r_chart(ws, column, subgroup_size=5, subgroup_col=None):
    return _single_chart(ws, column, subgroup_size, subgroup_col, False, "R")


@analysis("cc.s", menu=VARIABLES_MENU, title="S Chart", params=SUBGROUP_PARAMS[:3])
def s_chart(ws, column, subgroup_size=5, subgroup_col=None):
    return _single_chart(ws, column, subgroup_size, subgroup_col, False, "S")


def between_within(groups) -> dict:
    """Standard deviations within subgroups, between subgroups and of the two together.

    The moving range of the subgroup means measures both sources; what the within-subgroup
    variation explains of it is taken out to leave the between-subgroup part.
    """
    if any(len(g) < 2 for g in groups) or len(groups) < 2:
        raise ValueError("At least two subgroups with two or more observations are needed")
    within = _spc.sigma_pooled(groups)
    means = np.array([g.mean() for g in groups])
    of_means = _spc.sigma_moving_range(means)
    size = len(groups) / sum(1 / len(g) for g in groups)   # harmonic mean of the subgroup sizes
    between = float(np.sqrt(max(of_means ** 2 - within ** 2 / size, 0.0)))
    return {"within": within, "between": between, "total": float(np.sqrt(within ** 2 + between ** 2)),
            "means": means, "of_means": of_means}


@analysis("cc.imr_rs", menu=VARIABLES_MENU, title="I-MR-R/S (Between/Within) Chart",
          params=[MEASUREMENTS, SUBGROUP_SIZE, SUBGROUP_COLUMN])
def imr_rs(ws, column, subgroup_size=5, subgroup_col=None):
    groups = make_subgroups(ws, column, subgroup_size, subgroup_col)
    parts = between_within(groups)
    lim = imr_limits(parts["means"])
    kind = "R" if max(len(g) for g in groups) <= 8 else "S"
    spread = range_limits(groups, parts["within"]) if kind == "R" else sd_limits(groups, parts["within"])
    fig = new_figure(8.0, 8.0)
    top, middle, bottom = fig.subplots(3, 1, sharex=True)
    fig.suptitle(f"{tr('I-MR-R/S (Between/Within) Chart')}: {column}")
    bottom.set_xlabel(tr("Sample"))
    out_i = draw_chart(top, parts["means"], lim["center"], lim["lcl"], lim["ucl"], tr("Subgroup mean"))
    out_mr = draw_chart(middle, lim["mr"], lim["mr_center"], lim["mr_lcl"], lim["mr_ucl"],
                        tr("Moving range of the means"))
    out_w = draw_chart(bottom, spread["values"], spread["center"], spread["lcl"], spread["ucl"],
                       tr("Sample range") if kind == "R" else tr("Sample StDev"))
    res = Result("I-MR-R/S (Between/Within) Chart")
    res.table(pd.DataFrame({"StDev": [parts["between"], parts["within"], parts["total"]]},
                           index=pd.Index(["Between", "Within", "Between/Within"], name="Source")),
              title="Estimated standard deviations", digits=5)
    res.table(limits_table({
        "I": {"LCL": lim["lcl"], "CL": lim["center"], "UCL": lim["ucl"]},
        "MR": {"LCL": lim["mr_lcl"], "CL": lim["mr_center"], "UCL": lim["mr_ucl"]},
        kind: {"LCL": _last(spread["lcl"]), "CL": _last(spread["center"]), "UCL": _last(spread["ucl"])}}),
        title="Control limits (last subgroup)")
    res.figure(fig)
    for name, outside in (("I", out_i), ("MR", out_mr), (kind, out_w)):
        report_signals(res, name, {}, outside)
    return res


def standardize_by_part(x, parts, pooled=False) -> tuple[np.ndarray, pd.DataFrame]:
    """Z-scores of short-run data: each part's own mean, and its own or the pooled moving-range sigma."""
    x, parts = np.asarray(x, dtype=float), np.asarray(parts)
    names = list(dict.fromkeys(parts))
    ranges = {name: np.abs(np.diff(x[parts == name])) for name in names}
    every = np.concatenate(list(ranges.values()))
    if not len(every) or every.mean() == 0:
        raise ValueError("There is not enough data to estimate the variation")
    rows, z = {}, np.empty(len(x))
    for name in names:
        inside = parts == name
        own = ranges[name]
        if not pooled and (not len(own) or own.mean() == 0):
            raise ValueError(f"{name}: the part needs at least two different observations, or pool the standard deviation")
        sigma = (every if pooled else own).mean() / d2(2)
        rows[name] = {"N": int(inside.sum()), "Mean": float(x[inside].mean()), "StDev": float(sigma)}
        z[inside] = (x[inside] - x[inside].mean()) / sigma
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Part"
    return z, table


@analysis("cc.zmr", menu=VARIABLES_MENU, title="Z-MR Chart",
          params=[MEASUREMENTS, Param("part", "Part or run indicator", "column", numeric=False),
                  Param("pooled", "Pool the standard deviation over all parts", "bool", default=False), ALL_TESTS])
def zmr(ws, column, part, pooled=False, all_tests=False):
    frame = ws.df[[column, part]].dropna()
    labels = frame[part].map(_spc.format_value).to_numpy()
    z, table = standardize_by_part(frame[column].to_numpy(dtype=float), labels, pooled)
    mr = np.concatenate([[np.nan], np.abs(np.diff(z))])
    mr[1:][labels[1:] != labels[:-1]] = np.nan   # no moving range across a change of part
    signals = special_cause_tests(z, ALL if all_tests else (1,))
    fig, top, bottom = _two_panel(f"{tr('Z-MR Chart')}: {column}")
    bottom.set_xlabel(tr("Observation"))
    flagged = sorted({i for v in signals.values() for i in v})
    out_z = draw_chart(top, z, 0.0, -3.0, 3.0, tr("Standardized value"), flagged)
    out_mr = draw_chart(bottom, mr, d2(2), 0.0, d2(2) + 3 * d3(2), tr("Moving range"))
    for start in np.flatnonzero(labels[1:] != labels[:-1]):
        top.axvline(start + 1.5, color=_spc.GRAY, linestyle=":", linewidth=0.8)
    res = Result("Z-MR Chart")
    res.table(table, title="Parts", digits=5)
    res.table(limits_table({"Z": {"LCL": -3.0, "CL": 0.0, "UCL": 3.0},
                            "MR": {"LCL": 0.0, "CL": d2(2), "UCL": d2(2) + 3 * d3(2)}}), title="Control limits")
    res.figure(fig)
    report_signals(res, "Z", signals, out_z)
    report_signals(res, "MR", {}, out_mr)
    return res


ZONE_WEIGHTS = (0, 2, 4, 8)   # within 1, 2 and 3 standard deviations of the center line, and beyond
ZONE_SIGNAL = 8


def zone_scores(z) -> np.ndarray:
    """Cumulative zone scores: restarted when the points change sides and after a signal."""
    z = np.asarray(z, dtype=float)
    weights = np.select([np.abs(z) < 1, np.abs(z) < 2, np.abs(z) < 3], ZONE_WEIGHTS[:3], ZONE_WEIGHTS[3])
    out, total, last_side = np.zeros(len(z), dtype=int), 0, 0.0
    for i, (value, weight) in enumerate(zip(z, weights)):
        side = np.sign(value)
        if (side and last_side and side != last_side) or total >= ZONE_SIGNAL:
            total = 0
        total += weight
        out[i] = total
        last_side = side or last_side
    return out


@analysis("cc.zone", menu=VARIABLES_MENU, title="Zone Chart",
          params=[MEASUREMENTS, Param("subgroup_size", "Subgroup size", "int", default=1, minimum=1)])
def zone_chart(ws, column, subgroup_size=1):
    x, se = _points(ws, column, subgroup_size)
    center = float(x.mean())
    z = (x - center) / se
    scores = zone_scores(z)
    signal = np.flatnonzero(scores >= ZONE_SIGNAL)
    fig, ax = _one_panel(f"{tr('Zone Chart')}: {column}", "Sample" if subgroup_size > 1 else "Observation")
    index = np.arange(1, len(x) + 1)
    shown = np.clip(z, -3.5, 3.5)   # points beyond the last zone are drawn inside it
    ax.plot(index, shown, "-", color=_spc.BLUE, linewidth=1)
    for i, (xi, yi, score) in enumerate(zip(index, shown, scores)):
        ax.annotate(str(score), (xi, yi), ha="center", va="center", fontsize=7,
                    color="white", bbox={"boxstyle": "circle,pad=0.2", "linewidth": 0,
                                         "facecolor": _spc.RED if i in signal else _spc.BLUE})
    for level in range(-3, 4):
        ax.axhline(level, color=_spc.GREEN if level == 0 else _spc.RED if abs(level) == 3 else _spc.GRAY, linewidth=0.8)
        ax.annotate(f"{center + level * se:.4g}", (1.005, level), xycoords=("axes fraction", "data"), fontsize=7,
                    va="center")
    ax.set_ylim(-4, 4), ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_yticks([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], [str(w) for w in ZONE_WEIGHTS[2::-1] + ZONE_WEIGHTS[:3]])
    ax.set_ylabel(tr("Zone weight"))
    res = Result("Zone Chart")
    res.text(f"{tr('Center line')} = {center:.6g}    {tr('Estimated standard deviation')}: {se:.6g}    "
             f"{tr('Zone weights')}: {', '.join(str(w) for w in ZONE_WEIGHTS)}")
    res.figure(fig)
    if len(signal):
        res.text(f"{tr('Cumulative score of 8 or more at')}: {', '.join(str(i + 1) for i in signal)}")
    else:
        res.text(tr("No cumulative score reached 8."))
    return res


def imr_limits(x) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        raise ValueError("There is not enough data to estimate the variation")
    mr = np.abs(np.diff(x))
    mr_bar = float(mr.mean())
    sigma = mr_bar / d2(2)
    center = float(x.mean())
    return {"sigma": sigma, "center": center, "lcl": center - 3 * sigma, "ucl": center + 3 * sigma,
            "mr": np.concatenate([[np.nan], mr]), "mr_center": mr_bar, "mr_lcl": 0.0,
            "mr_ucl": (d2(2) + 3 * d3(2)) * sigma, "z": (x - center) / sigma if sigma > 0 else np.zeros(len(x))}


@analysis("cc.imr", menu=VARIABLES_MENU, title="I-MR Chart", params=[MEASUREMENTS, ALL_TESTS])
def i_mr(ws, column, all_tests=False):
    x = clean(ws.df[column])
    lim = imr_limits(x)
    signals = special_cause_tests(lim["z"], ALL if all_tests else (1,))
    fig, top, bottom = _two_panel(f"{tr('I-MR Chart')}: {column}")
    bottom.set_xlabel(tr("Observation"))
    flagged = sorted({i for v in signals.values() for i in v})
    out_i = draw_chart(top, x, lim["center"], lim["lcl"], lim["ucl"], tr("Individual value"), flagged)
    out_mr = draw_chart(bottom, lim["mr"], lim["mr_center"], lim["mr_lcl"], lim["mr_ucl"], tr("Moving range"))
    res = Result("I-MR Chart")
    res.text(f"N = {len(x)}    {tr('Estimated standard deviation')}: {lim['sigma']:.6g}")
    res.table(limits_table({"I": {"LCL": lim["lcl"], "CL": lim["center"], "UCL": lim["ucl"]},
                            "MR": {"LCL": lim["mr_lcl"], "CL": lim["mr_center"], "UCL": lim["mr_ucl"]}}),
              title="Control limits")
    res.figure(fig)
    report_signals(res, "I", signals, out_i)
    report_signals(res, "MR", {}, out_mr)
    return res


# ----- attributes -----

COUNTS = Param("column", "Counts", "column")
SIZE_COLUMN = Param("size_col", "Subgroup sizes (column)", "column", optional=True)
SIZE_CONSTANT = Param("size", "Subgroup size (if constant)", "int", default=50, minimum=1, maximum=10**9)
ATTRIBUTE_PARAMS = [COUNTS, SIZE_COLUMN, SIZE_CONSTANT, ALL_TESTS]


def attribute_data(ws, column, size_col, size):
    cols = [column] + ([size_col] if size_col else [])
    frame = ws.df[cols].dropna().astype(float)
    counts = frame[column].to_numpy()
    sizes = frame[size_col].to_numpy() if size_col else np.full(len(frame), float(size))
    if not len(counts):
        raise ValueError("The worksheet has no data.")
    if (counts < 0).any() or (sizes <= 0).any():
        raise ValueError("Counts must be non-negative and subgroup sizes positive")
    return counts, sizes


def p_limits(counts, sizes) -> dict:
    if (counts > sizes).any():
        raise ValueError("The number of defectives cannot exceed the subgroup size")
    p_bar = counts.sum() / sizes.sum()
    se = np.sqrt(p_bar * (1 - p_bar) / sizes)
    return {"values": counts / sizes, "center": float(p_bar), "se": se,
            "lcl": np.maximum(p_bar - 3 * se, 0.0), "ucl": np.minimum(p_bar + 3 * se, 1.0)}


def u_limits(counts, sizes) -> dict:
    u_bar = counts.sum() / sizes.sum()
    se = np.sqrt(u_bar / sizes)
    return {"values": counts / sizes, "center": float(u_bar), "se": se,
            "lcl": np.maximum(u_bar - 3 * se, 0.0), "ucl": u_bar + 3 * se}


def _attribute_chart(title: str, column: str, lim: dict, ylabel: str, all_tests: bool, note: str = "") -> Result:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(lim["se"] > 0, (lim["values"] - lim["center"]) / lim["se"], 0.0)
    signals = special_cause_tests(z, (1, 2, 3, 4) if all_tests else (1,))
    fig, ax = _one_panel(f"{tr(title)}: {column}")
    flagged = sorted({i for v in signals.values() for i in v})
    outside = draw_chart(ax, lim["values"], lim["center"], lim["lcl"], lim["ucl"], tr(ylabel), flagged)
    res = Result(title)
    if note:
        res.text(note)
    res.table(limits_table({title.split()[0]: {"LCL": _last(lim["lcl"]), "CL": lim["center"],
                                              "UCL": _last(lim["ucl"])}}), title="Control limits (last subgroup)")
    res.figure(fig)
    report_signals(res, title.split()[0], signals, outside)
    return res


@analysis("cc.p", menu=ATTRIBUTES_MENU, title="P Chart", params=ATTRIBUTE_PARAMS)
def p_chart(ws, column, size_col=None, size=50, all_tests=False):
    counts, sizes = attribute_data(ws, column, size_col, size)
    return _attribute_chart("P Chart", column, p_limits(counts, sizes), "Proportion", all_tests)


@analysis("cc.np", menu=ATTRIBUTES_MENU, title="NP Chart", params=[COUNTS, SIZE_CONSTANT, ALL_TESTS])
def np_chart(ws, column, size=50, all_tests=False):
    counts, sizes = attribute_data(ws, column, None, size)
    lim = p_limits(counts, sizes)
    scaled = {"values": counts, "center": lim["center"] * size, "se": lim["se"] * size,
              "lcl": lim["lcl"] * size, "ucl": lim["ucl"] * size}
    return _attribute_chart("NP Chart", column, scaled, "Count", all_tests)


@analysis("cc.c", menu=ATTRIBUTES_MENU, title="C Chart", params=[COUNTS, ALL_TESTS])
def c_chart(ws, column, all_tests=False):
    counts, sizes = attribute_data(ws, column, None, 1)
    return _attribute_chart("C Chart", column, u_limits(counts, sizes), "Count", all_tests)


@analysis("cc.u", menu=ATTRIBUTES_MENU, title="U Chart", params=ATTRIBUTE_PARAMS)
def u_chart(ws, column, size_col=None, size=50, all_tests=False):
    counts, sizes = attribute_data(ws, column, size_col, size)
    return _attribute_chart("U Chart", column, u_limits(counts, sizes), "Count per unit", all_tests)


def laney(lim: dict) -> tuple[dict, float]:
    """Laney's adjustment: widen the limits by the short-term variation of the z-scores."""
    z = (lim["values"] - lim["center"]) / lim["se"]
    sigma_z = _spc.sigma_moving_range(z)
    se = lim["se"] * sigma_z
    out = dict(lim, se=se, lcl=np.maximum(lim["center"] - 3 * se, 0.0), ucl=lim["center"] + 3 * se)
    return out, sigma_z


@analysis("cc.laney_p", menu=ATTRIBUTES_MENU, title="Laney P' Chart", params=ATTRIBUTE_PARAMS)
def laney_p(ws, column, size_col=None, size=50, all_tests=False):
    counts, sizes = attribute_data(ws, column, size_col, size)
    lim, sigma_z = laney(p_limits(counts, sizes))
    lim["ucl"] = np.minimum(lim["ucl"], 1.0)
    return _attribute_chart("Laney P' Chart", column, lim, "Proportion", all_tests, f"Sigma Z = {sigma_z:.5g}")


@analysis("cc.laney_u", menu=ATTRIBUTES_MENU, title="Laney U' Chart", params=ATTRIBUTE_PARAMS)
def laney_u(ws, column, size_col=None, size=50, all_tests=False):
    counts, sizes = attribute_data(ws, column, size_col, size)
    lim, sigma_z = laney(u_limits(counts, sizes))
    return _attribute_chart("Laney U' Chart", column, lim, "Count per unit", all_tests, f"Sigma Z = {sigma_z:.5g}")


# ----- time-weighted -----

def _points(ws, column, subgroup_size):
    """Plotted statistics (individuals or subgroup means) with their standard error."""
    if subgroup_size > 1:
        groups = make_subgroups(ws, column, subgroup_size)
        return np.array([g.mean() for g in groups]), _spc.sigma_pooled(groups) / np.sqrt(subgroup_size)
    x = clean(ws.df[column])
    return x, _spc.sigma_moving_range(x)


def ewma_limits(x, se, weight=0.2, width=3.0) -> dict:
    center = float(np.mean(x))
    z = np.empty(len(x))
    prev = center
    for i, value in enumerate(x):
        prev = weight * value + (1 - weight) * prev
        z[i] = prev
    i = np.arange(1, len(x) + 1)
    half = width * se * np.sqrt(weight / (2 - weight) * (1 - (1 - weight) ** (2 * i)))
    return {"values": z, "center": center, "lcl": center - half, "ucl": center + half}


@analysis("cc.ewma", menu=WEIGHTED_MENU, title="EWMA Chart",
          params=[MEASUREMENTS, Param("subgroup_size", "Subgroup size", "int", default=1, minimum=1),
                  Param("weight", "Weight of EWMA", "number", default=0.2, minimum=0.01, maximum=1.0)])
def ewma(ws, column, subgroup_size=1, weight=0.2):
    x, se = _points(ws, column, subgroup_size)
    lim = ewma_limits(x, se, weight)
    fig, ax = _one_panel(f"{tr('EWMA Chart')}: {column}")
    outside = draw_chart(ax, lim["values"], lim["center"], lim["lcl"], lim["ucl"], "EWMA")
    res = Result("EWMA Chart")
    res.text(f"λ = {weight:g}")
    res.table(limits_table({"EWMA": {"LCL": _last(lim["lcl"]), "CL": lim["center"], "UCL": _last(lim["ucl"])}}),
              title="Control limits (last subgroup)")
    res.figure(fig)
    report_signals(res, "EWMA", {}, outside)
    return res


def cusum_values(x, se, target, k=0.5):
    high, low = np.zeros(len(x)), np.zeros(len(x))
    h_prev = l_prev = 0.0
    for i, value in enumerate(x):
        h_prev = max(0.0, value - (target + k * se) + h_prev)
        l_prev = min(0.0, value - (target - k * se) + l_prev)
        high[i], low[i] = h_prev, l_prev
    return high, low


@analysis("cc.cusum", menu=WEIGHTED_MENU, title="CUSUM Chart",
          params=[MEASUREMENTS, Param("subgroup_size", "Subgroup size", "int", default=1, minimum=1),
                  Param("use_target", "Use the target below instead of the mean", "bool", default=False),
                  Param("target", "Target", "number", default=0.0),
                  Param("h", "h (decision interval)", "number", default=4.0, minimum=0.1),
                  Param("k", "k (allowable slack)", "number", default=0.5, minimum=0.0)])
def cusum(ws, column, subgroup_size=1, use_target=False, target=0.0, h=4.0, k=0.5):
    x, se = _points(ws, column, subgroup_size)
    centre = target if use_target else float(x.mean())
    high, low = cusum_values(x, se, centre, k)
    limit = h * se
    fig, ax = _one_panel(f"{tr('CUSUM Chart')}: {column}")
    index = np.arange(1, len(x) + 1)
    ax.plot(index, high, "-o", color=_spc.BLUE, markersize=4)
    ax.plot(index, low, "-o", color=_spc.BLUE, markersize=4)
    bad = np.flatnonzero((high > limit) | (low < -limit))
    ax.plot(index[bad], np.where(high[bad] > limit, high[bad], low[bad]), "s", color=_spc.RED, markersize=5)
    for level, name in ((limit, "UCL"), (0.0, "0"), (-limit, "LCL")):
        ax.axhline(level, color=_spc.GREEN if level == 0 else _spc.RED, linewidth=1)
        ax.annotate(f"{name}={level:.4g}" if level else "0", (1.005, level), xycoords=("axes fraction", "data"),
                    fontsize=7, va="center")
    ax.set_ylabel(tr("Cumulative sum"))
    res = Result("CUSUM Chart")
    res.text(f"{tr('Target')} = {centre:.6g}    h = {h:g}    k = {k:g}")
    res.table(limits_table({"CUSUM": {"LCL": -limit, "CL": 0.0, "UCL": limit}}), title="Control limits")
    res.figure(fig)
    report_signals(res, "CUSUM", {}, bad.tolist())
    return res


@analysis("cc.moving_average", menu=WEIGHTED_MENU, title="Moving Average Chart",
          params=[MEASUREMENTS, Param("subgroup_size", "Subgroup size", "int", default=1, minimum=1),
                  Param("span", "Length of moving average", "int", default=3, minimum=2)])
def moving_average(ws, column, subgroup_size=1, span=3):
    x, se = _points(ws, column, subgroup_size)
    center = float(x.mean())
    window = np.minimum(np.arange(1, len(x) + 1), span)
    ma = np.array([x[max(0, i - span + 1):i + 1].mean() for i in range(len(x))])
    half = 3 * se / np.sqrt(window)
    fig, ax = _one_panel(f"{tr('Moving Average Chart')}: {column}")
    outside = draw_chart(ax, ma, center, center - half, center + half, tr("Moving average"))
    res = Result("Moving Average Chart")
    res.table(limits_table({"MA": {"LCL": center - half[-1], "CL": center, "UCL": center + half[-1]}}),
              title="Control limits (last subgroup)")
    res.figure(fig)
    report_signals(res, "MA", {}, outside)
    return res


# ----- multivariate and rare events -----

def t_squared(X, alpha=0.0027) -> dict:
    """Hotelling T-squared for individual observations, with the phase I beta limit."""
    X = np.asarray(X, dtype=float)
    m, p = X.shape
    if m <= p + 1:
        raise ValueError("More observations than variables are needed")
    centered = X - X.mean(axis=0)
    inverse = np.linalg.inv(np.cov(X, rowvar=False))
    values = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    ucl = (m - 1) ** 2 / m * sps.beta.ppf(1 - alpha, p / 2, (m - p - 1) / 2)
    return {"values": values, "ucl": float(ucl), "center": float((m - 1) ** 2 / m *
                                                                 sps.beta.ppf(0.5, p / 2, (m - p - 1) / 2))}


@analysis("cc.t2", menu=OTHER_MENU, title="T² Chart", params=[Param("columns", "Variables", "columns")])
def t2_chart(ws, columns):
    if len(columns) < 2:
        raise ValueError("Select at least two columns")
    out = t_squared(ws.data(columns).to_numpy(dtype=float))
    fig, ax = _one_panel(f"{tr('T² Chart')}: {', '.join(columns)}", "Observation")
    outside = draw_chart(ax, out["values"], out["center"], 0.0, out["ucl"], "T²")
    res = Result("T² Chart")
    res.table(limits_table({"T²": {"LCL": 0.0, "CL": out["center"], "UCL": out["ucl"]}}), title="Control limits")
    res.figure(fig)
    report_signals(res, "T²", {}, outside)
    return res


def multivariate_subgroups(ws, columns, subgroup_size: int) -> list[np.ndarray]:
    if len(columns) < 2:
        raise ValueError("Select at least two columns")
    X = ws.data(columns).to_numpy(dtype=float)
    count = len(X) // subgroup_size
    if count < 2 or subgroup_size <= len(columns):
        raise ValueError("At least two subgroups are needed, each with more observations than variables")
    return [X[i * subgroup_size:(i + 1) * subgroup_size] for i in range(count)]


def subgroup_t_squared(groups: list[np.ndarray], alpha=0.0027) -> dict:
    """Hotelling T-squared of subgroup means about the grand mean, with the phase I F limit."""
    m, (n, p) = len(groups), groups[0].shape
    pooled = np.mean([np.atleast_2d(np.cov(g, rowvar=False)) for g in groups], axis=0)
    means = np.array([g.mean(axis=0) for g in groups])
    centered = means - means.mean(axis=0)
    values = n * np.einsum("ij,jk,ik->i", centered, np.linalg.inv(pooled), centered)
    df = m * n - m - p + 1
    factor = p * (m - 1) * (n - 1) / df
    return {"values": values, "ucl": float(factor * sps.f.ppf(1 - alpha, p, df)),
            "center": float(factor * sps.f.ppf(0.5, p, df)), "pooled": pooled}


def generalized_variance(groups: list[np.ndarray]) -> dict:
    """Determinants of the subgroup covariance matrices with three-sigma limits.

    For normal data E|S| = b1 |Sigma| and Var|S| = b2 |Sigma|^2, where b1 and b2 depend only on
    the subgroup size and the number of variables.
    """
    n, p = groups[0].shape
    i = np.arange(1, p + 1)
    b1 = np.prod(n - i) / (n - 1) ** p
    b2 = np.prod(n - i) * (np.prod(n - i + 2) - np.prod(n - i)) / (n - 1) ** (2 * p)
    covariances = [np.atleast_2d(np.cov(g, rowvar=False)) for g in groups]
    sigma = np.linalg.det(np.mean(covariances, axis=0)) / b1
    return {"values": np.array([np.linalg.det(c) for c in covariances]), "center": float(b1 * sigma),
            "lcl": float(max(0.0, sigma * (b1 - 3 * np.sqrt(b2)))), "ucl": float(sigma * (b1 + 3 * np.sqrt(b2)))}


MULTIVARIATE_PARAMS = [Param("columns", "Variables", "columns"),
                       Param("subgroup_size", "Subgroup size", "int", default=5, minimum=2, maximum=10000)]


@analysis("cc.t2_gv", menu=OTHER_MENU, title="T²-Generalized Variance Chart", params=MULTIVARIATE_PARAMS)
def t2_generalized_variance(ws, columns, subgroup_size=5):
    groups = multivariate_subgroups(ws, columns, subgroup_size)
    t2, gv = subgroup_t_squared(groups), generalized_variance(groups)
    fig, top, bottom = _two_panel(f"{tr('T²-Generalized Variance Chart')}: {', '.join(columns)}")
    out_t2 = draw_chart(top, t2["values"], t2["center"], 0.0, t2["ucl"], "T²")
    out_gv = draw_chart(bottom, gv["values"], gv["center"], gv["lcl"], gv["ucl"], tr("Generalized variance"))
    res = Result("T²-Generalized Variance Chart")
    res.table(limits_table({"T²": {"LCL": 0.0, "CL": t2["center"], "UCL": t2["ucl"]},
                            tr("Generalized variance"): {"LCL": gv["lcl"], "CL": gv["center"], "UCL": gv["ucl"]}}),
              title="Control limits")
    res.figure(fig)
    report_signals(res, "T²", {}, out_t2)
    report_signals(res, tr("Generalized variance"), {}, out_gv)
    return res


@analysis("cc.gv", menu=OTHER_MENU, title="Generalized Variance Chart", params=MULTIVARIATE_PARAMS)
def generalized_variance_chart(ws, columns, subgroup_size=5):
    gv = generalized_variance(multivariate_subgroups(ws, columns, subgroup_size))
    fig, ax = _one_panel(f"{tr('Generalized Variance Chart')}: {', '.join(columns)}")
    outside = draw_chart(ax, gv["values"], gv["center"], gv["lcl"], gv["ucl"], tr("Generalized variance"))
    res = Result("Generalized Variance Chart")
    res.table(limits_table({tr("Generalized variance"): {"LCL": gv["lcl"], "CL": gv["center"], "UCL": gv["ucl"]}}),
              title="Control limits")
    res.figure(fig)
    report_signals(res, tr("Generalized variance"), {}, outside)
    return res


def mewma_arl(h: float, p: int, weight: float, states: int = 100) -> float:
    """In-control average run length of a MEWMA chart with limit h on the asymptotic statistic.

    In control only the length of the EWMA vector matters, and it is a Markov chain: given its
    current length r, the next squared length is weight^2 times a noncentral chi-square with p
    degrees of freedom and noncentrality ((1 - weight) r / weight)^2. The chain is discretized
    into rings of equal width (Runger and Prabhu, 1996).
    """
    radius = np.sqrt(h * weight / (2 - weight))
    width = 2 * radius / (2 * states + 1)
    centers = np.arange(states + 1) * width
    edges = np.append(0.0, (np.arange(states + 1) + 0.5) * width)
    ncp = ((1 - weight) * centers / weight) ** 2
    cdf = np.where(ncp[:, None] > 0, sps.ncx2.cdf((edges[None, :] / weight) ** 2, p, np.maximum(ncp[:, None], 1e-300)),
                   sps.chi2.cdf((edges[None, :] / weight) ** 2, p))
    transition = np.diff(cdf, axis=1)
    return float(np.linalg.solve(np.eye(states + 1) - transition, np.ones(states + 1))[0])


def mewma_limit(p: int, weight: float, arl: float = 200.0) -> float:
    """Control limit h that gives the in-control average run length `arl`."""
    high = 1.0
    while mewma_arl(high, p, weight) < arl:   # the run length grows without bound, and fast, with h
        high *= 1.25
    return float(optimize.brentq(lambda h: mewma_arl(h, p, weight) - arl, high / 1.25, high))


def mewma(X, weight=0.1) -> np.ndarray:
    """MEWMA statistics of the rows of X, each with the exact covariance of its EWMA vector."""
    X = np.asarray(X, dtype=float)
    m, p = X.shape
    if m <= p + 1:
        raise ValueError("More observations than variables are needed")
    inverse = np.linalg.inv(np.cov(X, rowvar=False))
    centered = X - X.mean(axis=0)
    z, values = np.zeros(p), np.empty(m)
    for i, row in enumerate(centered):
        z = weight * row + (1 - weight) * z
        scale = weight / (2 - weight) * (1 - (1 - weight) ** (2 * (i + 1)))
        values[i] = z @ inverse @ z / scale
    return values


@analysis("cc.mewma", menu=OTHER_MENU, title="Multivariate EWMA Chart",
          params=[Param("columns", "Variables", "columns"),
                  Param("weight", "Weight of EWMA", "number", default=0.1, minimum=0.01, maximum=1.0),
                  Param("arl", "In-control average run length", "number", default=200.0, minimum=10, maximum=5000)])
def mewma_chart(ws, columns, weight=0.1, arl=200.0):
    if len(columns) < 2:
        raise ValueError("Select at least two columns")
    values = mewma(ws.data(columns).to_numpy(dtype=float), weight)
    ucl = mewma_limit(len(columns), weight, arl)
    fig, ax = _one_panel(f"{tr('Multivariate EWMA Chart')}: {', '.join(columns)}", "Observation")
    outside = draw_chart(ax, values, np.nan, 0.0, ucl, "MEWMA")
    res = Result("Multivariate EWMA Chart")
    res.text(f"λ = {weight:g}    {tr('In-control average run length')} = {arl:g}")
    res.table(limits_table({"MEWMA": {"LCL": 0.0, "UCL": ucl}}), title="Control limits")
    res.figure(fig)
    report_signals(res, "MEWMA", {}, outside)
    return res


def _quantile_chart(title: str, column: str, x, dist, ylabel: str, note: str) -> Result:
    lcl, center, ucl = (float(dist.ppf(q)) for q in (0.00135, 0.5, 0.99865))
    fig, ax = _one_panel(f"{tr(title)}: {column}", "Observation")
    outside = draw_chart(ax, x, center, lcl, ucl, tr(ylabel))
    res = Result(title)
    res.text(note)
    res.table(limits_table({title.split()[0]: {"LCL": lcl, "CL": center, "UCL": ucl}}), title="Control limits")
    res.figure(fig)
    report_signals(res, title.split()[0], {}, outside)
    return res


@analysis("cc.g", menu=OTHER_MENU, title="G Chart",
          params=[Param("column", "Number of opportunities between events", "column")])
def g_chart(ws, column):
    x = clean(ws.df[column])
    if len(x) < 2 or (x < 0).any():
        raise ValueError("Counts must be non-negative")
    p = 1 / (x.mean() + 1)
    # number of failures before the first event: geometric on 0, 1, 2, ...
    dist = sps.geom(p, loc=-1)
    return _quantile_chart("G Chart", column, x, dist, "Number between events",
                           f"{tr('Geometric distribution')}: p = {p:.5g}")


@analysis("cc.t", menu=OTHER_MENU, title="T Chart", params=[Param("column", "Time between events", "column")])
def t_chart(ws, column):
    x = clean(ws.df[column])
    if len(x) < 3 or (x <= 0).any():
        raise ValueError("Times between events must be positive")
    shape, _, scale = sps.weibull_min.fit(x, floc=0)
    return _quantile_chart("T Chart", column, x, sps.weibull_min(shape, scale=scale), "Time between events",
                           f"{tr('Weibull distribution')}: {tr('shape')} = {shape:.5g}, {tr('scale')} = {scale:.5g}")


# ----- Box-Cox -----

def box_cox_lambda(x, confidence=95.0) -> dict:
    """Maximum-likelihood lambda with a likelihood-ratio confidence interval."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or (x <= 0).any():
        raise ValueError("The Box-Cox transformation needs positive data")
    grid = np.linspace(-5, 5, 2001)
    llf = np.array([sps.boxcox_llf(lam, x) for lam in grid])
    best = float(sps.boxcox_normmax(x, method="mle", brack=(-2.0, 2.0)))
    cutoff = sps.boxcox_llf(best, x) - sps.chi2.ppf(confidence / 100, 1) / 2
    inside = grid[llf >= cutoff]
    lower, upper = (float(inside.min()), float(inside.max())) if len(inside) else (best, best)
    candidates = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    rounded = candidates[np.abs(candidates - best).argmin()]
    rounded = float(rounded) if lower <= rounded <= upper else round(best, 2)
    return {"lambda": best, "lower": lower, "upper": upper, "rounded": rounded, "grid": grid, "llf": llf}


def box_cox_transform(x, lam: float):
    x = np.asarray(x, dtype=float)
    return np.log(x) if lam == 0 else x ** lam


@analysis("cc.box_cox", menu=MENU, title="Box-Cox Transformation",
          params=[MEASUREMENTS, Param("store", "Store transformed data", "bool", default=False)])
def box_cox(ws, column, store=False):
    x = ws.df[column].to_numpy(dtype=float)
    out = box_cox_lambda(x[~np.isnan(x)])
    fig, ax = _one_panel(f"{tr('Box-Cox Transformation')}: {column}", "Lambda")
    keep = out["llf"] > out["llf"].max() - 12
    ax.plot(out["grid"][keep], out["llf"][keep], color=_spc.BLUE)
    for value, color in ((out["lambda"], _spc.RED), (out["lower"], _spc.GRAY), (out["upper"], _spc.GRAY)):
        ax.axvline(value, color=color, linestyle="--", linewidth=1)
    ax.set_ylabel(tr("Log-likelihood"))
    res = Result("Box-Cox Transformation")
    res.table(pd.DataFrame([{"Estimate": out["lambda"], "CI Lower": out["lower"], "CI Upper": out["upper"],
                             "Rounded value": out["rounded"]}], index=pd.Index(["Lambda"], name="")), digits=3)
    res.text(tr("Transformation: Y' = ln(Y) when lambda = 0, otherwise Y' = Y ^ lambda"))
    res.figure(fig)
    if store:
        with np.errstate(all="ignore"):
            res.store(f"BoxCox_{column}", box_cox_transform(x, out["rounded"]).tolist())
    return res
