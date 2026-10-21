"""Quality tools: run chart, Pareto, cause-and-effect, distribution identification, capability and sampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats import _spc, distfit
from openstat.stats._spc import (MEASUREMENTS, SUBGROUP_COLUMN, SUBGROUP_SIZE, draw_chart, make_subgroups,
                                 within_sigma)
from openstat.stats._tests import CONFIDENCE
from openstat.stats._util import anderson_darling_normal, clean, median_ranks
from openstat.stats.basic import clopper_pearson, poisson_rate_ci
from openstat.stats.control import (attribute_data, between_within, imr_limits, p_limits, range_limits, u_limits,
                                    xbar_limits)

MENU = ("Stat", "Quality Tools")
CAPABILITY_MENU = MENU + ("Capability Analysis",)
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"

LSL = Param("lsl", "Lower spec limit (blank = none)", "text", default="")
USL = Param("usl", "Upper spec limit (blank = none)", "text", default="")
TARGET = Param("target", "Target (blank = none)", "text", default="")


def optional_number(text) -> float | None:
    text = str(text or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Not a number: {text}") from exc


def spec_limits(lsl, usl) -> tuple[float | None, float | None]:
    lsl, usl = optional_number(lsl), optional_number(usl)
    if lsl is None and usl is None:
        raise ValueError("Enter at least one specification limit")
    if lsl is not None and usl is not None and lsl >= usl:
        raise ValueError("The lower spec limit must be less than the upper spec limit")
    return lsl, usl


# ----- run chart -----

def run_chart_tests(values) -> dict:
    """Runs about the median and runs up or down, with normal-approximation p-values."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    median = float(np.median(x))
    above = x > median
    a, b = int(above.sum()), int((~above).sum())
    runs = int(1 + (above[1:] != above[:-1]).sum())
    out = {"median": median, "runs_median": runs, "expected_median": np.nan, "p_clustering": np.nan,
           "p_mixtures": np.nan}
    if a and b and n > 2:
        expected = 2 * a * b / n + 1
        variance = 2 * a * b * (2 * a * b - n) / (n ** 2 * (n - 1))
        if variance > 0:
            z = (runs - expected) / np.sqrt(variance)
            out.update(expected_median=expected, p_clustering=float(sps.norm.cdf(z)),
                       p_mixtures=float(sps.norm.sf(z)))
    steps = np.sign(np.diff(x))
    steps = steps[steps != 0]
    updown = int(1 + (steps[1:] != steps[:-1]).sum()) if len(steps) else 0
    out.update(runs_updown=updown, expected_updown=np.nan, p_trends=np.nan, p_oscillation=np.nan)
    if n > 2:
        expected, variance = (2 * n - 1) / 3, (16 * n - 29) / 90
        z = (updown - expected) / np.sqrt(variance)
        out.update(expected_updown=expected, p_trends=float(sps.norm.cdf(z)), p_oscillation=float(sps.norm.sf(z)))
    return out


@analysis("quality.run_chart", menu=MENU, title="Run Chart",
          params=[MEASUREMENTS, Param("subgroup_size", "Subgroup size", "int", default=1, minimum=1)])
def run_chart(ws, column, subgroup_size=1):
    groups = make_subgroups(ws, column, subgroup_size)
    points = np.array([g.mean() for g in groups])
    out = run_chart_tests(points)
    fig = new_figure(8.0, 4.0)
    ax = fig.add_subplot()
    ax.plot(np.arange(1, len(points) + 1), points, "-o", color=BLUE, markersize=4)
    ax.axhline(out["median"], color=RED, linewidth=1)
    ax.set_xlabel(tr("Sample") if subgroup_size > 1 else tr("Observation")), ax.set_ylabel(column)
    ax.set_title(f"{tr('Run Chart')}: {column}")
    res = Result("Run Chart").figure(fig)
    res.table(pd.DataFrame({"Observed runs": [out["runs_median"], out["runs_updown"]],
                            "Expected runs": [out["expected_median"], out["expected_updown"]]},
                           index=pd.Index(["Runs about the median", "Runs up or down"], name="Test")))
    return res.table(pd.DataFrame({"P-Value": [out["p_clustering"], out["p_mixtures"], out["p_trends"],
                                               out["p_oscillation"]]},
                                  index=pd.Index(["Clustering", "Mixtures", "Trends", "Oscillation"],
                                                 name="Pattern")), title="Approximate p-values")


# ----- Pareto and cause-and-effect -----

def pareto_table(labels, weights=None, combine_after: float | None = 95.0) -> pd.DataFrame:
    frame = pd.DataFrame({"label": [format_value(v) for v in labels],
                          "weight": 1.0 if weights is None else np.asarray(weights, dtype=float)})
    counts = frame.groupby("label")["weight"].sum().sort_values(ascending=False, kind="stable")
    if counts.sum() <= 0:
        raise ValueError("Counts must be non-negative and not all zero")
    if combine_after is not None and len(counts) > 2:
        cumulative = 100 * counts.cumsum() / counts.sum()
        keep = int(np.searchsorted(cumulative.to_numpy(), combine_after, side="left")) + 1
        if keep < len(counts) - 1:
            other = counts.iloc[keep:].sum()
            counts = counts.iloc[:keep]
            counts.loc[tr("Other")] = other
    table = pd.DataFrame({"Count": counts, "Percent": 100 * counts / counts.sum()})
    table["Cum %"] = table["Percent"].cumsum()
    return table


@analysis("quality.pareto", menu=MENU, title="Pareto Chart",
          params=[Param("category", "Defects or attribute data", "column", numeric=False),
                  Param("frequency", "Frequencies", "column", optional=True),
                  Param("combine", "Combine remaining defects after 95%", "bool", default=True)])
def pareto(ws, category, frequency=None, combine=True):
    cols = [category] + ([frequency] if frequency else [])
    frame = ws.df[cols].dropna()
    table = pareto_table(frame[category], frame[frequency] if frequency else None, 95.0 if combine else None)
    table.index.name = category
    fig = new_figure(max(6.0, 1.5 + 0.7 * len(table)), 4.2)
    ax = fig.add_subplot()
    ax.bar(list(table.index), table["Count"], color=BLUE)
    ax.set_ylabel(tr("Count")), ax.set_title(f"{tr('Pareto Chart')}: {category}")
    ax.tick_params(axis="x", rotation=30)
    line = ax.twinx()
    line.plot(list(table.index), table["Cum %"], "-o", color=RED, markersize=4)
    line.set_ylim(0, 105), line.set_ylabel(tr("Percent"))
    ax.set_ylim(0, table["Count"].sum() * 1.05)
    return Result("Pareto Chart").figure(fig).table(table, digits=1)


@analysis("quality.fishbone", menu=MENU, title="Cause-and-Effect Diagram",
          params=[Param("columns", "Columns of causes (one branch per column)", "columns", numeric=False),
                  Param("effect", "Effect", "text", default="")])
def cause_and_effect(ws, columns, effect=""):
    branches = [(name, [format_value(v) for v in ws.df[name].dropna()]) for name in columns]
    fig = new_figure(9.0, 5.5)
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.set_xlim(0, 10), ax.set_ylim(-5, 5)
    ax.annotate("", xy=(9.0, 0), xytext=(0.3, 0), arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2})
    ax.text(9.1, 0, effect or tr("Effect"), va="center", ha="left", fontsize=11, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "black"})
    per_side = (len(branches) + 1) // 2
    for i, (name, causes) in enumerate(branches):
        side = 1 if i % 2 == 0 else -1
        x_spine = 8.2 - (i // 2) * (7.2 / max(per_side, 1))
        x_top = x_spine - 1.3
        ax.plot([x_top, x_spine], [side * 4.0, 0], color="black", lw=1.2)
        ax.text(x_top, side * 4.25, name, ha="center", va="center", fontsize=10, fontweight="bold",
                bbox={"facecolor": "#ddebf7", "edgecolor": "black"})
        for j, cause in enumerate(causes[:8]):
            frac = (j + 1) / (min(len(causes), 8) + 1)
            x, y = x_top + (x_spine - x_top) * frac, side * 4.0 * (1 - frac)
            ax.plot([x - 0.9, x], [y, y], color=GRAY, lw=0.8)
            ax.text(x - 0.95, y, cause, ha="right", va="center", fontsize=8)
    return Result("Cause-and-Effect Diagram").figure(fig)


# ----- distribution identification and transformations -----

def quantile_plot(ax, x, dist, title: str) -> None:
    ordered = np.sort(x)
    theoretical = dist.ppf(median_ranks(len(ordered)))
    ax.plot(theoretical, ordered, "o", color=BLUE, markersize=3)
    span = [min(theoretical[0], ordered[0]), max(theoretical[-1], ordered[-1])]
    ax.plot(span, span, color=RED, linewidth=1)
    ax.set_title(title, fontsize=9)


def fits_table(fits) -> pd.DataFrame:
    rows = {}
    for f in fits:
        params = ", ".join(f"{tr(k)} = {v:.5g}" for k, v in f.params.items())
        rows[tr(f.name)] = {"AD": f.ad, "P-Value": f.p_value, "Parameters": params}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Distribution"
    return table


@analysis("quality.identify", menu=MENU, title="Individual Distribution Identification", params=[MEASUREMENTS])
def identify_distribution(ws, column):
    x = clean(ws.df[column])
    fits = distfit.fit_all(x)
    if not fits:
        raise ValueError("At least three observations are needed")
    res = Result("Individual Distribution Identification")
    res.text(f"N = {len(x)}    {tr('Distributions are ordered from best to worst fit (smallest Anderson-Darling).')}")
    res.table(fits_table(fits), title="Goodness-of-Fit Test")
    shown = fits[:4]
    fig = new_figure(8.0, 6.0 if len(shown) > 2 else 3.2)
    axes = np.atleast_1d(fig.subplots((len(shown) + 1) // 2, 2, squeeze=False)).ravel()
    for ax, f in zip(axes, shown):
        quantile_plot(ax, x, f.dist, f"{tr(f.name)} (AD = {f.ad:.3f})")
    for ax in axes[len(shown):]:
        ax.set_axis_off()
    fig.supxlabel(tr("Theoretical quantile")), fig.supylabel(column)
    return res.figure(fig)


def johnson_transform(x) -> dict:
    """Best Johnson-family transformation to normality (SU, SB or SL), judged by Anderson-Darling."""
    x = np.asarray(x, dtype=float)
    candidates = []
    try:
        a, b, loc, scale = sps.johnsonsu.fit(x)
        candidates.append(("SU", lambda v, p=(a, b, loc, scale): p[0] + p[1] * np.arcsinh((v - p[2]) / p[3]),
                           f"{a:.5g} + {b:.5g} * asinh((X - {loc:.5g}) / {scale:.5g})"))
    except Exception:
        pass
    try:
        a, b, loc, scale = sps.johnsonsb.fit(x)
        if loc < x.min() and loc + scale > x.max():
            def sb(v, p=(a, b, loc, scale)):
                y = (v - p[2]) / p[3]
                return p[0] + p[1] * np.log(y / (1 - y))
            candidates.append(("SB", sb, f"{a:.5g} + {b:.5g} * ln((X - {loc:.5g}) / ({loc + scale:.5g} - X))"))
    except Exception:
        pass
    if (x > 0).all():
        logs = np.log(x)
        m, s = logs.mean(), logs.std(ddof=1)
        candidates.append(("SL", lambda v, p=(m, s): (np.log(v) - p[0]) / p[1],
                           f"(ln(X) - {m:.5g}) / {s:.5g}"))
    best = None
    for family, func, text in candidates:
        with np.errstate(all="ignore"):
            z = func(x)
        if not np.all(np.isfinite(z)):
            continue
        ad, p = anderson_darling_normal(z)
        if best is None or ad < best["ad"]:
            best = {"family": family, "func": func, "text": text, "ad": ad, "p": p}
    if best is None:
        raise ValueError("No Johnson transformation could be fitted")
    return best


@analysis("quality.johnson", menu=MENU, title="Johnson Transformation",
          params=[MEASUREMENTS, Param("store", "Store transformed data", "bool", default=False)])
def johnson(ws, column, store=False):
    raw = ws.df[column].to_numpy(dtype=float)
    x = raw[~np.isnan(raw)]
    if len(x) < 8:
        raise ValueError("At least eight observations are needed")
    best = johnson_transform(x)
    ad0, p0 = anderson_darling_normal(x)
    z = best["func"](x)
    res = Result("Johnson Transformation")
    res.text(f"{tr('Transformation type')}: {best['family']}    Z = {best['text']}")
    res.table(pd.DataFrame({"AD": [ad0, best["ad"]], "P-Value": [p0, best["p"]]},
                           index=pd.Index(["Original data", "Transformed data"], name="")))
    fig = new_figure(8.0, 3.4)
    left, right = fig.subplots(1, 2)
    quantile_plot(left, x, sps.norm(x.mean(), x.std(ddof=1)), tr("Original data"))
    quantile_plot(right, z, sps.norm(z.mean(), z.std(ddof=1)), tr("Transformed data"))
    res.figure(fig)
    if store:
        with np.errstate(all="ignore"):
            res.store(f"Johnson_{column}", best["func"](raw).tolist())
    return res


# ----- capability -----

def ppm(value: float) -> float:
    return 1e6 * value


def capability_indices(mean, sigma, lsl, usl) -> dict:
    lower = (mean - lsl) / (3 * sigma) if lsl is not None else np.nan
    upper = (usl - mean) / (3 * sigma) if usl is not None else np.nan
    both = (usl - lsl) / (6 * sigma) if lsl is not None and usl is not None else np.nan
    return {"potential": both, "lower": lower, "upper": upper, "k": float(np.nanmin([lower, upper]))}


def capability_normal(groups, lsl, usl, target=None) -> dict:
    x = np.concatenate(groups)
    n = len(x)
    if n < 2:
        raise ValueError("At least two observations are needed")
    mean, overall, within = float(x.mean()), float(x.std(ddof=1)), within_sigma(groups)
    if within <= 0 or overall <= 0:
        raise ValueError("The data have no variation")
    out = {"n": n, "mean": mean, "within": within, "overall": overall,
           "cp": capability_indices(mean, within, lsl, usl), "pp": capability_indices(mean, overall, lsl, usl),
           "cpm": np.nan}
    if target is not None and lsl is not None and usl is not None:
        out["cpm"] = (usl - lsl) / (6 * np.sqrt(((x - target) ** 2).sum() / (n - 1)))

    def tails(sigma):
        below = sps.norm.cdf(lsl, mean, sigma) if lsl is not None else 0.0
        above = sps.norm.sf(usl, mean, sigma) if usl is not None else 0.0
        return ppm(below), ppm(above)

    observed = (ppm((x < lsl).mean()) if lsl is not None else 0.0, ppm((x > usl).mean()) if usl is not None else 0.0)
    out["ppm"] = {"Observed": observed, "Expected within": tails(within), "Expected overall": tails(overall)}
    return out


def _spec_lines(ax, lsl, usl, target):
    for value, label in ((lsl, "LSL"), (usl, "USL"), (target, tr("Target"))):
        if value is not None:
            ax.axvline(value, color=RED if label != tr("Target") else "#00803c", linestyle="--", linewidth=1.2)
            ax.annotate(label, (value, 1.01), xycoords=("data", "axes fraction"), ha="center", fontsize=8)


def capability_histogram(ax, x, out, lsl, usl, target, title):
    ax.hist(x, bins="auto", density=True, color=BLUE, edgecolor="white")
    limits = [v for v in (lsl, usl) if v is not None] + [x.min(), x.max()]
    span = max(limits) - min(limits)
    grid = np.linspace(min(limits) - 0.15 * span, max(limits) + 0.15 * span, 300)
    ax.plot(grid, sps.norm.pdf(grid, out["mean"], out["within"]), color=RED, label=tr("Within"))
    ax.plot(grid, sps.norm.pdf(grid, out["mean"], out["overall"]), color="black", linestyle="--",
            label=tr("Overall"))
    _spec_lines(ax, lsl, usl, target)
    ax.legend(fontsize=8)
    ax.set_title(title, pad=14)
    ax.set_yticks([])


def capability_tables(res: Result, out: dict, lsl, usl, target) -> None:
    res.table(pd.DataFrame([{"LSL": np.nan if lsl is None else lsl, "Target": np.nan if target is None else target,
                             "USL": np.nan if usl is None else usl, "Sample mean": out["mean"], "N": out["n"],
                             "StDev (within)": out["within"], "StDev (overall)": out["overall"]}]),
              title="Process Data", index=False, digits=5)
    cp, pp = out["cp"], out["pp"]
    res.table(pd.DataFrame({"Potential (within)": [cp["potential"], cp["lower"], cp["upper"], cp["k"], np.nan],
                            "Overall": [pp["potential"], pp["lower"], pp["upper"], pp["k"], out["cpm"]]},
                           index=pd.Index(["Cp / Pp", "CPL / PPL", "CPU / PPU", "Cpk / Ppk", "Cpm"], name="Index")),
              title="Capability Indices", digits=3)
    perf = pd.DataFrame({name: [lo, hi, lo + hi] for name, (lo, hi) in out["ppm"].items()},
                        index=pd.Index(["PPM < LSL", "PPM > USL", "PPM Total"], name="Performance"))
    res.table(perf, title="Performance", digits=2)


CAPABILITY_PARAMS = [MEASUREMENTS, SUBGROUP_SIZE, SUBGROUP_COLUMN, LSL, USL, TARGET]


@analysis("quality.capability_normal", menu=CAPABILITY_MENU, title="Capability Analysis (Normal)",
          params=CAPABILITY_PARAMS)
def capability(ws, column, subgroup_size=5, subgroup_col=None, lsl="", usl="", target=""):
    low, high = spec_limits(lsl, usl)
    goal = optional_number(target)
    groups = make_subgroups(ws, column, subgroup_size, subgroup_col)
    out = capability_normal(groups, low, high, goal)
    fig = new_figure(7.0, 4.2)
    capability_histogram(fig.add_subplot(), np.concatenate(groups), out, low, high, goal,
                         f"{tr('Process Capability')}: {column}")
    res = Result("Capability Analysis (Normal)").figure(fig)
    capability_tables(res, out, low, high, goal)
    return res


@analysis("quality.capability_between_within", menu=CAPABILITY_MENU, title="Capability Analysis (Between/Within)",
          params=CAPABILITY_PARAMS)
def capability_between_within(ws, column, subgroup_size=5, subgroup_col=None, lsl="", usl="", target=""):
    low, high = spec_limits(lsl, usl)
    goal = optional_number(target)
    groups = make_subgroups(ws, column, subgroup_size, subgroup_col)
    parts = between_within(groups)
    out = capability_normal(groups, low, high, goal)
    # the potential capability counts the variation between subgroups as part of the process
    out["within"] = parts["total"]
    out["cp"] = capability_indices(out["mean"], parts["total"], low, high)
    out["ppm"]["Expected within"] = (
        ppm(sps.norm.cdf(low, out["mean"], parts["total"])) if low is not None else 0.0,
        ppm(sps.norm.sf(high, out["mean"], parts["total"])) if high is not None else 0.0)
    fig = new_figure(7.0, 4.2)
    capability_histogram(fig.add_subplot(), np.concatenate(groups), out, low, high, goal,
                         f"{tr('Process Capability')}: {column}")
    res = Result("Capability Analysis (Between/Within)").figure(fig)
    res.table(pd.DataFrame({"StDev": [parts["between"], parts["within"], parts["total"], out["overall"]]},
                           index=pd.Index(["Between", "Within", "Between/Within", "Overall"], name="Source")),
              title="Estimated standard deviations", digits=5)
    res.text(tr("In the tables below, within stands for the between/within standard deviation."))
    capability_tables(res, out, low, high, goal)
    return res


@analysis("quality.capability_multiple", menu=CAPABILITY_MENU, title="Capability Analysis for Multiple Variables",
          params=[Param("columns", "Variables", "columns"), SUBGROUP_SIZE, LSL, USL, TARGET])
def capability_multiple(ws, columns, subgroup_size=5, lsl="", usl="", target=""):
    low, high = spec_limits(lsl, usl)
    goal = optional_number(target)
    rows, outs = {}, {}
    for name in columns:
        groups = make_subgroups(ws, name, subgroup_size)
        out = outs[name] = capability_normal(groups, low, high, goal)
        rows[name] = {"N": out["n"], "Mean": out["mean"], "StDev (within)": out["within"],
                      "StDev (overall)": out["overall"], "Cp": out["cp"]["potential"], "Cpk": out["cp"]["k"],
                      "Pp": out["pp"]["potential"], "Ppk": out["pp"]["k"], "Cpm": out["cpm"],
                      "PPM Total (overall)": sum(out["ppm"]["Expected overall"])}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Variable"
    fig = new_figure(max(6.0, 3.2 * len(columns)), 3.6)
    axes = fig.subplots(1, len(columns), squeeze=False)[0]
    for ax, name in zip(axes, columns):
        capability_histogram(ax, clean(ws.df[name]), outs[name], low, high, goal, name)
    res = Result("Capability Analysis for Multiple Variables")
    return res.table(table, digits=3).figure(fig)


NONNORMAL = [n for n in distfit.NAMES if n != "Normal"]


def capability_nonnormal(x, dist, lsl, usl) -> dict:
    """Overall capability by the percentile (ISO) method."""
    q_low, median, q_high = (float(dist.ppf(q)) for q in (0.00135, 0.5, 0.99865))
    pp = (usl - lsl) / (q_high - q_low) if lsl is not None and usl is not None else np.nan
    ppl = (median - lsl) / (median - q_low) if lsl is not None else np.nan
    ppu = (usl - median) / (q_high - median) if usl is not None else np.nan
    below = float(dist.cdf(lsl)) if lsl is not None else 0.0
    above = float(dist.sf(usl)) if usl is not None else 0.0
    observed = ((x < lsl).mean() if lsl is not None else 0.0, (x > usl).mean() if usl is not None else 0.0)
    return {"Pp": pp, "PPL": ppl, "PPU": ppu, "Ppk": float(np.nanmin([ppl, ppu])),
            "ppm": {"Observed": (ppm(observed[0]), ppm(observed[1])), "Expected overall": (ppm(below), ppm(above))}}


@analysis("quality.capability_nonnormal", menu=CAPABILITY_MENU, title="Capability Analysis (Nonnormal)",
          params=[MEASUREMENTS, Param("distribution", "Distribution", "choice", default="Weibull", choices=NONNORMAL),
                  LSL, USL])
def capability_nonnormal_analysis(ws, column, distribution="Weibull", lsl="", usl=""):
    low, high = spec_limits(lsl, usl)
    x = clean(ws.df[column])
    f = distfit.fit(x, distribution)
    out = capability_nonnormal(x, f.dist, low, high)
    fig = new_figure(7.0, 4.2)
    ax = fig.add_subplot()
    ax.hist(x, bins="auto", density=True, color=BLUE, edgecolor="white")
    limits = [v for v in (low, high) if v is not None] + [x.min(), x.max()]
    grid = np.linspace(min(limits), max(limits) + 0.1 * (max(limits) - min(limits)), 300)
    ax.plot(grid, f.dist.pdf(grid), color=RED)
    _spec_lines(ax, low, high, None)
    ax.set_title(f"{tr('Process Capability')}: {column} ({tr(distribution)})", pad=14), ax.set_yticks([])
    res = Result("Capability Analysis (Nonnormal)").figure(fig)
    res.table(fits_table([f]), title="Distribution")
    res.table(pd.DataFrame([{k: out[k] for k in ("Pp", "PPL", "PPU", "Ppk")}]), title="Overall Capability",
              index=False, digits=3)
    perf = pd.DataFrame({name: [lo, hi, lo + hi] for name, (lo, hi) in out["ppm"].items()},
                        index=pd.Index(["PPM < LSL", "PPM > USL", "PPM Total"], name="Performance"))
    return res.table(perf, title="Performance", digits=2)


ATTRIBUTE_PARAMS = [Param("column", "Counts", "column"),
                    Param("size_col", "Subgroup sizes (column)", "column", optional=True),
                    Param("size", "Subgroup size (if constant)", "int", default=50, minimum=1, maximum=10**9),
                    CONFIDENCE]


def _attribute_capability(title, column, lim, ylabel, summary: dict) -> Result:
    fig = new_figure(8.0, 5.5)
    top, bottom = fig.subplots(2, 1)
    draw_chart(top, lim["values"], lim["center"], lim["lcl"], lim["ucl"], tr(ylabel))
    top.set_title(f"{tr(title)}: {column}")
    cumulative = np.cumsum(lim["counts"]) / np.cumsum(lim["sizes"])
    bottom.plot(np.arange(1, len(cumulative) + 1), cumulative, "-o", color=BLUE, markersize=4)
    bottom.axhline(lim["center"], color=RED, linewidth=1)
    bottom.set_xlabel(tr("Sample")), bottom.set_ylabel(tr("Cumulative"))
    return Result(title).figure(fig).table(pd.DataFrame([summary]), title="Summary", index=False)


@analysis("quality.capability_binomial", menu=CAPABILITY_MENU, title="Capability Analysis (Binomial)",
          params=ATTRIBUTE_PARAMS)
def capability_binomial(ws, column, size_col=None, size=50, confidence=95.0):
    counts, sizes = attribute_data(ws, column, size_col, size)
    lim = dict(p_limits(counts, sizes), counts=counts, sizes=sizes)
    p_bar = lim["center"]
    lo, hi = clopper_pearson(int(counts.sum()), int(sizes.sum()), confidence)
    z = float(sps.norm.isf(p_bar)) if 0 < p_bar < 1 else np.nan
    return _attribute_capability("Capability Analysis (Binomial)", column, lim, "Proportion", {
        "% Defective": 100 * p_bar, "CI Lower": 100 * lo, "CI Upper": 100 * hi, "PPM Defective": ppm(p_bar),
        "Process Z": z})


@analysis("quality.capability_poisson", menu=CAPABILITY_MENU, title="Capability Analysis (Poisson)",
          params=ATTRIBUTE_PARAMS)
def capability_poisson(ws, column, size_col=None, size=1, confidence=95.0):
    counts, sizes = attribute_data(ws, column, size_col, size)
    lim = dict(u_limits(counts, sizes), counts=counts, sizes=sizes)
    lo, hi = poisson_rate_ci(int(counts.sum()), float(sizes.sum()), confidence)
    return _attribute_capability("Capability Analysis (Poisson)", column, lim, "Count per unit", {
        "Mean DPU": lim["center"], "CI Lower": lo, "CI Upper": hi, "Min DPU": float(lim["values"].min()),
        "Max DPU": float(lim["values"].max())})


@analysis("quality.sixpack", menu=CAPABILITY_MENU, title="Capability Sixpack (Normal)", params=CAPABILITY_PARAMS)
def sixpack(ws, column, subgroup_size=5, subgroup_col=None, lsl="", usl="", target=""):
    low, high = spec_limits(lsl, usl)
    goal = optional_number(target)
    groups = make_subgroups(ws, column, subgroup_size, subgroup_col)
    out = capability_normal(groups, low, high, goal)
    x = np.concatenate(groups)
    individuals = all(len(g) == 1 for g in groups)
    fig = new_figure(11.0, 7.0)
    axes = fig.subplots(2, 3)
    if individuals:
        lim = imr_limits(x)
        draw_chart(axes[0][0], x, lim["center"], lim["lcl"], lim["ucl"], tr("Individual value"))
        draw_chart(axes[1][0], lim["mr"], lim["mr_center"], lim["mr_lcl"], lim["mr_ucl"], tr("Moving range"))
    else:
        if any(len(g) < 2 for g in groups):
            raise ValueError("Subgroups need at least two observations; use an individuals chart")
        sigma = _spc.sigma_rbar(groups)
        xbar, rng = xbar_limits(groups, sigma), range_limits(groups, sigma)
        draw_chart(axes[0][0], xbar["values"], xbar["center"], xbar["lcl"], xbar["ucl"], tr("Sample mean"))
        draw_chart(axes[1][0], rng["values"], rng["center"], rng["lcl"], rng["ucl"], tr("Sample range"))
    last = groups[-25:]
    for i, g in enumerate(last):
        axes[0][1].plot(np.full(len(g), i + 1), g, "o", color=BLUE, markersize=3)
    axes[0][1].axhline(out["mean"], color=GRAY, linewidth=1)
    axes[0][1].set_title(tr("Last 25 subgroups"), fontsize=9)
    capability_histogram(axes[0][2], x, out, low, high, goal, tr("Capability histogram"))
    quantile_plot(axes[1][1], x, sps.norm(out["mean"], out["overall"]), tr("Normal Probability Plot"))
    ad, p = anderson_darling_normal(x)
    axes[1][1].set_xlabel(f"AD = {ad:.3f}, P = {'<0.005' if p < 0.005 else f'{p:.3f}'}", fontsize=8)
    text_ax = axes[1][2]
    text_ax.set_axis_off()
    cp, pp = out["cp"], out["pp"]
    lines = [tr("Within"), f"  {tr('StDev')}  {out['within']:.5g}", f"  Cp   {cp['potential']:.3f}",
             f"  Cpk  {cp['k']:.3f}", "", tr("Overall"), f"  {tr('StDev')}  {out['overall']:.5g}",
             f"  Pp   {pp['potential']:.3f}", f"  Ppk  {pp['k']:.3f}", f"  Cpm  {out['cpm']:.3f}"]
    text_ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10)
    text_ax.set_title(tr("Capability Indices"), fontsize=9)
    fig.suptitle(f"{tr('Capability Sixpack (Normal)')}: {column}")
    res = Result("Capability Sixpack (Normal)").figure(fig)
    capability_tables(res, out, low, high, goal)
    return res


# ----- tolerance intervals and acceptance sampling -----

def tolerance_factor(n: int, coverage: float, confidence: float) -> float:
    """Two-sided normal tolerance factor (Howe's approximation)."""
    nu = n - 1
    z = sps.norm.ppf((1 + coverage) / 2)
    return float(np.sqrt(nu * (1 + 1 / n) * z ** 2 / sps.chi2.ppf(1 - confidence, nu)))


def one_sided_tolerance_factor(n: int, coverage: float, confidence: float) -> float:
    """Exact one-sided normal tolerance factor, from the noncentral t distribution."""
    return float(sps.nct.ppf(confidence, n - 1, sps.norm.ppf(coverage) * np.sqrt(n)) / np.sqrt(n))


TOLERANCE_SIDES = ["Two-sided", "Lower bound", "Upper bound"]
TOLERANCE_DISTRIBUTIONS = ["Normal", "Lognormal"]


@analysis("quality.tolerance", menu=MENU, title="Tolerance Intervals",
          params=[Param("columns", "Variables", "columns"), CONFIDENCE,
                  Param("coverage", "Minimum percentage of population in interval", "number", default=95.0,
                        minimum=1, maximum=99.999),
                  Param("side", "Interval", "choice", default=TOLERANCE_SIDES[0], choices=TOLERANCE_SIDES),
                  Param("distribution", "Distribution", "choice", default=TOLERANCE_DISTRIBUTIONS[0],
                        choices=TOLERANCE_DISTRIBUTIONS)])
def tolerance_intervals(ws, columns, confidence=95.0, coverage=95.0, side=TOLERANCE_SIDES[0],
                        distribution=TOLERANCE_DISTRIBUTIONS[0]):
    conf, cov = confidence / 100, coverage / 100
    lognormal = distribution == TOLERANCE_DISTRIBUTIONS[1]
    parametric, nonpar = {}, {}
    for name in columns:
        x = clean(ws.df[name])
        n = len(x)
        if n < 2:
            raise ValueError(f"{name}: at least two observations are needed")
        if lognormal and (x <= 0).any():
            raise ValueError(f"{name}: the lognormal distribution needs positive data")
        y = np.log(x) if lognormal else x   # a lognormal interval is the normal one on the logarithms
        mean, sd = y.mean(), y.std(ddof=1)
        k = tolerance_factor(n, cov, conf) if side == TOLERANCE_SIDES[0] else one_sided_tolerance_factor(n, cov, conf)
        lower = mean - k * sd if side != TOLERANCE_SIDES[2] else -np.inf
        upper = mean + k * sd if side != TOLERANCE_SIDES[1] else np.inf
        if lognormal:
            lower, upper = np.exp(lower), np.exp(upper)
        parametric[name] = {"N": n, "Mean": x.mean(), "StDev": x.std(ddof=1), "Lower": lower, "Upper": upper, "k": k}
        if side == TOLERANCE_SIDES[0]:
            achieved = 1 - n * cov ** (n - 1) + (n - 1) * cov ** n
            nonpar[name] = {"Lower": x.min(), "Upper": x.max(), "Achieved confidence (%)": 100 * achieved}
        else:   # the smallest (largest) observation bounds the population with confidence 1 - coverage^n
            nonpar[name] = {"Lower": x.min() if side == TOLERANCE_SIDES[1] else -np.inf,
                            "Upper": x.max() if side == TOLERANCE_SIDES[2] else np.inf,
                            "Achieved confidence (%)": 100 * (1 - cov ** n)}
    res = Result("Tolerance Intervals")
    res.text(f"{tr('Confidence level (%)')}: {confidence:g}    "
             f"{tr('Minimum percentage of population in interval')}: {coverage:g}    {tr(side)}")
    table = pd.DataFrame.from_dict(parametric, orient="index")
    table.index.name = "Variable"
    res.table(table, title="Lognormal method" if lognormal else "Normal method")
    table = pd.DataFrame.from_dict(nonpar, orient="index")
    table.index.name = "Variable"
    return res.table(table, title="Nonparametric method (smallest and largest observation)")


def sampling_plan(aql: float, rql: float, alpha: float = 0.05, beta: float = 0.10) -> tuple[int, int]:
    """Smallest single sampling plan (n, c) meeting both risks under the binomial model."""
    if not 0 < aql < rql < 1:
        raise ValueError("The AQL must be less than the RQL, both between 0 and 100%")
    for c in range(0, 2000):
        low, high = c + 1, c + 2
        while sps.binom.cdf(c, high, rql) > beta:
            high *= 2
            if high > 10**7:
                raise ValueError("No sampling plan found")
        while low < high:  # smallest n with consumer's risk <= beta
            mid = (low + high) // 2
            if sps.binom.cdf(c, mid, rql) <= beta:
                high = mid
            else:
                low = mid + 1
        if sps.binom.cdf(c, low, aql) >= 1 - alpha:
            return low, c
    raise ValueError("No sampling plan found")


@analysis("quality.acceptance", menu=MENU, title="Acceptance Sampling by Attributes",
          params=[Param("aql", "Acceptable quality level (AQL, %)", "number", default=1.0, minimum=1e-6, maximum=99),
                  Param("rql", "Rejectable quality level (RQL, %)", "number", default=5.0, minimum=1e-6, maximum=99.9),
                  Param("alpha", "Producer's risk (alpha)", "number", default=0.05, minimum=0.0001, maximum=0.5),
                  Param("beta", "Consumer's risk (beta)", "number", default=0.10, minimum=0.0001, maximum=0.5)])
def acceptance_sampling(ws, aql=1.0, rql=5.0, alpha=0.05, beta=0.10):
    n, c = sampling_plan(aql / 100, rql / 100, alpha, beta)
    grid = np.linspace(0, min(1.0, 2.5 * rql / 100), 200)
    fig = new_figure()
    ax = fig.add_subplot()
    ax.plot(100 * grid, sps.binom.cdf(c, n, grid), color=BLUE)
    for level in (aql, rql):
        ax.plot(level, sps.binom.cdf(c, n, level / 100), "o", color=RED)
    ax.set_xlabel(tr("Lot percent defective")), ax.set_ylabel(tr("Probability of acceptance"))
    ax.set_title(f"{tr('Operating Characteristic (OC) Curve')}  (n = {n}, c = {c})")
    res = Result("Acceptance Sampling by Attributes")
    res.table(pd.DataFrame([{"Sample size": n, "Acceptance number": c}]), title="Generated plan", index=False)
    res.text(f"{tr('Accept the lot if the number of defectives in the sample is at most')} {c}.")
    res.table(pd.DataFrame({"Probability of accepting": [sps.binom.cdf(c, n, aql / 100),
                                                         sps.binom.cdf(c, n, rql / 100)],
                            "Probability of rejecting": [sps.binom.sf(c, n, aql / 100),
                                                         sps.binom.sf(c, n, rql / 100)]},
                           index=pd.Index([f"AQL {aql:g}%", f"RQL {rql:g}%"], name="Percent defective")))
    return res.figure(fig)


def variables_plan(aql: float, rql: float, alpha=0.05, beta=0.10, sigma_known=False) -> tuple[int, float]:
    """Sample size n and critical distance k of a single-limit variables plan (normal data).

    The lot is accepted when the mean is at least k standard deviations inside the limit.
    """
    if not 0 < aql < rql < 1:
        raise ValueError("The AQL must be less than the RQL, both between 0 and 100%")
    z_alpha, z_beta = sps.norm.isf(alpha), sps.norm.isf(beta)
    z_aql, z_rql = sps.norm.isf(aql), sps.norm.isf(rql)
    k = (z_alpha * z_rql + z_beta * z_aql) / (z_alpha + z_beta)
    n = ((z_alpha + z_beta) / (z_aql - z_rql)) ** 2
    if not sigma_known:
        n *= 1 + k ** 2 / 2
    return max(2, int(np.ceil(n))), float(k)


def variables_acceptance(defective, n: int, k: float, sigma_known=False):
    """Probability of accepting a lot with the given fraction defective."""
    z = sps.norm.isf(np.asarray(defective, dtype=float))
    if sigma_known:
        return sps.norm.cdf(np.sqrt(n) * (z - k))
    return sps.nct.sf(k * np.sqrt(n), n - 1, z * np.sqrt(n))


@analysis("quality.acceptance_variables", menu=MENU, title="Acceptance Sampling by Variables",
          params=[Param("aql", "Acceptable quality level (AQL, %)", "number", default=1.0, minimum=1e-6, maximum=99),
                  Param("rql", "Rejectable quality level (RQL, %)", "number", default=5.0, minimum=1e-6, maximum=99.9),
                  Param("alpha", "Producer's risk (alpha)", "number", default=0.05, minimum=0.0001, maximum=0.5),
                  Param("beta", "Consumer's risk (beta)", "number", default=0.10, minimum=0.0001, maximum=0.5),
                  Param("sigma", "Historical standard deviation (0 = unknown)", "number", default=0.0, minimum=0.0),
                  Param("column", "Measurements of a lot to accept or reject", "column", optional=True), LSL, USL])
def acceptance_by_variables(ws, aql=1.0, rql=5.0, alpha=0.05, beta=0.10, sigma=0.0, column=None, lsl="", usl=""):
    known = sigma > 0
    n, k = variables_plan(aql / 100, rql / 100, alpha, beta, known)
    res = Result("Acceptance Sampling by Variables")
    res.table(pd.DataFrame([{"Sample size": n, "Critical distance (k)": k}]), title="Generated plan", index=False)
    res.text(tr("Accept the lot if (mean - lower spec) / StDev and (upper spec - mean) / StDev are at least k."))
    levels = np.array([aql, rql]) / 100
    accept = variables_acceptance(levels, n, k, known)
    res.table(pd.DataFrame({"Probability of accepting": accept, "Probability of rejecting": 1 - accept},
                           index=pd.Index([f"AQL {aql:g}%", f"RQL {rql:g}%"], name="Percent defective")))
    grid = np.linspace(1e-6, min(0.999, 2.5 * rql / 100), 200)
    fig = new_figure()
    ax = fig.add_subplot()
    ax.plot(100 * grid, variables_acceptance(grid, n, k, known), color=BLUE)
    ax.plot([aql, rql], accept, "o", color=RED)
    ax.set_xlabel(tr("Lot percent defective")), ax.set_ylabel(tr("Probability of acceptance"))
    ax.set_title(f"{tr('Operating Characteristic (OC) Curve')}  (n = {n}, k = {k:.4g})")
    res.figure(fig)
    if column:
        low, high = spec_limits(lsl, usl)
        x = clean(ws.df[column])
        if len(x) < 2:
            raise ValueError("At least two observations are needed")
        mean, sd = float(x.mean()), sigma if known else float(x.std(ddof=1))
        distances = {name: value for name, value in (("Z.LSL", (mean - low) / sd if low is not None else None),
                                                     ("Z.USL", (high - mean) / sd if high is not None else None))
                     if value is not None}
        res.table(pd.DataFrame([{"N": len(x), "Mean": mean, "StDev": sd, **distances}]),
                  title=f"{tr('Lot')}: {column}", index=False)
        ok = min(distances.values()) >= k
        res.text(f"{tr('Decision')}: {tr('accept the lot') if ok else tr('reject the lot')}"
                 + (f"    ({tr('the plan asks for')} {n} {tr('measurements')})" if len(x) < n else ""))
    return res


# ----- multi-vari and symmetry -----

@analysis("quality.multi_vari", menu=MENU, title="Multi-Vari Chart",
          params=[Param("response", "Response", "column"),
                  Param("factor1", "Factor 1 (points within a group)", "column", numeric=False),
                  Param("factor2", "Factor 2 (groups)", "column", numeric=False, optional=True),
                  Param("factor3", "Factor 3 (panels)", "column", numeric=False, optional=True)])
def multi_vari(ws, response, factor1, factor2=None, factor3=None):
    factors = [f for f in (factor1, factor2, factor3) if f]
    if len(set(factors)) != len(factors):
        raise ValueError("Select a different column for each factor")
    frame = ws.df[[response] + factors].dropna()
    labels = {f: frame[f].map(format_value) for f in factors}
    y = frame[response].astype(float)
    panels = ws.levels(factor3, labels[factor3]) if factor3 else [None]
    outer = ws.levels(factor2, labels[factor2]) if factor2 else [None]
    inner = ws.levels(factor1, labels[factor1])
    fig = new_figure(max(6.0, 0.5 * len(inner) * len(outer) * len(panels) + 2), 4.2)
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    step = len(inner) + 1
    for ax, panel in zip(axes, panels):
        in_panel = labels[factor3] == panel if factor3 else pd.Series(True, index=frame.index)
        group_means = []
        for g, group in enumerate(outer):
            in_group = in_panel & (labels[factor2] == group if factor2 else True)
            positions = g * step + np.arange(len(inner))
            means = [y[in_group & (labels[factor1] == level)].mean() for level in inner]
            ax.plot(positions, means, "-o", color=BLUE, markersize=5)   # the means of factor 1 inside the group
            group_means.append(y[in_group].mean())
        centers = np.arange(len(outer)) * step + (len(inner) - 1) / 2
        if factor2:
            ax.plot(centers, group_means, "--s", color=RED, markersize=6)
        ticks = [g * step + i for g in range(len(outer)) for i in range(len(inner))]
        ax.set_xticks(ticks, inner * len(outer), fontsize=7)
        if factor2:
            for center, group in zip(centers, outer):
                ax.annotate(group, (center, -0.14), xycoords=("data", "axes fraction"), ha="center", fontsize=8)
        if factor3:
            ax.set_title(f"{factor3} = {panel}", fontsize=9)
    axes[0].set_ylabel(response)
    fig.suptitle(f"{tr('Multi-Vari Chart')}: {response} {tr('by')} {', '.join(factors)}")
    table = y.groupby([labels[f] for f in reversed(factors)]).agg(["count", "mean", "std"])
    table.columns = ["N", "Mean", "StDev"]
    ranks = [{v: i for i, v in enumerate(ws.levels(f, labels[f]))} for f in reversed(factors)]
    table = table.loc[sorted(table.index, key=lambda k: [r[v] for r, v in zip(ranks, k if isinstance(k, tuple) else (k,))])]
    table.index = [", ".join(map(str, i)) if isinstance(i, tuple) else i for i in table.index]
    table.index.name = ", ".join(reversed(factors))
    return Result("Multi-Vari Chart").figure(fig).table(table)


def symmetry_points(values) -> tuple[np.ndarray, np.ndarray]:
    """Distances below and above the median of the i-th smallest and i-th largest observations."""
    x = np.sort(clean(values))
    if len(x) < 4:
        raise ValueError("At least four observations are needed")
    median, half = np.median(x), len(x) // 2
    return median - x[:half], x[::-1][:half] - median


@analysis("quality.symmetry", menu=MENU, title="Symmetry Plot", params=[Param("columns", "Variables", "columns")])
def symmetry_plot(ws, columns):
    fig = new_figure(max(5.0, 4.0 * len(columns)), 4.0)
    axes = fig.subplots(1, len(columns), squeeze=False)[0]
    rows = {}
    for ax, name in zip(axes, columns):
        below, above = symmetry_points(ws.df[name])
        limit = max(below.max(), above.max())
        ax.plot(below, above, "o", color=BLUE, markersize=4)
        ax.plot([0, limit], [0, limit], color=RED, linewidth=1)
        ax.set_xlabel(tr("Distance below the median")), ax.set_ylabel(tr("Distance above the median"))
        ax.set_title(name, fontsize=9)
        x = clean(ws.df[name])
        rows[name] = {"N": len(x), "Median": float(np.median(x)), "Skewness": float(sps.skew(x, bias=False)),
                      "Points above the line (%)": 100 * float((above > below).mean())}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Variable"
    res = Result("Symmetry Plot").figure(fig)
    res.text(tr("Points above the line show a longer upper tail; points below it, a longer lower tail."))
    return res.table(table)
