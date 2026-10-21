"""Graphs of one variable or of groups: histogram, dotplot, boxplot, probability plot and others."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._util import anderson_darling_normal, clean, mean_ci, median_ranks, quartiles

MENU = ("Graph",)
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"
PALETTE = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#7030a0", "#c00000", "#00b0f0", "#7f7f7f"]

VARIABLES = Param("columns", "Variables", "columns")
BY = Param("by", "By variable", "column", numeric=False, optional=True)


def groups(ws, columns, by=None) -> list[tuple[str, np.ndarray]]:
    """(label, values) for each column, split by the levels of `by` when given."""
    out = []
    for name in columns:
        if by:
            labels = ws.df[by].map(format_value)
            for level in ws.levels(by):
                out.append((f"{name}\n{level}" if len(columns) > 1 else level,
                            clean(ws.df[name][labels == level])))
        else:
            out.append((name, clean(ws.df[name])))
    return [(label, x) for label, x in out if len(x)]


def _axes(title: str, xlabel: str = "", ylabel: str = "", width=6.0, height=4.0):
    fig = new_figure(width, height)
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def histogram_figure(values, name: str, fit_normal: bool = False, bins="auto"):
    values = clean(values)
    fig, ax = _axes(f"{tr('Histogram')}: {name}", name)
    ax.hist(values, bins=bins, density=fit_normal, color=BLUE, edgecolor="white")
    if fit_normal and len(values) > 1 and values.std(ddof=1) > 0:
        mean, sd = values.mean(), values.std(ddof=1)
        x = np.linspace(values.min() - sd, values.max() + sd, 200)
        ax.plot(x, sps.norm.pdf(x, mean, sd), color=RED)
        ax.set_ylabel(tr("Density"))
    else:
        ax.set_ylabel(tr("Frequency"))
    return fig


@analysis("graph.histogram", menu=MENU, title="Histogram",
          params=[VARIABLES, Param("fit_normal", "Fit normal curve", "bool", default=False)])
def histogram(ws, columns, fit_normal=False):
    res = Result("Histogram")
    for name in columns:
        res.figure(histogram_figure(ws.df[name], name, fit_normal))
    return res


def dotplot_figure(values, name: str):
    x = np.sort(clean(values))
    fig, ax = _axes(f"{tr('Dotplot')}: {name}", name, height=2.8)
    if len(x):
        width = (x.max() - x.min()) / 40 or 1.0
        bins = np.round((x - x.min()) / width).astype(int)
        heights = np.zeros(len(x))
        counts: dict[int, int] = {}
        for i, b in enumerate(bins):
            counts[b] = counts.get(b, 0) + 1
            heights[i] = counts[b]
        ax.plot(x.min() + bins * width, heights, "o", color=BLUE, markersize=5)
        ax.set_ylim(0, max(heights.max() + 1, 10))
    ax.set_yticks([])
    return fig


@analysis("graph.dotplot", menu=MENU, title="Dotplot", params=[VARIABLES])
def dotplot(ws, columns):
    res = Result("Dotplot")
    for name in columns:
        res.figure(dotplot_figure(ws.df[name], name))
    return res


def stem_and_leaf(values) -> str:
    x = np.sort(clean(values))
    if not len(x):
        return ""
    span = max(abs(x[0]), abs(x[-1])) or 1.0
    unit = 10.0 ** (np.floor(np.log10(span)) - 1)
    scaled = np.trunc(x / unit + (1e-9 * np.sign(x))).astype(int)
    stems = np.where(scaled < 0, -(-scaled // 10), scaled // 10)
    leaves = np.abs(scaled) % 10
    lines = [f"N = {len(x)}   {tr('Leaf unit')} = {unit:g}", ""]
    for stem in range(stems.min(), stems.max() + 1):
        row = leaves[stems == stem]
        row = np.sort(row)[::-1] if stem < 0 else np.sort(row)
        lines.append(f"{len(row):4d} {stem:5d} | {''.join(str(v) for v in row)}")
    return "\n".join(lines)


@analysis("graph.stem", menu=MENU, title="Stem-and-Leaf", params=[VARIABLES])
def stem(ws, columns):
    res = Result("Stem-and-Leaf")
    for name in columns:
        res.heading(name).pre(stem_and_leaf(ws.df[name]))
    return res


def box_stats(values, label: str) -> dict:
    """Box statistics with whiskers at the most extreme points within 1.5 IQR."""
    x = clean(values)
    q1, med, q3 = quartiles(x)
    iqr = q3 - q1
    inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    lo, hi = (inside.min(), inside.max()) if len(inside) else (q1, q3)
    return {"label": label, "q1": q1, "med": med, "q3": q3, "whislo": min(lo, q1), "whishi": max(hi, q3),
            "fliers": x[(x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)]}


@analysis("graph.boxplot", menu=MENU, title="Boxplot", params=[VARIABLES, BY])
def boxplot(ws, columns, by=None):
    data = groups(ws, columns, by)
    fig, ax = _axes(tr("Boxplot"), by or "", ", ".join(columns), width=max(6.0, 1.0 + 0.9 * len(data)))
    ax.bxp([box_stats(x, label) for label, x in data], patch_artist=True,
           boxprops={"facecolor": BLUE, "edgecolor": "black"}, medianprops={"color": "black"},
           flierprops={"marker": "*", "markeredgecolor": RED})
    return Result("Boxplot").figure(fig)


@analysis("graph.interval", menu=MENU, title="Interval Plot",
          params=[VARIABLES, BY,
                  Param("confidence", "Confidence level (%)", "number", default=95.0, minimum=1, maximum=99.999)])
def interval_plot(ws, columns, by=None, confidence=95.0):
    data = groups(ws, columns, by)
    fig, ax = _axes(f"{tr('Interval Plot')} ({confidence:g}% {tr('CI for the mean')})", by or "",
                    ", ".join(columns), width=max(6.0, 1.0 + 0.9 * len(data)))
    for i, (_label, x) in enumerate(data):
        mean, lo, hi = mean_ci(x, confidence / 100)
        if not np.isnan(lo):
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=BLUE, capsize=6)
        ax.plot(i, mean, "o", color=BLUE)
    ax.set_xticks(range(len(data)), [label for label, _ in data])
    ax.set_xlim(-0.6, len(data) - 0.4)
    return Result("Interval Plot").figure(fig)


@analysis("graph.individual", menu=MENU, title="Individual Value Plot", params=[VARIABLES, BY])
def individual_value_plot(ws, columns, by=None):
    data = groups(ws, columns, by)
    fig, ax = _axes(tr("Individual Value Plot"), by or "", ", ".join(columns),
                    width=max(6.0, 1.0 + 0.9 * len(data)))
    rng = np.random.default_rng(0)
    for i, (_label, x) in enumerate(data):
        ax.plot(i + rng.uniform(-0.12, 0.12, len(x)), x, "o", color=BLUE, alpha=0.75, markersize=5)
    ax.set_xticks(range(len(data)), [label for label, _ in data])
    ax.set_xlim(-0.6, len(data) - 0.4)
    return Result("Individual Value Plot").figure(fig)


def probability_plot_figure(values, name: str):
    """Normal probability plot with the fitted line and the Anderson-Darling test."""
    x = np.sort(clean(values))
    n = len(x)
    fig, ax = _axes(f"{tr('Probability Plot')}: {name}", name, tr("Percent"))
    if n < 2:
        return fig
    mean, sd = x.mean(), x.std(ddof=1)
    ax.plot(x, sps.norm.ppf(median_ranks(n)), "o", color=BLUE, markersize=4)
    if sd > 0:
        line = np.array([mean - 3.3 * sd, mean + 3.3 * sd])
        ax.plot(line, (line - mean) / sd, color=RED)
    ticks = np.array([0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9])
    ax.set_yticks(sps.norm.ppf(ticks / 100), [f"{t:g}" for t in ticks])
    ax.set_ylim(sps.norm.ppf(0.0005), sps.norm.ppf(0.9995))
    ax.grid(True, color="#dddddd")
    ad, p = anderson_darling_normal(x)
    p_text = "<0.005" if p < 0.005 else f"{p:.3f}"
    ax.text(0.98, 0.03, f"{tr('Mean')} {mean:.4g}\n{tr('StDev')} {sd:.4g}\nN {n}\nAD {ad:.3f}\nP {p_text}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": GRAY})
    return fig


@analysis("graph.probability", menu=MENU, title="Probability Plot", params=[VARIABLES])
def probability_plot(ws, columns):
    res = Result("Probability Plot")
    for name in columns:
        res.figure(probability_plot_figure(ws.df[name], name))
    return res


@analysis("graph.ecdf", menu=MENU, title="Empirical CDF", params=[VARIABLES])
def empirical_cdf(ws, columns):
    fig, ax = _axes(tr("Empirical CDF"), ", ".join(columns), tr("Percent"))
    for i, name in enumerate(columns):
        x = np.sort(clean(ws.df[name]))
        if not len(x):
            continue
        color = PALETTE[i % len(PALETTE)]
        ax.step(x, 100 * np.arange(1, len(x) + 1) / len(x), where="post", color=color, label=name)
        if len(x) > 1 and x.std(ddof=1) > 0:
            grid = np.linspace(x.min(), x.max(), 200)
            ax.plot(grid, 100 * sps.norm.cdf(grid, x.mean(), x.std(ddof=1)), color=color, linestyle="--")
    ax.legend()
    return Result("Empirical CDF").figure(fig)


BAR_FUNCTIONS = ["Count", "Mean", "Sum"]


def _category_values(ws, category, function, value):
    if function == "Count" or not value:
        return ws.df[category].dropna().map(format_value).value_counts(sort=False).sort_index()
    grouped = ws.df.groupby(ws.df[category].map(format_value))[value]
    return grouped.mean() if function == "Mean" else grouped.sum()


@analysis("graph.bar", menu=MENU, title="Bar Chart",
          params=[Param("category", "Categorical variable", "column", numeric=False),
                  Param("function", "Bars represent", "choice", default="Count", choices=BAR_FUNCTIONS),
                  Param("value", "Value variable", "column", optional=True)])
def bar_chart(ws, category, function="Count", value=None):
    series = _category_values(ws, category, function, value)
    ylabel = tr("Count") if function == "Count" or not value else f"{tr(function)} ({value})"
    fig, ax = _axes(tr("Bar Chart"), category, ylabel, width=max(6.0, 1.0 + 0.6 * len(series)))
    ax.bar([str(i) for i in series.index], series.to_numpy(), color=BLUE)
    return Result("Bar Chart").figure(fig)


@analysis("graph.pie", menu=MENU, title="Pie Chart",
          params=[Param("category", "Categorical variable", "column", numeric=False)])
def pie_chart(ws, category):
    series = _category_values(ws, category, "Count", None)
    fig, ax = _axes(f"{tr('Pie Chart')}: {category}")
    ax.pie(series.to_numpy(), labels=[str(i) for i in series.index], autopct="%1.1f%%", colors=PALETTE)
    return Result("Pie Chart").figure(fig)
