"""More graphs: binned scatterplot, parallel coordinates and probability distribution plot."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.graphs.basic import BLUE, MENU, PALETTE, RED, VARIABLES, _axes
from openstat.graphs.relation import GROUP, X, Y
from openstat.i18n import tr
from openstat.stats import distributions as dists


@analysis("graph.binned_scatter", menu=MENU, title="Binned Scatterplot",
          params=[Y, X, Param("bins", "Number of bins on each axis", "int", default=20, minimum=2, maximum=500)])
def binned_scatterplot(ws, y, x, bins=20):
    frame = ws.df[[y, x]].dropna().astype(float)
    if len(frame) < 2:
        raise ValueError("At least two observations are needed")
    counts, x_edges, y_edges = np.histogram2d(frame[x], frame[y], bins=bins)
    fig, ax = _axes(f"{tr('Binned Scatterplot')}: {y} vs {x}", x, y)
    shown = np.where(counts.T > 0, counts.T, np.nan)   # empty bins stay blank
    mesh = ax.pcolormesh(x_edges, y_edges, shown, cmap="Blues", vmin=0)
    fig.colorbar(mesh, ax=ax, label=tr("Count"))
    res = Result("Binned Scatterplot").figure(fig)
    res.text(f"N = {len(frame)}    {tr('Bins with data')}: {int((counts > 0).sum())} / {counts.size}    "
             f"{tr('Largest count in a bin')}: {int(counts.max())}")
    return res


MAX_LINES = 2000


@analysis("graph.parallel", menu=MENU, title="Parallel Coordinates Plot",
          params=[VARIABLES, GROUP,
                  Param("standardize", "Put every variable on a 0 to 1 scale", "bool", default=True)])
def parallel_coordinates(ws, columns, by=None, standardize=True):
    if len(columns) < 2:
        raise ValueError("Select at least two columns")
    frame = ws.df[list(columns) + ([by] if by else [])].dropna()
    values = frame[list(columns)].astype(float)
    if standardize:
        spread = (values.max() - values.min()).replace(0, 1.0)
        values = (values - values.min()) / spread
    if len(values) > MAX_LINES:   # an even sample keeps the picture readable
        keep = np.linspace(0, len(values) - 1, MAX_LINES).astype(int)
        values, frame = values.iloc[keep], frame.iloc[keep]
    labels = frame[by].map(format_value) if by else pd.Series("", index=frame.index)
    fig, ax = _axes(tr("Parallel Coordinates Plot"), "", tr("Scaled value") if standardize else "",
                    width=max(6.0, 1.3 * len(columns)))
    positions = np.arange(len(columns))
    for i, level in enumerate(ws.levels(by, labels) if by else [""]):
        rows = values[(labels == level).to_numpy()].to_numpy()
        ax.plot(positions, rows.T, color=PALETTE[i % len(PALETTE)] if by else BLUE, alpha=0.5, linewidth=0.8)
        if by:
            ax.plot([], [], color=PALETTE[i % len(PALETTE)], label=level)
    for position in positions:
        ax.axvline(position, color="black", linewidth=0.6)
    ax.set_xticks(positions, list(columns))
    if by:
        ax.legend(title=by, fontsize=8)
    return Result("Parallel Coordinates Plot").figure(fig)


SHADES = ["None", "Left tail", "Right tail", "Both tails", "Middle"]


def shaded_region(dist, shade: str, probability: float) -> list[tuple[float, float]]:
    """Intervals of x that hold `probability` of the distribution, as (from, to) pairs.

    The tails of a discrete distribution take the whole values that fit in the probability asked for.
    """
    discrete = dists.is_discrete(dist)
    low, high = (float(v) for v in dist.support())

    def left(p):
        x = float(dist.ppf(p))
        return x - 1 if discrete and dist.cdf(x) > p + 1e-12 else x

    def right(p):   # for a discrete distribution isf gives the last value left out of the tail
        return float(dist.isf(p)) + (1 if discrete else 0)

    if shade == SHADES[1]:
        return [(low, left(probability))]
    if shade == SHADES[2]:
        return [(right(probability), high)]
    if shade == SHADES[3]:
        return [(low, left(probability / 2)), (right(probability / 2), high)]
    if shade == SHADES[4]:
        return [(float(dist.ppf((1 - probability) / 2)), float(dist.isf((1 - probability) / 2)))]
    return []


@analysis("graph.distribution", menu=MENU, title="Probability Distribution Plot",
          params=[Param("distribution", "Distribution", "choice", default=dists.NAMES[0], choices=dists.NAMES),
                  Param("p1", "Parameter 1", "number", default=0.0), Param("p2", "Parameter 2", "number", default=1.0),
                  Param("p3", "Parameter 3", "number", default=0.0),
                  Param("shade", "Shaded area", "choice", default=SHADES[0], choices=SHADES),
                  Param("probability", "Probability of the shaded area", "number", default=0.05, minimum=1e-9,
                        maximum=1 - 1e-9)])
def distribution_plot(ws, distribution=dists.NAMES[0], p1=0.0, p2=1.0, p3=0.0, shade=SHADES[0], probability=0.05):
    dist = dists.make(distribution, p1, p2, p3)
    discrete = dists.is_discrete(dist)
    start, stop = float(dist.ppf(0.0005)), float(dist.ppf(0.9995))
    fig, ax = _axes(f"{tr('Probability Distribution Plot')}\n{tr(distribution)}: {p1:g}, {p2:g}, {p3:g}", "X",
                    tr("Probability") if discrete else tr("Density"))
    regions = shaded_region(dist, shade, probability)
    if discrete:
        grid = np.arange(np.floor(start), np.ceil(stop) + 1)
        inside = np.zeros(len(grid), dtype=bool)
        for low, high in regions:
            inside |= (grid >= low) & (grid <= high)
        ax.bar(grid[~inside], dist.pmf(grid[~inside]), width=0.15, color=BLUE)
        ax.bar(grid[inside], dist.pmf(grid[inside]), width=0.15, color=RED)
    else:
        grid = np.linspace(start, stop, 500)
        ax.plot(grid, dist.pdf(grid), color=BLUE)
        for low, high in regions:
            part = grid[(grid >= low) & (grid <= high)]
            ax.fill_between(part, dist.pdf(part), color=RED, alpha=0.5)
    ax.set_ylim(bottom=0)
    res = Result("Probability Distribution Plot").figure(fig)
    table = pd.DataFrame([{"Mean": float(dist.mean()), "StDev": float(dist.std()), "Median": float(dist.median())}])
    res.table(table, index=False)
    if regions:
        shown = "; ".join(f"{low:.5g} {tr('to')} {high:.5g}" for low, high in regions)
        actual = sum(float(dist.cdf(high) - dist.cdf(low - 1 if discrete else low)) for low, high in regions)
        res.text(f"{tr(shade)}: {shown}    {tr('Probability')} = {actual:.5g}")
    return res
