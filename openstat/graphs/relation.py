"""Graphs of relationships: scatter, matrix, series, contour and 3D."""
from __future__ import annotations

import numpy as np

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.graphs.basic import BLUE, MENU, PALETTE, RED, VARIABLES, _axes
from openstat.i18n import tr

Y = Param("y", "Y variable", "column")
X = Param("x", "X variable", "column")
Z = Param("z", "Z variable", "column")
GROUP = Param("by", "Groups", "column", numeric=False, optional=True)


@analysis("graph.scatter", menu=MENU, title="Scatterplot",
          params=[Y, X, GROUP, Param("fit", "Add regression line", "bool", default=False)])
def scatterplot(ws, y, x, by=None, fit=False):
    cols = [y, x] + ([by] if by else [])
    frame = ws.df[cols].dropna()
    fig, ax = _axes(f"{tr('Scatterplot')}: {y} vs {x}", x, y)
    parts = [(None, frame)] if not by else [(k, frame[frame[by].map(format_value) == k]) for k in ws.levels(by, frame[by].map(format_value))]
    for i, (label, g) in enumerate(parts):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(g[x], g[y], "o", color=color, markersize=5, label=label)
        if fit and len(g) > 1 and g[x].nunique() > 1:
            slope, intercept = np.polyfit(g[x].astype(float), g[y].astype(float), 1)
            line = np.array([g[x].min(), g[x].max()], dtype=float)
            ax.plot(line, intercept + slope * line, color=color if by else RED)
    if by:
        ax.legend(title=by)
    return Result("Scatterplot").figure(fig)


@analysis("graph.matrix", menu=MENU, title="Matrix Plot", params=[VARIABLES])
def matrix_plot(ws, columns):
    frame = ws.data(columns).astype(float)
    k = len(columns)
    fig = new_figure(max(5.0, 2.2 * k), max(4.5, 2.0 * k))
    axes = fig.subplots(k, k, squeeze=False)
    for i, row in enumerate(columns):
        for j, col in enumerate(columns):
            ax = axes[i][j]
            if i == j:
                ax.hist(frame[col], bins="auto", color=BLUE, edgecolor="white")
            else:
                ax.plot(frame[col], frame[row], "o", color=BLUE, markersize=3)
            if i == k - 1:
                ax.set_xlabel(col)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(row)
            else:
                ax.set_yticklabels([])
    fig.suptitle(tr("Matrix Plot"))
    return Result("Matrix Plot").figure(fig)


@analysis("graph.bubble", menu=MENU, title="Bubble Plot",
          params=[Y, X, Param("size", "Bubble size variable", "column")])
def bubble_plot(ws, y, x, size):
    frame = ws.df[[y, x, size]].dropna().astype(float)
    s = frame[size] - frame[size].min()
    s = 30 + 570 * (s / s.max() if s.max() > 0 else 0.5)
    fig, ax = _axes(f"{tr('Bubble Plot')}: {y} vs {x} ({size})", x, y)
    ax.scatter(frame[x], frame[y], s=s, color=BLUE, alpha=0.5, edgecolor="black")
    return Result("Bubble Plot").figure(fig)


@analysis("graph.marginal", menu=MENU, title="Marginal Plot", params=[Y, X])
def marginal_plot(ws, y, x):
    frame = ws.df[[y, x]].dropna().astype(float)
    fig = new_figure(6.0, 5.5)
    grid = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    main = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=main)
    right = fig.add_subplot(grid[1, 1], sharey=main)
    main.plot(frame[x], frame[y], "o", color=BLUE, markersize=5)
    main.set_xlabel(x)
    main.set_ylabel(y)
    top.hist(frame[x], bins="auto", color=BLUE, edgecolor="white")
    right.hist(frame[y], bins="auto", color=BLUE, edgecolor="white", orientation="horizontal")
    top.tick_params(labelbottom=False)
    right.tick_params(labelleft=False)
    fig.suptitle(f"{tr('Marginal Plot')}: {y} vs {x}")
    return Result("Marginal Plot").figure(fig)


@analysis("graph.timeseries", menu=MENU, title="Time Series Plot",
          params=[VARIABLES, Param("stamp", "Time stamp", "column", numeric=False, optional=True)])
def time_series_plot(ws, columns, stamp=None):
    fig, ax = _axes(tr("Time Series Plot"), stamp or tr("Index"), ", ".join(columns), width=7.5)
    index = np.arange(1, ws.nrows + 1)
    for i, name in enumerate(columns):
        ax.plot(index, ws.df[name].astype(float), "-o", color=PALETTE[i % len(PALETTE)], markersize=3, label=name)
    if stamp:
        step = max(1, ws.nrows // 12)
        ax.set_xticks(index[::step], [format_value(v) for v in ws.df[stamp].tolist()[::step]], rotation=45)
    if len(columns) > 1:
        ax.legend()
    return Result("Time Series Plot").figure(fig)


@analysis("graph.line", menu=MENU, title="Line Plot",
          params=[Param("y", "Y variable", "column"),
                  Param("category", "Categorical variable", "column", numeric=False),
                  GROUP])
def line_plot(ws, y, category, by=None):
    """Mean of Y at each level of a categorical variable."""
    cols = [y, category] + ([by] if by else [])
    frame = ws.df[cols].dropna()
    fig, ax = _axes(f"{tr('Line Plot')}: {tr('Mean')} ({y})", category, f"{tr('Mean')} ({y})")
    parts = [(None, frame)] if not by else [(k, frame[frame[by].map(format_value) == k]) for k in ws.levels(by, frame[by].map(format_value))]
    levels = ws.levels(category, frame[category].map(format_value))
    for i, (label, g) in enumerate(parts):
        means = g.groupby(g[category].map(format_value))[y].mean().reindex(levels)
        ax.plot(levels, means.to_numpy(), "-o", color=PALETTE[i % len(PALETTE)], label=label)
    if by:
        ax.legend(title=by)
    return Result("Line Plot").figure(fig)


@analysis("graph.area", menu=MENU, title="Area Graph", params=[VARIABLES])
def area_graph(ws, columns):
    frame = ws.df[list(columns)].astype(float).fillna(0.0)
    fig, ax = _axes(tr("Area Graph"), tr("Index"), tr("Sum"), width=7.5)
    ax.stackplot(np.arange(1, len(frame) + 1), *[frame[c] for c in columns], labels=list(columns),
                 colors=PALETTE, alpha=0.85)
    ax.legend(loc="upper left")
    return Result("Area Graph").figure(fig)


def _xyz(ws, z, y, x):
    frame = ws.df[[z, y, x]].dropna().astype(float)
    if len(frame) < 3:
        raise ValueError("At least three complete rows are needed")
    return frame[x].to_numpy(), frame[y].to_numpy(), frame[z].to_numpy()


@analysis("graph.contour", menu=MENU, title="Contour Plot", params=[Z, Y, X])
def contour_plot(ws, z, y, x):
    xs, ys, zs = _xyz(ws, z, y, x)
    fig, ax = _axes(f"{tr('Contour Plot')}: {z} vs {y}, {x}", x, y)
    filled = ax.tricontourf(xs, ys, zs, levels=10, cmap="viridis")
    fig.colorbar(filled, ax=ax, label=z)
    return Result("Contour Plot").figure(fig)


@analysis("graph.surface", menu=MENU, title="3D Surface Plot", params=[Z, Y, X])
def surface_plot(ws, z, y, x):
    xs, ys, zs = _xyz(ws, z, y, x)
    fig = new_figure(6.5, 5.0)
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(xs, ys, zs, cmap="viridis", linewidth=0.2)
    ax.set_xlabel(x), ax.set_ylabel(y), ax.set_zlabel(z)
    ax.set_title(f"{tr('3D Surface Plot')}: {z} vs {y}, {x}")
    return Result("3D Surface Plot").figure(fig)


@analysis("graph.scatter3d", menu=MENU, title="3D Scatterplot", params=[Z, Y, X])
def scatter_3d(ws, z, y, x):
    xs, ys, zs = _xyz(ws, z, y, x)
    fig = new_figure(6.5, 5.0)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xs, ys, zs, color=BLUE)
    ax.set_xlabel(x), ax.set_ylabel(y), ax.set_zlabel(z)
    ax.set_title(f"{tr('3D Scatterplot')}: {z} vs {y}, {x}")
    return Result("3D Scatterplot").figure(fig)


@analysis("graph.heatmap", menu=MENU, title="Correlation Heatmap", params=[VARIABLES])
def correlation_heatmap(ws, columns):
    corr = ws.df[list(columns)].astype(float).corr()
    k = len(columns)
    fig, ax = _axes(tr("Correlation Heatmap"), width=max(5.0, 1.2 + 0.9 * k), height=max(4.0, 1.0 + 0.8 * k))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k), columns, rotation=45, ha="right")
    ax.set_yticks(range(k), columns)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    return Result("Correlation Heatmap").figure(fig)
