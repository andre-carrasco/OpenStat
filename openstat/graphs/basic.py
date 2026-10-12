"""Basic graphs."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.graphs import new_figure
from openstat.i18n import tr


def histogram_figure(values, name: str, fit_normal: bool = False, bins="auto"):
    values = np.asarray(values, dtype=float)
    fig = new_figure()
    ax = fig.add_subplot()
    ax.hist(values, bins=bins, density=fit_normal, color="#5b9bd5", edgecolor="white")
    if fit_normal and len(values) > 1 and values.std(ddof=1) > 0:
        mean, sd = values.mean(), values.std(ddof=1)
        x = np.linspace(values.min() - sd, values.max() + sd, 200)
        ax.plot(x, sps.norm.pdf(x, mean, sd), color="#c00000")
        ax.set_ylabel(tr("Density"))
    else:
        ax.set_ylabel(tr("Frequency"))
    ax.set_xlabel(name)
    ax.set_title(f"{tr('Histogram')}: {name}")
    return fig


@analysis("graph.histogram", menu=("Graph",), title="Histogram",
          params=[Param("columns", "Variables", "columns"),
                  Param("fit_normal", "Fit normal curve", "bool", default=False)])
def histogram(ws, columns, fit_normal=False):
    res = Result("Histogram")
    for name in columns:
        res.figure(histogram_figure(ws.column(name), name, fit_normal))
    return res
