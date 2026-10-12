from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def new_figure(width: float = 6.0, height: float = 4.0) -> Figure:
    """Figure created without pyplot, so it works in any thread and backend."""
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)  # a raster canvas, for code that needs a renderer before saving
    return fig


from openstat.graphs import basic, relation  # noqa: E402,F401
