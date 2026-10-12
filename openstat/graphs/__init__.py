from matplotlib.figure import Figure


def new_figure(width: float = 6.0, height: float = 4.0) -> Figure:
    """Figure created without pyplot, so it works in any thread and backend."""
    return Figure(figsize=(width, height), layout="constrained")


from openstat.graphs import basic, relation  # noqa: E402,F401
