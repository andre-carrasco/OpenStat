"""Screening, split-plot, optimal and constrained mixture designs, and the plots and predictions built on DOE models."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from openstat.core.worksheet import Worksheet
from openstat.stats import doe


def tables(res):
    return [payload[0] for kind, payload in res.blocks if kind == "table"]


def test_conference_matrices_are_orthogonal():
    for order in doe.CONFERENCE_ORDERS:
        c = doe.conference_matrix(order)
        assert (np.diag(c) == 0).all() and set(np.unique(c)) == {-1.0, 0.0, 1.0}
        assert np.array_equal(c.T @ c, (order - 1) * np.eye(order))
    with pytest.raises(ValueError):
        doe.conference_matrix(16)


def test_definitive_screening_properties():
    for factors in (3, 6, 7, 10, 13):
        design = np.vstack([doe.definitive_screening(factors), np.zeros(factors)])
        assert design.shape[1] == factors and np.array_equal(design[:len(design) // 2], -design[len(design) // 2:-1])
        assert np.allclose(design.T @ design, (design ** 2).sum(axis=0)[0] * np.eye(factors))   # orthogonal main effects
        for a, b in combinations(range(factors), 2):                                           # clear of interactions
            assert np.allclose(design.T @ (design[:, a] * design[:, b]), 0)
        assert np.allclose(design.T @ design ** 2, 0)                                          # and of the squares
    assert len(doe.definitive_screening(6)) == 12 and len(doe.definitive_screening(7)) == 16
    frame = doe.create_definitive_screening(None, 6, 1, True, seed=3).worksheets[0][1]
    assert len(frame) == 13 and (frame[list("ABCDEF")] == 0).all(axis=1).sum() == 1
    with pytest.raises(ValueError):
        doe.definitive_screening(31)


def test_split_plot_keeps_hard_factors_constant_in_a_whole_plot():
    frame = doe.create_split_plot(None, hard=2, easy=2, replicates=2, randomize=True, seed=9).worksheets[0][1]
    assert len(frame) == 32 and frame["WP"].nunique() == 8
    assert (frame.groupby("WP")[["A", "B"]].nunique() == 1).all().all()
    assert (frame.groupby("WP")[["C", "D"]].apply(lambda g: len(g.drop_duplicates())) == 4).all()
    assert frame["RunOrder"].tolist() == list(range(1, 33)) and sorted(frame["StdOrder"]) == list(range(1, 33))
    assert frame["WP"].diff().fillna(0).ne(0).sum() == 7              # the runs of a whole plot stay together
    plain = doe.create_split_plot(None, 1, 2, 1, randomize=False).worksheets[0][1]
    assert plain["StdOrder"].tolist() == list(range(1, 9))


def test_d_optimal_selects_the_corners():
    grid = np.array([[a, b] for a in np.linspace(-1, 1, 5) for b in np.linspace(-1, 1, 5)])
    linear = np.column_stack([np.ones(len(grid)), grid])
    chosen, logdet = doe.d_optimal_rows(linear, 4, seed=1)
    assert sorted(map(tuple, grid[chosen])) == [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    assert logdet == pytest.approx(np.log(64.0))
    with pytest.raises(ValueError):
        doe.d_optimal_rows(linear, 2)

    ws = Worksheet("C", pd.DataFrame(grid * [10, 2] + [50, 7], columns=["T", "P"]))
    res = doe.select_optimal(ws, ["T", "P"], 9, doe.RSM_MODELS[0], seed=1)
    picked = res.worksheets[0][1]
    assert len(picked) == 9
    # the 3 x 3 factorial is the D-optimal nine-run design for a quadratic model on a square
    assert sorted(picked["T"].unique()) == [40.0, 50.0, 60.0] and sorted(picked["P"].unique()) == [5.0, 7.0, 9.0]
    assert tables(res)[0].iloc[0]["Relative to all candidates (%)"] > 100


def test_extreme_vertices():
    # Unconstrained bounds give the pure components
    assert doe.extreme_vertices(np.zeros(3), np.ones(3)).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    lower, upper = np.array([0.1, 0.2, 0.1]), np.array([0.6, 0.7, 0.5])
    points = doe.extreme_vertices(lower, upper)
    assert np.allclose(points.sum(axis=1), 1)
    assert ((points >= lower - 1e-9) & (points <= upper + 1e-9)).all()
    on_bound = np.isclose(points, lower) | np.isclose(points, upper)
    assert (on_bound.sum(axis=1) >= 2).all() and len(points) == 6
    frame = doe.create_extreme_vertices(None, "0.1, 0.2, 0.1", "0.6, 0.7, 0.5", True, 1, False).worksheets[0][1]
    assert len(frame) == 7 and frame[["A", "B", "C"]].iloc[-1].tolist() == pytest.approx(points.mean(axis=0).tolist())
    for low, high in [("0.5, 0.6", "0.9, 0.9"), ("0.1, 0.1", "0.2, 0.3"), ("0.3, 0.2", "0.2, 0.9"), ("0.1", "0.2")]:
        with pytest.raises(ValueError):
            doe.create_extreme_vertices(None, low, high)


def test_binary_response_factorial_recovers_the_model():
    design = doe.two_level_design(3)[0]
    design = np.tile(design, (2, 1))
    eta = 0.2 + 0.9 * design[:, 0] - 0.6 * design[:, 1] + 0.4 * design[:, 0] * design[:, 1]
    trials = np.full(len(design), 4000.0)
    events = np.random.default_rng(6).binomial(4000, 1 / (1 + np.exp(-eta))).astype(float)
    ws = Worksheet("B", pd.DataFrame({"e": events, "n": trials, "A": design[:, 0], "B": design[:, 1], "C": design[:, 2]}))
    coefs = tables(doe.analyze_binary_factorial(ws, "e", "n", ["A", "B", "C"]))[0]
    assert coefs.loc["A", "Coef"] == pytest.approx(0.9, abs=0.05)
    assert coefs.loc["A*B", "Coef"] == pytest.approx(0.4, abs=0.05)
    assert abs(coefs.loc["C", "Coef"]) < 0.05 and coefs.loc["A", "Odds Ratio"] == pytest.approx(np.exp(coefs.loc["A", "Coef"]))
    X = sm.add_constant(np.column_stack([design, design[:, 0] * design[:, 1], design[:, 0] * design[:, 2],
                                         design[:, 1] * design[:, 2]]))
    ref = sm.GLM(np.column_stack([events, trials - events]), X, family=sm.families.Binomial()).fit()
    assert coefs["Coef"].tolist() == pytest.approx(ref.params.tolist())
    ws.df.loc[0, "e"] = 5000.0
    with pytest.raises(ValueError):
        doe.analyze_binary_factorial(ws, "e", "n", ["A", "B", "C"])


@pytest.fixture
def surface_ws():
    rng = np.random.default_rng(2)
    grid = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], dtype=float)
    a, b, c = grid.T
    y1 = 50 + 4 * a - 3 * b + 2 * c - 5 * a ** 2 - 2 * b ** 2 + 1.5 * a * b + rng.normal(0, 0.05, len(grid))
    y2 = 20 - 2 * a + 4 * b + c + rng.normal(0, 0.05, len(grid))
    return Worksheet("R", pd.DataFrame({"y1": y1, "y2": y2, "A": a, "B": b, "C": c}))


def test_surface_grid_and_plots(surface_ws):
    frame = surface_ws.df
    fit = doe.fit_surface(frame, "y1", ["A", "B", "C"])[0]
    _, _, at_mean = doe.surface_grid(fit, frame, ["A", "B", "C"], doe.RSM_MODELS[0], "A", "B", size=3)
    _, _, at_one = doe.surface_grid(fit, frame, ["A", "B", "C"], doe.RSM_MODELS[0], "A", "B", {"C": 1.0}, size=3)
    assert at_mean[1, 1] == pytest.approx(50.0, abs=0.1)              # the centre, with C at its mean of 0
    assert (at_one - at_mean) == pytest.approx(np.full((3, 3), 2.0), abs=0.1)
    res = doe.surface_plots(surface_ws, "y1", ["A", "B", "C"], "A", "C", doe.RSM_MODELS[0], "B=0.5")
    assert len(res.figures()) == 1 and "B = 0.5" in res.blocks[0][1]
    assert doe.parse_hold("C=5; D = -2", ["C", "D"]) == {"C": 5.0, "D": -2.0}
    for x, y, hold in [("A", "A", ""), ("A", "y2", ""), ("A", "B", "Z=1"), ("A", "B", "C=x")]:
        with pytest.raises(ValueError):
            doe.surface_plots(surface_ws, "y1", ["A", "B", "C"], x, y, doe.RSM_MODELS[0], hold)


def test_overlaid_contour(surface_ws):
    res = doe.overlaid_contour(surface_ws, ["y1", "y2"], ["A", "B", "C"], "A", "B", "45, 18", "60, 21")
    share = float(res.blocks[1][1].rsplit(": ", 1)[1].rstrip("%"))
    assert 0 < share < 100 and len(res.figures()) == 1
    nothing = doe.overlaid_contour(surface_ws, ["y1", "y2"], ["A", "B", "C"], "A", "B", "500, 18", "600, 21")
    assert nothing.blocks[1][1].endswith(" 0.0%")
    for low, high in [("45", "60, 21"), ("45, 30", "60, 21"), ("a, b", "60, 21")]:
        with pytest.raises(ValueError):
            doe.overlaid_contour(surface_ws, ["y1", "y2"], ["A", "B", "C"], "A", "B", low, high)


def test_optimizer_with_goals_by_response(surface_ws):
    assert doe.parse_goals("y1=max; y2 = Target 21,5", ["y1", "y2"]) == {"y1": ("Maximize", None), "y2": ("Target", 21.5)}
    assert doe.parse_goals(" y2=MINIMIZE ", ["y1", "y2"]) == {"y2": ("Minimize", None)}
    for bad in ["y3=max", "y1", "y1=target abc"]:
        with pytest.raises(ValueError):
            doe.parse_goals(bad, ["y1", "y2"])
    frame = surface_ws.df
    both_max = doe.optimize_responses(frame, ["y1", "y2"], ["A", "B", "C"], "Maximize")
    mixed = doe.optimize_responses(frame, ["y1", "y2"], ["A", "B", "C"], "Maximize",
                                   goals={"y2": ("Minimize", None)})
    assert mixed["goals"] == {"y1": ("Maximize", None), "y2": ("Minimize", None)}
    assert mixed["predicted"]["y2"] < both_max["predicted"]["y2"]      # B moves down when y2 is to be small
    assert mixed["x"][1] < both_max["x"][1]
    table = tables(doe.response_optimizer(surface_ws, ["y1", "y2"], ["A", "B", "C"], "Maximize", 0.0,
                                          doe.RSM_MODELS[0], "y2=target 20"))[1]
    assert table.loc["y2", "Goal"] == "Target (20)" and table.loc["y2", "Predicted response"] == pytest.approx(20, abs=0.5)


def test_taguchi_prediction_is_additive():
    array = np.array([[1, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1]], dtype=float)
    y = 10 + 3 * (array[:, 0] == 2) + 5 * (array[:, 1] == 2) - 2 * (array[:, 2] == 2)
    ws = Worksheet("T", pd.DataFrame({"r1": y - 0.5, "r2": y + 0.5, "A": array[:, 0], "B": array[:, 1], "C": array[:, 2]}))
    table = tables(doe.predict_taguchi(ws, ["r1", "r2"], ["A", "B", "C"], doe.SN_RATIOS[0], "A=2, B=2, C=1; A=1,B=1,C=1"))[0]
    assert table["Mean"].tolist() == pytest.approx([18.0, 10.0])       # 18 is not one of the runs
    assert table.index[0] == "A = 2, B = 2, C = 1" and table["S/N ratio"].iloc[0] > table["S/N ratio"].iloc[1]
    with pytest.raises(ValueError):
        doe.predict_taguchi(ws, ["r1", "r2"], ["A", "B", "C"], doe.SN_RATIOS[0], "A=3, B=1, C=1")


def test_modify_design():
    base = doe.finish_design(doe.two_level_design(3, "1/2")[0], "ABC", randomize=False)
    ws = Worksheet("D", base)
    folded = doe.modify_design(ws, ["A", "B", "C"], doe.MODIFICATIONS[2], seed=4).worksheets[0][1]
    assert len(folded) == 8 and len(folded[["A", "B", "C"]].drop_duplicates()) == 8     # the full factorial
    assert sorted(folded["StdOrder"]) == list(range(1, 9)) and folded["RunOrder"].tolist() == list(range(1, 9))
    replicated = doe.modify_design(ws, ["A", "B", "C"], doe.MODIFICATIONS[1], seed=4).worksheets[0][1]
    assert len(replicated) == 8 and len(replicated[["A", "B", "C"]].drop_duplicates()) == 4
    shuffled = doe.modify_design(ws, ["A", "B", "C"], doe.MODIFICATIONS[0], seed=4).worksheets[0][1]
    assert sorted(shuffled["StdOrder"]) == [1, 2, 3, 4] and shuffled["StdOrder"].tolist() != [1, 2, 3, 4]
    back = doe.modify_design(Worksheet("S", shuffled), ["A", "B", "C"], doe.MODIFICATIONS[3]).worksheets[0][1]
    assert back[["A", "B", "C"]].to_numpy().tolist() == base[["A", "B", "C"]].to_numpy().tolist()
    uncoded = Worksheet("U", pd.DataFrame({"T": [100.0, 150.0, 100.0], "P": [1.0, 1.0, 3.0]}))
    extra = doe.modify_design(uncoded, ["T", "P"], doe.MODIFICATIONS[2], seed=1).worksheets[0][1]
    assert sorted(extra["T"]) == [100, 100, 100, 150, 150, 150] and sorted(extra["P"]) == [1, 1, 1, 3, 3, 3]
