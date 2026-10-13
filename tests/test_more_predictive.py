"""MARS, test sets, model comparison, key predictors and the newer graphs."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.graphs import more
from openstat.stats import _mars, predictive
from openstat.stats import distributions as dists


def tables(res):
    return {payload[1]: payload[0] for kind, payload in res.blocks if kind == "table"}


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    n = 300
    x1, x2, noise = rng.uniform(0, 10, n), rng.uniform(0, 10, n), rng.normal(size=n)
    y = 5 + 2 * np.maximum(0, x1 - 4) - 3 * np.maximum(0, 6 - x2) + rng.normal(0, 0.3, n)
    return Worksheet("P", pd.DataFrame({
        "y": y, "x1": x1, "x2": x2, "noise": noise,
        "label": np.where((x1 > 5) ^ (x2 > 5), "odd", "even").astype(object),
        "site": np.array(["a", "b", "c"], dtype=object)[np.arange(n) % 3]}))


def test_mars_recovers_the_hinges(data):
    X = data.df[["x1", "x2", "noise"]].to_numpy()
    y = data.df["y"].to_numpy()
    model = _mars.MarsRegressor().fit(X, y)
    assert 1 - model.rss_ / ((y - y.mean()) ** 2).sum() > 0.99
    knots = {(v, round(t)) for term in model.terms_ for v, t, _ in term.factors}
    assert (0, 4) in knots and (1, 6) in knots                         # the knots of the true function
    assert all(v != 2 for term in model.terms_ for v, _, _ in term.factors)       # the noise is pruned away
    assert model.feature_importances_[2] == 0 and (model.feature_importances_[:2] > 0).all()
    assert model.predict(np.array([[9.0, 8.0, 0.0]]))[0] == pytest.approx(15.0, abs=0.3)
    assert model.path_[0]["terms"] >= len(model.terms_) and model.path_[-1]["terms"] == 1
    assert _mars.Term(((0, 4.0, 1), (1, 6.0, -1))).describe(["a", "b"]) == "max(0, a - 4) * max(0, 6 - b)"
    assert _mars.gcv(10.0, 100, 1, 2.0) == pytest.approx(10.0 / 100 / 0.99 ** 2) and np.isinf(_mars.gcv(1.0, 5, 5, 3.0))


def test_mars_interactions_and_binary_predictors():
    rng = np.random.default_rng(5)
    n = 400
    a, b, flag = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.integers(0, 2, n).astype(float)
    y = 3 * np.maximum(0, a) * np.maximum(0, b) + 2 * flag + rng.normal(0, 0.05, n)
    X = np.column_stack([a, b, flag])
    additive = _mars.MarsRegressor(max_degree=1).fit(X, y)
    crossed = _mars.MarsRegressor(max_degree=2).fit(X, y)
    assert crossed.rss_ < 0.2 * additive.rss_
    assert any(len(term.factors) == 2 for term in crossed.terms_)
    assert any(term.factors == ((2, 0.0, 1),) for term in crossed.terms_)       # the 0/1 column enters as itself
    constant = _mars.MarsRegressor().fit(X, np.full(n, 7.0))
    assert len(constant.terms_) == 1 and constant.predict(X[:3]) == pytest.approx([7.0] * 3)


def test_mars_analysis_and_test_set(data):
    res = predictive.mars_regression(data, "y", ["x1", "x2", "noise"], ["site"], folds=3, store=True, holdout=25)
    t = tables(res)
    summary = t["Model Summary"]
    assert list(summary.columns) == ["Training", "3-fold", "Test"]
    assert summary.loc["R-sq", "Test"] > 98 and summary.loc["R-sq", "3-fold"] > 98
    assert "1" in t["Basis Functions"].index and any("x1" in name for name in t["Basis Functions"].index)
    assert len(res.stored["PFITS_y"]) == 300 and "75" in res.blocks[0][1]
    assert predictive.holdout_split(300, 25, 1).sum() == 225
    assert predictive.holdout_split(300, 25, 1).tolist() == predictive.holdout_split(300, 25, 1).tolist()
    assert predictive.holdout_split(300, 0, 1).all()
    with pytest.raises(ValueError):
        predictive.holdout_split(12, 50, 1)

    res = predictive.cart_classification(data, "label", ["x1", "x2"], holdout=30)
    t = tables(res)
    assert t["Confusion Matrix (test set)"].to_numpy().sum() == 90
    assert t["Model Summary"].loc["Accuracy (%)", "Test"] > 85


def test_compare_models(data):
    frame, rows = predictive.predictor_matrix(data, "y", ["x1", "x2", "noise"], [])
    y, classes = predictive.response_values(data, "y", rows, False)
    table = predictive.compare_models(frame, y, False, folds=3, trees=40)
    assert set(table.index) == {"Linear regression", "CART", "Random forest", "Gradient boosting", "MARS"}
    assert table["R-sq"].is_monotonic_decreasing and table.index[0] == "MARS"     # the data are a MARS model
    assert table.loc["MARS", "R-sq"] > table.loc["Linear regression", "R-sq"]

    res = predictive.compare(data, "label", "Categorical", ["x1", "x2", "noise"], folds=3, trees=40)
    table = [p[0] for k, p in res.blocks if k == "table"][0]
    assert "MARS" not in table.index and table["Accuracy (%)"].is_monotonic_decreasing
    assert table.loc["Logistic regression", "Accuracy (%)"] < table.loc["Random forest", "Accuracy (%)"]   # XOR
    assert res.blocks[-2][1].endswith(table.index[0])
    with pytest.raises(ValueError):
        predictive.compare(data, "label", "Continuous", ["x1"])


def test_key_predictors(data):
    table = predictive.key_predictors(data, "y", False, ["x1", "x2", "noise"], ["site"], folds=3, trees=40)
    assert table["Predictors"].tolist() == [4, 3, 2, 1]
    assert set(table["Least important"].iloc[:2]) == {"noise", "site"}           # the two that carry no signal
    assert table["Score"].iloc[2] > 90 and table["Score"].iloc[3] < table["Score"].iloc[2]
    res = predictive.discover_key_predictors(data, "y", "Continuous", ["x1", "x2", "noise"], ["site"], 3, 40)
    assert res.blocks[-2][0] == "text" and "x1" in res.blocks[-2][1] and "x2" in res.blocks[-2][1]
    by_class = predictive.key_predictors(data, "label", True, ["x1", "x2", "noise"], [], folds=3, trees=40)
    assert by_class["Least important"].iloc[0] == "noise"


def test_binned_scatter_and_parallel_coordinates(data):
    res = more.binned_scatterplot(data, "y", "x1", 10)
    assert len(res.figures()) == 1 and "N = 300" in res.blocks[1][1]
    res = more.parallel_coordinates(data, ["y", "x1", "x2"], "site")
    lines = res.figures()[0].axes[0].lines
    assert len(lines) == 300 + 3 + 3                                    # rows, legend entries and the three axes
    ys = np.concatenate([line.get_ydata() for line in lines[:100]])
    assert ys.min() >= 0 and ys.max() <= 1
    raw = more.parallel_coordinates(data, ["y", "x1"], None, standardize=False)
    assert max(line.get_ydata().max() for line in raw.figures()[0].axes[0].lines[:300]) == pytest.approx(
        max(data.df["y"].max(), data.df["x1"].max()))
    with pytest.raises(ValueError):
        more.parallel_coordinates(data, ["y"])


def test_probability_distribution_plot():
    normal = dists.make(dists.NAMES[0], 0, 1)
    assert more.shaded_region(normal, "Right tail", 0.05)[0][0] == pytest.approx(1.6449, abs=1e-4)
    both = more.shaded_region(normal, "Both tails", 0.05)
    assert both[0][1] == pytest.approx(-1.96, abs=1e-3) and both[1][0] == pytest.approx(1.96, abs=1e-3)
    assert more.shaded_region(normal, "Middle", 0.95)[0] == pytest.approx((-1.96, 1.96), abs=1e-3)
    assert more.shaded_region(normal, "None", 0.05) == []
    res = more.distribution_plot(None, "Chi-square (degrees of freedom)", 4, 0, 0, "Right tail", 0.05)
    assert f"{sps.chi2.isf(0.05, 4):.5g}" in res.blocks[-1][1] and res.blocks[-1][1].endswith("= 0.05")
    table = [p[0] for k, p in res.blocks if k == "table"][0]
    assert table.iloc[0]["Mean"] == pytest.approx(4.0) and table.iloc[0]["StDev"] == pytest.approx(np.sqrt(8))
    discrete = more.distribution_plot(None, "Binomial (trials, event probability)", 10, 0.5, 0, "Left tail", 0.06)
    # the shaded bars are the whole values up to the quantile, so their probability is not exactly 0.06
    assert discrete.blocks[-1][1].endswith(f"= {sps.binom.cdf(2, 10, 0.5):.5g}")
    assert len(more.distribution_plot(None).blocks) == 2
    with pytest.raises(ValueError):
        more.distribution_plot(None, dists.NAMES[0], 0, -1)
