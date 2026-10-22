"""Graphs and grouped analyses follow the column's value order (Worksheet.levels)."""
import pandas as pd
import pytest

from openstat.core.worksheet import Worksheet
from openstat.graphs import basic as graphs_basic
from openstat.stats import anova, basic, calc, data, doe, multivariate, predictive, quality, regression, tables

REVERSED = ["D", "C", "B", "A"]


def tables_of(res):
    return [block[1][0] for block in res.blocks if block[0] == "table"]


def test_one_way_anova_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    res = anova.one_way(sample_ws, "Y", "Group")
    means = next(t for t in tables_of(res) if "Mean" in t.columns)
    assert list(means.index) == REVERSED


def test_boxplot_by_group_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    assert [label for label, _ in graphs_basic.groups(sample_ws, ["Y"], "Group")] == REVERSED


def test_cross_tabulation_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    res = tables.cross_tabulation(sample_ws, "Group", "Pass")
    assert list(tables_of(res)[0].index)[:4] == REVERSED


def test_unstack_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    res = data.unstack(sample_ws, "Y", "Group")
    assert list(res.stored) == [f"Y_{g}" for g in REVERSED]


def test_describe_and_indicator_follow_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    described = tables_of(basic.describe(sample_ws, ["Y"], by="Group"))[0]
    assert list(described.index) == [f"Y (Group = {g})" for g in REVERSED]
    assert list(calc.indicator(sample_ws, "Group").stored) == [f"Group_{g}" for g in REVERSED]


def test_numeric_factor_in_numeric_order():
    ws = Worksheet("N", pd.DataFrame({"y": [float(i) for i in range(30)], "f": [1, 2, 10] * 10}))
    means = next(t for t in tables_of(anova.one_way(ws, "y", "f")) if "Mean" in t.columns)
    assert list(means.index) == ["1", "2", "10"]


def test_bar_chart_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    assert list(graphs_basic._category_values(sample_ws, "Group", "Count", None).index) == REVERSED
    assert list(graphs_basic._category_values(sample_ws, "Group", "Mean", "Y").index) == REVERSED


def test_tally_and_describe_table_follow_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    assert list(tables_of(tables.tally(sample_ws, ["Group"]))[0].index) == REVERSED
    assert list(tables_of(tables.describe_table(sample_ws, "Group", "Y"))[0].index) == REVERSED


def test_factorial_plots_and_multi_vari_follow_value_order(sample_ws):
    ws = Worksheet("N", pd.DataFrame({"y": [float(i) for i in range(30)], "f": [1, 2, 10] * 10,
                                      "g": ["a", "b"] * 15}))
    means = next(t for t in tables_of(anova.one_way(ws, "y", "f")) if "Mean" in t.columns)
    assert list(means.index) == ["1", "2", "10"]
    fig = anova.main_effects_plot(ws, "y", ["f"]).blocks[0][1]
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["1", "2", "10"]
    sample_ws.value_orders["Group"] = REVERSED
    table = tables_of(quality.multi_vari(sample_ws, "Y", "Group"))[0]
    assert list(table.index) == REVERSED


def test_binary_logistic_event_follows_value_order(sample_ws):
    sample_ws.value_orders["Pass"] = ["yes", "no"]
    html = regression.binary_logistic(sample_ws, "Pass", continuous=["X"]).to_html()
    assert "Pass = no" in html          # event is now the last level, "no"
    fitted = regression.binary_fitted_line(sample_ws, "Pass", "X").to_html()
    assert "Pass = no" in fitted


def test_reference_level_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = ["C", "A", "B", "D"]
    res = regression.fit_regression(sample_ws, "Y", continuous=["X"], categorical=["Group"], plots=False)
    assert "Group = C" in res.to_html()
    coefs = next(t for t in tables_of(res) if "Coef" in t.columns)
    assert "Group_C" not in coefs.index and "Group_A" in coefs.index   # C is the reference: no coefficient


def test_default_reference_level_is_first_alphabetical(sample_ws):
    html = regression.fit_regression(sample_ws, "Y", categorical=["Group"], plots=False).to_html()
    assert "Group = A" in html


def test_ordinal_response_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = ["C", "A", "D", "B"]
    _, levels = regression._category_codes(sample_ws, "Group", list(range(sample_ws.nrows)))
    assert levels == ["C", "A", "D", "B"]


def test_proportion_event_follows_value_order(sample_ws):
    no = int((sample_ws.df["Pass"] == "no").sum())
    sample_ws.value_orders["Pass"] = ["yes", "no"]
    res = basic.one_proportion(sample_ws, column="Pass")
    assert tables_of(res)[0]["X"].iloc[0] == no
    assert "Pass = no" in res.to_html()


def test_text_two_level_factor_follows_value_order():
    ws = Worksheet("F", pd.DataFrame({"f": ["lo", "hi", "lo", "hi"]}))
    assert list(doe.code_two_level(ws.df["f"], "f", ws)) == [1.0, -1.0, 1.0, -1.0]   # alphabetical: hi is low
    ws.value_orders["f"] = ["lo", "hi"]
    assert list(doe.code_two_level(ws.df["f"], "f", ws)) == [-1.0, 1.0, -1.0, 1.0]


def test_multiple_correspondence_follows_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    res = multivariate.multiple_correspondence(sample_ws, ["Group", "Pass"])
    categories = next(t for t in tables_of(res) if t.index.name == "Category")
    assert list(categories.index)[:4] == [f"Group = {g}" for g in REVERSED]


def test_indicator_predictors_follow_value_order(sample_ws):
    sample_ws.value_orders["Group"] = REVERSED
    X, _ = predictive.predictor_matrix(sample_ws, "Y", [], ["Group"])
    assert list(X.columns) == [f"Group = {g}" for g in REVERSED]


def test_predict_codes_points_with_the_fit_levels(sample_ws):
    sample_ws.df.loc[0, "Y"] = float("nan")   # the first row (Group A) is dropped from the fit
    used = sample_ws.df.dropna(subset=["Y"])
    for order in (["C", "A", "B", "D"], "occurrence"):
        sample_ws.value_orders["Group"] = order
        res = regression.predict(sample_ws, "Y", categorical=["Group"], values="Group=A; Group=B; Group=C; Group=D")
        fits = tables_of(res)[0]["Fit"].to_numpy()
        assert fits == pytest.approx(used.groupby("Group")["Y"].mean().loc[list("ABCD")].to_numpy())


@pytest.mark.parametrize("g, order", [([2, 2, 10, 10, 30, 30], None),
                                      (["a", "a", "b", "b", "c", "c"], ["c", "b", "a"])])
def test_one_way_anova_stores_fits_by_row(g, order):
    ws = Worksheet("W", pd.DataFrame({"y": [1.0, 2, 10, 11, 20, 21], "g": g}))
    if order:
        ws.value_orders["g"] = order
    res = anova.one_way(ws, "y", "g", store=True)
    assert res.stored["FITS_y"] == [1.5, 1.5, 10.5, 10.5, 20.5, 20.5]
    assert res.stored["RESI_y"] == [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
