"""Graphs and grouped analyses follow the column's value order (Worksheet.levels)."""
import pandas as pd

from openstat.core.worksheet import Worksheet
from openstat.graphs import basic as graphs_basic
from openstat.stats import anova, basic, calc, data, tables

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
