import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from openstat.core.expression import ExpressionError, evaluate
from openstat.core.worksheet import Worksheet
from openstat.graphs.basic import box_stats, stem_and_leaf
from openstat.stats import calc, data
from openstat.stats import distributions as dists
from openstat.stats._util import anderson_darling_normal


@pytest.fixture
def ws():
    return Worksheet("W", pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, np.nan],
                                        "G": np.array(["x", "y", "x", "y"], dtype=object)}))


def test_expression_arithmetic(ws):
    assert evaluate(ws, "A * 2 + 1").tolist() == [3.0, 5.0, 7.0, 9.0]
    assert evaluate(ws, "A ^ 2").tolist() == [1.0, 4.0, 9.0, 16.0]
    assert evaluate(ws, "C1 + c1").tolist() == [2.0, 4.0, 6.0, 8.0]
    assert evaluate(ws, "mean(A)") == 2.5
    assert evaluate(ws, "sqrt(A)")[3] == 2.0
    assert evaluate(ws, "A > 2 and G == 'x'").tolist() == [False, False, True, False]
    assert evaluate(ws, "where(A > 2, 1, 0)").tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize("expr", ["__import__('os')", "A.real", "open('x')", "[1,2]", "lambda: 1", ""])
def test_expression_rejects_unsafe(ws, expr):
    with pytest.raises(ExpressionError):
        evaluate(ws, expr)


def test_calculator_stores(ws):
    res = calc.calculator(ws, "Total", "A + B")
    assert res.stored["Total"][:3] == [11.0, 22.0, 33.0]
    assert np.isnan(res.stored["Total"][3])


def test_column_statistics(ws):
    table = calc.column_statistics(ws, ["A", "B"]).blocks[0][1][0]
    assert table.loc["A", "Sum"] == 10.0
    assert table.loc["B", "N"] == 3.0 and table.loc["B", "N*"] == 1.0
    assert table.loc["A", "Sum of squares"] == 30.0


def test_patterned():
    res = calc.patterned(None, 1, 3, 1, repeat_each=2, repeat_all=2)
    assert res.stored["Sequence"] == [1, 1, 2, 2, 3, 3] * 2
    with pytest.raises(ValueError):
        calc.patterned(None, 1, 3, -1)


def test_random_seeded():
    a = calc.random_data(None, 50, dists.NAMES[0], 10, 2, 0, seed=7).stored["Random"]
    b = calc.random_data(None, 50, dists.NAMES[0], 10, 2, 0, seed=7).stored["Random"]
    assert a == b and len(a) == 50


def test_probability_functions(ws):
    normal = dists.NAMES[0]
    cdf = calc.probability(ws, calc.FUNCTIONS[1], normal, 0, 1, 0, None, 1.96).blocks[1][1][0]
    assert cdf.iloc[0, 1] == pytest.approx(0.975, abs=1e-4)
    inv = calc.probability(ws, calc.FUNCTIONS[2], normal, 0, 1, 0, None, 0.975).blocks[1][1][0]
    assert inv.iloc[0, 1] == pytest.approx(1.95996, abs=1e-4)
    binom = "Binomial (trials, event probability)"
    pmf = calc.probability(ws, calc.FUNCTIONS[0], binom, 10, 0.5, 0, None, 5).blocks[1][1][0]
    assert pmf.iloc[0, 1] == pytest.approx(sps.binom.pmf(5, 10, 0.5))


def test_distribution_parameterisations():
    assert dists.make("Lognormal (location, scale)", 1.0, 0.5).median() == pytest.approx(np.exp(1.0))
    assert dists.make("Weibull (shape, scale)", 2.0, 3.0).cdf(3.0) == pytest.approx(1 - np.exp(-1))
    assert dists.make("Uniform (lower, upper)", 2.0, 6.0).mean() == pytest.approx(4.0)
    assert dists.make("Triangular (lower, mode, upper)", 0.0, 1.0, 4.0).cdf(1.0) == pytest.approx(0.25)
    assert dists.make("Integer (minimum, maximum)", 1, 6).pmf(6) == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        dists.make(dists.NAMES[0], 0.0, -1.0)


def test_sort_subset(ws):
    df = data.sort(ws, ["A"], descending=True).worksheets[0][1]
    assert df["A"].tolist() == [4.0, 3.0, 2.0, 1.0]
    df = data.subset(ws, "A >= 3").worksheets[0][1]
    assert df["A"].tolist() == [3.0, 4.0]


def test_stack_unstack(ws):
    res = data.stack(ws, ["A", "B"])
    assert res.stored["Stacked"] == [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0]
    assert res.stored["Subscripts"] == ["A"] * 4 + ["B"] * 3
    res = data.unstack(ws, "A", "G")
    assert res.stored == {"A_x": [1.0, 3.0], "A_y": [2.0, 4.0]}


def test_standardize_rank_recode(ws):
    z = data.standardize(ws, ["A"]).stored["Std_A"]
    assert np.mean(z) == pytest.approx(0.0) and np.std(z, ddof=1) == pytest.approx(1.0)
    assert data.rank(ws, "A").stored["Rank_A"] == [1.0, 2.0, 3.0, 4.0]
    assert data.recode(ws, "G", "x=1; y=2").stored["Coded_G"] == [1.0, 2.0, 1.0, 2.0]
    assert data.recode(ws, "A", "1:2=low; 3:4=high").stored["Coded_A"] == ["low", "low", "high", "high"]


def test_merge_split_copy(ws):
    other = Worksheet("O", pd.DataFrame({"G": np.array(["x", "z"], dtype=object), "A": [100.0, 300.0]}))
    df = data.merge(ws, other, ["G"], data.JOINS[1]).worksheets[0][1]
    assert df.columns.tolist() == ["A", "B", "G", "A_2"]
    assert df["A"].tolist() == [1.0, 3.0] and df["A_2"].tolist() == [100.0, 100.0]
    assert len(data.merge(ws, other, ["G"], data.JOINS[0]).worksheets[0][1]) == 5
    side = data.merge(ws, other).worksheets[0][1]
    assert side.columns.tolist() == ["A", "B", "G", "G_2", "A_2"] and len(side) == 4
    with pytest.raises(ValueError):
        data.merge(ws, other, ["B"])

    parts = dict(data.split(ws, "G").worksheets)
    assert parts["W (G = x)"]["A"].tolist() == [1.0, 3.0]
    assert data.copy_columns(ws, ["A"], "B > 10").worksheets[0][1]["A"].tolist() == [2.0, 3.0]


def test_concatenate_delete_rows(ws):
    assert data.concatenate(ws, ["G", "A"], "-").stored["Concat"] == ["x-1", "y-2", "x-3", "y-4"]
    assert data.delete_rows(ws, "1, 3:9").worksheets[0][1]["A"].tolist() == [2.0]
    assert data.delete_rows(ws, "2;3").worksheets[0][1]["A"].tolist() == [1.0, 4.0]
    for bad in ["", "0", "3:2", "a"]:
        with pytest.raises(ValueError):
            data.delete_rows(ws, bad)


def test_dates_change_type_and_extract():
    ws = Worksheet("D", pd.DataFrame({"When": [pd.Timestamp("2024-03-15 13:45:10"), pd.NaT, pd.Timestamp("2023-12-31")],
                                      "Text": np.array(["2024-02-29", "2021-01-01", None], dtype=object),
                                      "N": [1.0, 2.5, np.nan]}))
    assert ws.is_datetime("When") and not ws.is_numeric("When")
    assert [ws.get_text(r, 0) for r in range(3)] == ["2024-03-15 13:45:10", "", "2023-12-31"]
    assert data.extract_date(ws, "When", "Quarter").stored["When_Quarter"][0] == 1.0
    assert data.extract_date(ws, "When", "Day of the week (1 = Monday)").stored["When_Dayoftheweek"][2] == 7.0
    assert data.extract_date(ws, "When", "Minute").stored["When_Minute"][0] == 45.0
    assert data.extract_date(ws, "Text", "Day of the year").stored["Text_Dayoftheyear"][:2] == [60.0, 1.0]
    with pytest.raises(ValueError):
        data.extract_date(ws, "N")

    serial = data.change_type(ws, "When", "Numeric").stored["When_num"]
    assert serial[2] == 45291.0 and serial[1] is None      # spreadsheet day number of 2023-12-31
    assert serial[0] == pytest.approx(45366 + (13 * 3600 + 45 * 60 + 10) / 86400)
    assert data.change_type(ws, "N", "Date/time").stored["N_date"][0] == pd.Timestamp("1899-12-31")
    assert data.change_type(ws, "Text", "Date/time").stored["Text_date"][0] == pd.Timestamp("2024-02-29")
    assert data.change_type(ws, "N", "Text").stored["N_text"] == ["1", "2.5", None]
    assert data.change_type(ws, "Text", "Numeric").stored["Text_num"][0] == pytest.approx(45351.0)


def test_indicator_mesh(ws):
    res = calc.indicator(ws, "G")
    assert res.stored == {"G_x": [1.0, 0.0, 1.0, 0.0], "G_y": [0.0, 1.0, 0.0, 1.0]}
    res = calc.mesh(None, 0, 1, 2, 0, 2, 3, "X + 10 * Y")
    assert res.stored["X"] == [0.0, 1.0] * 3 and res.stored["Y"] == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    assert res.stored["Z"] == [0.0, 1.0, 10.0, 11.0, 20.0, 21.0]


def test_matrices():
    ws = Worksheet("M", pd.DataFrame({"a": [2.0, 1.0], "b": [1.0, 3.0], "c": [1.0, 0.0], "d": [0.0, 1.0]}))
    ops = calc.MATRIX_OPS
    assert calc.matrix(ws, ["a", "b"], ops[0]).worksheets[0][1].to_numpy().tolist() == [[5.0, 5.0], [5.0, 10.0]]
    assert calc.matrix(ws, ["a", "b"], ops[1], ["c", "d"]).worksheets[0][1].to_numpy().tolist() == [[2, 1], [1, 3]]
    inverse = calc.matrix(ws, ["a", "b"], ops[2]).worksheets[0][1].to_numpy()
    assert inverse @ np.array([[2.0, 1.0], [1.0, 3.0]]) == pytest.approx(np.eye(2))
    eigen = calc.matrix(ws, ["a", "b"], ops[3])
    assert eigen.blocks[1][1][0]["Eigenvalue"].tolist() == pytest.approx([(5 + 5 ** 0.5) / 2, (5 - 5 ** 0.5) / 2])
    summary = calc.matrix(ws, ["a", "b"], ops[4]).blocks[1][1][0]
    assert summary.iloc[0].tolist() == pytest.approx([5.0, 5.0, 2])
    with pytest.raises(ValueError):
        calc.matrix(ws, ["a", "b", "c"], ops[2])
    with pytest.raises(ValueError):
        calc.matrix(Worksheet("S", pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})), ["a", "b"], ops[2])


def test_anderson_darling_matches_scipy():
    x = np.random.default_rng(3).normal(size=80)
    ad, p = anderson_darling_normal(x)
    assert ad == pytest.approx(sps.anderson(x, "norm", method="interpolate").statistic, rel=1e-6)
    assert 0.0 <= p <= 1.0
    _, p_skew = anderson_darling_normal(np.random.default_rng(3).exponential(size=80))
    assert p_skew < 0.01


def test_box_stats_and_stem():
    stats = box_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], "v")
    assert stats["fliers"].tolist() == [100.0]
    assert stats["whishi"] == 9.0
    text = stem_and_leaf([10, 11, 12, 23, 25, 31])
    assert "1 | 012" in text and "2 | 35" in text and "3 | 1" in text
