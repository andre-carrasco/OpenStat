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
