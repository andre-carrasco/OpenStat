"""Basic statistics, tables, nonparametrics and equivalence, checked against SciPy and published values."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import basic, equivalence, nonparametric, tables
from openstat.stats._tests import GREATER, LESS, NOT_EQUAL


def table(res, i=0):
    return [payload[0] for kind, payload in res.blocks if kind == "table"][i]


@pytest.fixture
def ws():
    rng = np.random.default_rng(11)
    n = 30
    return Worksheet("W", pd.DataFrame({
        "A": rng.normal(10, 2, n), "B": rng.normal(11, 3, n), "C": rng.exponential(2, n),
        "G": np.array(["g1", "g2", "g3"], dtype=object)[np.arange(n) % 3],
        "Blk": np.array([f"b{i}" for i in range(10)], dtype=object)[np.arange(n) // 3],
    }))


def test_one_sample_t(ws):
    row = table(basic.one_sample_t(ws, ["A"], mu0=9.5)).loc["A"]
    ref = sps.ttest_1samp(ws.df["A"], 9.5)
    ci = ref.confidence_interval(0.95)
    assert row["T-Value"] == pytest.approx(ref.statistic)
    assert row["P-Value"] == pytest.approx(ref.pvalue)
    assert (row["CI Lower"], row["CI Upper"]) == pytest.approx((ci.low, ci.high))


@pytest.mark.parametrize("alternative,scipy_alt", [(LESS, "less"), (GREATER, "greater")])
def test_one_sample_t_one_sided(ws, alternative, scipy_alt):
    row = table(basic.one_sample_t(ws, ["A"], mu0=9.5, alternative=alternative)).loc["A"]
    ref = sps.ttest_1samp(ws.df["A"], 9.5, alternative=scipy_alt)
    ci = ref.confidence_interval(0.95)
    assert row["P-Value"] == pytest.approx(ref.pvalue)
    if alternative == LESS:
        assert np.isnan(row["CI Lower"]) and row["CI Upper"] == pytest.approx(ci.high)
    else:
        assert np.isnan(row["CI Upper"]) and row["CI Lower"] == pytest.approx(ci.low)


def test_one_sample_z(ws):
    row = table(basic.one_sample_z(ws, ["A"], sigma=2.0, mu0=10.0)).loc["A"]
    z = (ws.df["A"].mean() - 10) / (2 / np.sqrt(30))
    assert row["Z-Value"] == pytest.approx(z)
    assert row["P-Value"] == pytest.approx(2 * sps.norm.sf(abs(z)))


@pytest.mark.parametrize("pooled", [False, True])
def test_two_sample_t(ws, pooled):
    out = basic.two_sample_t_values(ws.df["A"], ws.df["B"], pooled=pooled)
    ref = sps.ttest_ind(ws.df["A"], ws.df["B"], equal_var=pooled)
    ci = ref.confidence_interval(0.95)
    assert out["T-Value"] == pytest.approx(ref.statistic)
    assert out["P-Value"] == pytest.approx(ref.pvalue)
    assert out["DF"] == pytest.approx(ref.df)
    assert (out["CI Lower"], out["CI Upper"]) == pytest.approx((ci.low, ci.high))


def test_paired_t(ws):
    row = table(basic.paired_t(ws, "A", "B"), 1).iloc[0]
    ref = sps.ttest_rel(ws.df["A"], ws.df["B"])
    assert row["T-Value"] == pytest.approx(ref.statistic)
    assert row["P-Value"] == pytest.approx(ref.pvalue)


def test_one_proportion_exact():
    row = table(basic.one_proportion(None, None, 12, 40, p0=0.5)).iloc[0]
    ref = sps.binomtest(12, 40, 0.5)
    ci = ref.proportion_ci(0.95, method="exact")
    assert row["P-Value"] == pytest.approx(ref.pvalue)
    assert (row["CI Lower"], row["CI Upper"]) == pytest.approx((ci.low, ci.high))


def test_one_proportion_from_column():
    ws = Worksheet(df=pd.DataFrame({"R": np.array(["no", "yes", "yes", "yes", None], dtype=object)}))
    row = table(basic.one_proportion(ws, "R")).iloc[0]
    assert (row["X"], row["N"]) == (3, 4)


def test_two_proportions():
    row = table(basic.two_proportions(None, 12, 40, 20, 45), 1).iloc[0]
    pooled = 32 / 85
    z = (12 / 40 - 20 / 45) / np.sqrt(pooled * (1 - pooled) * (1 / 40 + 1 / 45))
    assert row["Z-Value"] == pytest.approx(z)
    assert row["Fisher's exact P-Value"] == pytest.approx(sps.fisher_exact([[12, 28], [20, 25]]).pvalue)


def test_poisson_rate_exact_interval():
    # Garwood interval for 10 events in one unit: (4.795, 18.390)
    lo, hi = basic.poisson_rate_ci(10, 1)
    assert (lo, hi) == pytest.approx((4.7954, 18.3904), abs=1e-3)
    assert basic.poisson_exact_p(10, 10.0, NOT_EQUAL) == pytest.approx(1.0)
    assert basic.poisson_exact_p(3, 10.0, LESS) == pytest.approx(sps.poisson.cdf(3, 10))


def test_one_variance(ws):
    row = table(basic.one_variance(ws, ["A"], sigma0=2.0)).loc["A"]
    n, var = 30, ws.df["A"].var(ddof=1)
    assert row["Chi-Square"] == pytest.approx((n - 1) * var / 4)
    assert row["CI Lower (StDev)"] == pytest.approx(np.sqrt((n - 1) * var / sps.chi2.ppf(0.975, n - 1)))
    assert row["CI Upper (StDev)"] == pytest.approx(np.sqrt((n - 1) * var / sps.chi2.ppf(0.025, n - 1)))


def test_two_variances(ws):
    tests = table(basic.two_variances(ws, "A", "B"), 2)
    f = ws.df["A"].var(ddof=1) / ws.df["B"].var(ddof=1)
    assert tests.loc["F", "Statistic"] == pytest.approx(f)
    assert tests.loc["F", "P-Value"] == pytest.approx(2 * min(sps.f.cdf(f, 29, 29), sps.f.sf(f, 29, 29)))
    assert tests.loc["Levene", "P-Value"] == pytest.approx(sps.levene(ws.df["A"], ws.df["B"]).pvalue)


def test_correlation(ws):
    row = table(basic.correlation(ws, ["A", "B"]), 1).iloc[0]
    ref = sps.pearsonr(ws.df["A"], ws.df["B"])
    ci = ref.confidence_interval(0.95)
    assert row["Correlation"] == pytest.approx(ref.statistic)
    assert row["P-Value"] == pytest.approx(ref.pvalue)
    assert (row["CI Lower"], row["CI Upper"]) == pytest.approx((ci.low, ci.high))


def test_median_ci_order_statistics():
    # n = 10: the 95% interval is (x(2), x(9))
    assert basic.median_ci(np.arange(1, 11)) == (2.0, 9.0)


def test_grubbs_nist_example():
    # NIST/SEMATECH e-Handbook 1.3.5.17.1: G = 2.4687
    data = [199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57]
    out = basic.grubbs(data)
    assert out["G"] == pytest.approx(2.4687, abs=1e-4)
    assert out["Outlier"] == 245.57 and out["P-Value"] < 0.05


def test_normality_tests(ws):
    for test in basic.NORMALITY_TESTS:
        t = table(basic.normality_test(ws, ["A", "C"], test))
        assert len(t) == 2
    r, p = basic.ryan_joiner(ws.df["C"])
    assert r < 0.97 and p == "<0.010"
    r, p = basic.ryan_joiner(sps.norm.ppf((np.arange(1, 51) - 0.375) / 50.25))
    assert r == pytest.approx(1.0) and p == ">0.100"


def test_gof_poisson():
    counts = np.random.default_rng(5).poisson(2.5, 300)
    ws = Worksheet(df=pd.DataFrame({"N": counts.astype(float)}))
    res = basic.gof_poisson(ws, "N")
    assert table(res)["Observed"].sum() == 300
    assert table(res)["Poisson probability"].sum() == pytest.approx(1.0)
    assert table(res, 1).iloc[0]["P-Value"] > 0.01


def test_chi_square_association():
    ws = Worksheet(df=pd.DataFrame({"c1": [10.0, 20.0, 30.0], "c2": [20.0, 20.0, 15.0]}))
    res = tables.chi_square_association(ws, ["c1", "c2"])
    ref = sps.chi2_contingency([[10, 20], [20, 20], [30, 15]], correction=False)
    tests = table(res, 2)
    assert tests.loc["Pearson", "Chi-Square"] == pytest.approx(ref.statistic)
    assert tests.loc["Pearson", "P-Value"] == pytest.approx(ref.pvalue)
    lr = sps.chi2_contingency([[10, 20], [20, 20], [30, 15]], correction=False, lambda_="log-likelihood")
    assert tests.loc["Likelihood ratio", "Chi-Square"] == pytest.approx(lr.statistic)


def test_crosstab_fisher():
    ws = Worksheet(df=pd.DataFrame({"r": np.array(list("aaaabbbb"), dtype=object),
                                    "c": np.array(list("xxxyxyyy"), dtype=object)}))
    res = tables.cross_tabulation(ws, "r", "c")
    extra = table(res, 3).iloc[0]
    assert extra["Fisher's exact P-Value"] == pytest.approx(sps.fisher_exact([[3, 1], [1, 3]]).pvalue)


def test_chi_square_gof():
    ws = Worksheet(df=pd.DataFrame({"obs": [18.0, 22.0, 30.0, 10.0], "p": [0.25, 0.25, 0.25, 0.25]}))
    for props in (None, "p"):
        row = table(tables.chi_square_gof(ws, "obs", props), 1).iloc[0]
        ref = sps.chisquare([18, 22, 30, 10])
        assert row["Chi-Square"] == pytest.approx(ref.statistic)
        assert row["P-Value"] == pytest.approx(ref.pvalue)


def test_tally():
    ws = Worksheet(df=pd.DataFrame({"g": np.array(["a", "b", "a", None], dtype=object)}))
    t = table(tables.tally(ws, ["g"]))
    assert t["Count"].tolist() == [2, 1] and t["CumPct"].iloc[-1] == pytest.approx(100.0)


def test_sign_and_wilcoxon(ws):
    row = table(nonparametric.sign_test(ws, ["A"], median0=9.0)).loc["A"]
    above = int((ws.df["A"] > 9).sum())
    assert row["P-Value"] == pytest.approx(sps.binomtest(above, 30, 0.5).pvalue)
    row = table(nonparametric.wilcoxon(ws, ["A"], median0=9.0)).loc["A"]
    d = ws.df["A"].to_numpy() - 9
    assert row["Wilcoxon statistic"] == pytest.approx(sps.rankdata(np.abs(d))[d > 0].sum())
    assert 0 <= row["P-Value"] <= 1


def test_mann_whitney(ws):
    row = table(nonparametric.mann_whitney(ws, "A", "B"), 1).iloc[0]
    ref = sps.mannwhitneyu(ws.df["A"], ws.df["B"], method="asymptotic")
    assert row["P-Value"] == pytest.approx(ref.pvalue)
    assert row["W-Value"] == pytest.approx(ref.statistic + 30 * 31 / 2)


def test_kruskal_wallis(ws):
    tests = table(nonparametric.kruskal_wallis(ws, "A", "G"), 1)
    ref = sps.kruskal(*[g.to_numpy() for _, g in ws.df.groupby("G")["A"]])
    assert tests.loc["Adjusted for ties", "H-Value"] == pytest.approx(ref.statistic)
    assert tests.loc["Adjusted for ties", "P-Value"] == pytest.approx(ref.pvalue)


def test_mood_and_friedman(ws):
    row = table(nonparametric.mood_median(ws, "A", "G"), 1).iloc[0]
    assert row["DF"] == 2 and 0 <= row["P-Value"] <= 1
    row = table(nonparametric.friedman(ws, "A", "G", "Blk"), 1).iloc[0]
    wide = ws.df.pivot(index="Blk", columns="G", values="A")
    ref = sps.friedmanchisquare(*[wide[c] for c in wide.columns])
    assert row["Chi-Square"] == pytest.approx(ref.statistic)
    assert row["P-Value"] == pytest.approx(ref.pvalue)


def test_runs_test():
    ws = Worksheet(df=pd.DataFrame({"x": [1.0, 2, 1, 2, 1, 2, 1, 2, 1, 2]}))
    row = table(nonparametric.runs_test(ws, ["x"])).loc["x"]
    assert row["Observed runs"] == 10 and row["Expected runs"] == pytest.approx(6.0)
    assert row["P-Value"] < 0.05


def test_tost_matches_one_sided_t(ws):
    x = ws.df["A"].to_numpy() - 10
    out = equivalence.tost(float(x.mean()), float(x.std(ddof=1) / np.sqrt(30)), 29, -1.0, 1.0)
    assert out["p_lower"] == pytest.approx(sps.ttest_1samp(x, -1.0, alternative="greater").pvalue)
    assert out["p_upper"] == pytest.approx(sps.ttest_1samp(x, 1.0, alternative="less").pvalue)
    ci = sps.ttest_1samp(x, 0).confidence_interval(0.90)
    assert out["ci"] == pytest.approx((ci.low, ci.high))
    with pytest.raises(ValueError):
        equivalence.tost(0.0, 1.0, 5, 1.0, -1.0)


def test_crossover_effect_recovers_treatment_difference():
    rng = np.random.default_rng(2)
    n = 200
    subject = rng.normal(0, 3, n)
    seq = np.array(["AB", "BA"], dtype=object)[np.arange(n) % 2]
    a, b, period = 5.0, 3.0, 0.7
    p1 = subject + np.where(seq == "AB", a, b) + rng.normal(0, 0.5, n)
    p2 = subject + np.where(seq == "AB", b, a) + period + rng.normal(0, 0.5, n)
    effect, se, df = equivalence.crossover_effect(p1, p2, seq)
    assert df == n - 2
    assert effect == pytest.approx(a - b, abs=4 * se)
