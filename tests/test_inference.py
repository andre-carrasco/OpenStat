"""Basic statistics, tables, nonparametrics and equivalence, checked against SciPy and published values."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import basic, equivalence, nonparametric, resampling, tables
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


def test_dixon_matches_published_critical_values():
    # Critical values of r10 from Dixon (1951) and Rorabacher (1991), with their upper-tail probabilities
    for n, q, tail in [(3, 0.941, 0.05), (3, 0.970, 0.025), (5, 0.642, 0.05), (5, 0.821, 0.005),
                       (7, 0.507, 0.05), (10, 0.466, 0.025)]:
        assert basic.dixon_tail(q, n) == pytest.approx(tail, abs=1.5e-3)
    assert basic.dixon_tail(0.0, 6) == pytest.approx(1.0, abs=1e-6)
    # the usual textbook example: Q = 0.455 is below the 95% critical value 0.466
    out = basic.dixon([0.189, 0.167, 0.187, 0.183, 0.186, 0.182, 0.181, 0.184, 0.181, 0.177])
    assert out["Q"] == pytest.approx(0.4545, abs=1e-4) and out["Outlier"] == 0.167
    assert 0.05 < out["P-Value"] < 0.10
    assert basic.dixon([1.0, 2.0, 3.0, 4.0, 40.0])["P-Value"] < 0.01
    with pytest.raises(ValueError):
        basic.dixon([1.0, 1.0, 1.0])


def test_outlier_test_choice_and_store_describe(ws):
    ws.df.loc[0, "A"] = 60.0
    for test in basic.OUTLIER_TESTS:
        res = basic.outlier_test(ws, ["A"], 0.05, test)
        assert table(res).loc["A", "Outlier"] == 60.0 and table(res).loc["A", "P-Value"] < 0.05
    frame = basic.store_describe(ws, ["A", "B"], "G").worksheets[0][1]
    assert frame.shape[0] == 6 and frame["G"].tolist() == ["g1", "g2", "g3"] * 2
    row = frame[(frame["Variable"] == "B") & (frame["G"] == "g2")].iloc[0]
    values = ws.df.loc[ws.df["G"] == "g2", "B"]
    assert row["Mean"] == pytest.approx(values.mean()) and row["Variance"] == pytest.approx(values.var(ddof=1))
    assert len(basic.store_describe(ws, ["A"]).worksheets[0][1]) == 1


def test_bootstrap_and_randomization(ws):
    a, b = ws.df["A"].to_numpy(), ws.df["B"].to_numpy()
    res = resampling.bootstrap_one_sample(ws, ["A"], "Mean", 4000, 95.0, seed=1)
    row = table(res).loc["A"]
    assert row["Mean"] == pytest.approx(a.mean())
    assert row["Bootstrap StDev"] == pytest.approx(a.std(ddof=0) / np.sqrt(len(a)), rel=0.08)
    assert row["CI Lower"] < a.mean() < row["CI Upper"]
    again = table(resampling.bootstrap_one_sample(ws, ["A"], "Mean", 4000, 95.0, seed=1)).loc["A"]
    assert again["CI Lower"] == row["CI Lower"]
    median = table(resampling.bootstrap_one_sample(ws, ["A"], "Median", 500, 90.0, seed=2)).loc["A"]
    assert median["Median"] == np.median(a)

    diff = table(resampling.bootstrap_two_sample(ws, "A", "B", 4000, 95.0, seed=3), 1).iloc[0]
    assert diff["Difference"] == pytest.approx(a.mean() - b.mean())
    assert diff["Bootstrap StDev"] == pytest.approx(np.sqrt(a.var() / len(a) + b.var() / len(b)), rel=0.08)

    one = table(resampling.randomization_one_sample(ws, ["A"], a.mean() - 0.3, NOT_EQUAL, 5000, seed=4)).loc["A"]
    assert one["P-Value"] == pytest.approx(sps.ttest_1samp(a, a.mean() - 0.3).pvalue, abs=0.05)
    assert table(resampling.randomization_one_sample(ws, ["A"], 5.0, GREATER, 500, seed=4)).loc["A", "P-Value"] == 0.0
    assert table(resampling.randomization_one_sample(ws, ["A"], 5.0, LESS, 500, seed=4)).loc["A", "P-Value"] == 1.0

    two = table(resampling.randomization_two_sample(ws, "A", "B", NOT_EQUAL, 5000, seed=5), 1).iloc[0]
    ref = sps.permutation_test((a, b), lambda x, y: x.mean() - y.mean(), n_resamples=5000, random_state=5)
    assert two["P-Value"] == pytest.approx(ref.pvalue, abs=0.03)
    assert resampling.permuted_differences(a, b, 50, np.random.default_rng(0)).shape == (50,)


def test_ordinal_measures_and_kappa():
    counts = np.array([[12.0, 5.0, 2.0], [6.0, 9.0, 7.0], [1.0, 4.0, 14.0]])
    r, c = np.nonzero(counts)
    rows = np.repeat(r, counts[r, c].astype(int))
    cols = np.repeat(c, counts[r, c].astype(int))
    out = tables.ordinal_measures(counts)
    assert out["Kendall's tau-b"] == pytest.approx(sps.kendalltau(rows, cols).statistic)
    assert out["Somers' D (columns dependent)"] == pytest.approx(sps.somersd(rows, cols).statistic)
    assert out["Somers' D (rows dependent)"] == pytest.approx(sps.somersd(cols, rows).statistic)
    conc, disc = out["Concordant pairs"], out["Discordant pairs"]
    assert out["Goodman-Kruskal gamma"] == pytest.approx((conc - disc) / (conc + disc)) and conc > disc

    from sklearn.metrics import cohen_kappa_score
    ws = Worksheet(df=pd.DataFrame({"r": rows.astype(float), "c": cols.astype(float)}))
    res = tables.cross_tabulation(ws, "r", "c", measures=True)
    assert table(res, 4).iloc[0]["Kendall's tau-b"] == pytest.approx(out["Kendall's tau-b"])
    assert table(res, 5).iloc[0]["Kappa"] == pytest.approx(cohen_kappa_score(rows, cols))
    assert tables.ordered(["10", "9", "2"]) == ["2", "9", "10"] and tables.ordered(["b", "a"]) == ["a", "b"]


def test_mcnemar():
    from statsmodels.stats.contingency_tables import mcnemar

    before = ["no"] * 30 + ["yes"] * 30
    after = ["no"] * 18 + ["yes"] * 12 + ["no"] * 3 + ["yes"] * 27
    ws = Worksheet(df=pd.DataFrame({"before": np.array(before, dtype=object), "after": np.array(after, dtype=object)}))
    res = tables.mcnemar(ws, "before", "after")
    row = table(res, 1).iloc[0]
    assert row["Discordant pairs"] == 15 and row["Difference"] == pytest.approx((12 - 3) / 60)
    assert row["Exact P-Value"] == pytest.approx(mcnemar([[18, 12], [3, 27]], exact=True).pvalue)
    assert row["CI Lower"] < row["Difference"] < row["CI Upper"]
    with pytest.raises(ValueError):
        tables.mcnemar(Worksheet(df=pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 1.0]})), "a", "b")


def test_cochran_mantel_haenszel():
    layers = {"s1": [[10, 5], [4, 12]], "s2": [[8, 9], [3, 14]], "s3": [[15, 6], [7, 10]]}
    records = [(("t", "c")[i], ("y", "n")[j], name)
               for name, t in layers.items() for i in range(2) for j in range(2) for _ in range(t[i][j])]
    ws = Worksheet(df=pd.DataFrame(np.array(records, dtype=object), columns=["trt", "resp", "site"]))
    res = tables.cochran_mantel_haenszel(ws, "trt", "resp", "site")
    # categories sort as c < t and n < y, so each layer is seen with rows and columns reversed
    t = np.array([np.array(v)[::-1, ::-1] for v in layers.values()], dtype=float)
    n = t.sum(axis=(1, 2))
    expected = t[:, 0].sum(axis=1) * t[:, :, 0].sum(axis=1) / n
    variance = (t[:, 0].sum(axis=1) * t[:, 1].sum(axis=1) * t[:, :, 0].sum(axis=1) * t[:, :, 1].sum(axis=1)
                / (n ** 2 * (n - 1)))
    chi2 = (abs((t[:, 0, 0] - expected).sum()) - 0.5) ** 2 / variance.sum()
    row = table(res, 1).iloc[0]
    assert row["CMH Chi-Square"] == pytest.approx(chi2)
    assert row["P-Value"] == pytest.approx(sps.chi2.sf(chi2, 1))
    assert row["Common odds ratio"] == pytest.approx((t[:, 0, 0] * t[:, 1, 1] / n).sum() / (t[:, 0, 1] * t[:, 1, 0] / n).sum())
    assert table(res).loc["s1", "Odds ratio"] == pytest.approx(10 * 12 / (5 * 4))
    assert 0 <= table(res, 2).iloc[0]["P-Value"] <= 1


def test_table_of_descriptive_statistics(ws):
    one = table(tables.describe_table(ws, "G", "A"))
    assert one.loc["g2", "Mean"] == pytest.approx(ws.df.loc[ws.df["G"] == "g2", "A"].mean())
    two = table(tables.describe_table(ws, "G", "A", "Blk", "N"))
    assert two.shape == (3, 10) and two.loc["g1", "b0"] == 1


def test_pairwise_statistics(ws):
    a, b = ws.df["A"].to_numpy(), ws.df["B"].to_numpy()
    res = nonparametric.pairwise_averages(ws, "A")
    assert len(res.stored["Averages_A"]) == 30 * 31 // 2
    assert table(res).iloc[0, 1] == pytest.approx(np.median(nonparametric.walsh_averages(a)))
    res = nonparametric.pairwise_differences(ws, "A", "B")
    assert len(res.stored["Differences_A_B"]) == 900
    assert table(res).iloc[0, 1] == pytest.approx(np.median(np.subtract.outer(a, b)))
    res = nonparametric.pairwise_slopes(ws, "B", "A")
    assert table(res).iloc[0, 1] == pytest.approx(sps.theilslopes(b, a).slope)
    with pytest.raises(ValueError):
        nonparametric.pairwise_slopes(Worksheet(df=pd.DataFrame({"y": [1.0, 2.0], "x": [3.0, 3.0]})), "y", "x")


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
