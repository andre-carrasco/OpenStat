"""Prediction, stability, comparisons, analysis of means, balanced ANOVA and Cox regression,
checked against statsmodels, SciPy, lifelines and direct calculation."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import anova, regression, reliability


def tables(res):
    return [payload[0] for kind, payload in res.blocks if kind == "table"]


@pytest.fixture
def ws():
    rng = np.random.default_rng(21)
    n = 60
    x1, x2 = rng.normal(10, 2, n), rng.normal(5, 1, n)
    a = np.array(["a1", "a2", "a3"], dtype=object)[rng.integers(0, 3, n)]
    effect = pd.Series(a).map({"a1": 0.0, "a2": 2.0, "a3": -1.0}).to_numpy()
    y = 4 + 1.5 * x1 - 2.0 * x2 + effect + rng.normal(0, 1, n)
    return Worksheet("W", pd.DataFrame({"y": y, "x1": x1, "x2": x2, "a": a,
                                        "bin": np.where(y > np.median(y), "hi", "lo").astype(object)}))


def test_predict_matches_statsmodels(ws):
    res = regression.predict(ws, "y", ["x1", "x2"], ["a"], False, "x1=11, x2=4.5, a=a2; x1=30, x2=5, a=a1", 90.0)
    table = tables(res)[0]
    ref = smf.ols("y ~ x1 + x2 + C(a)", ws.df).fit().get_prediction(
        pd.DataFrame({"x1": [11.0, 30.0], "x2": [4.5, 5.0], "a": ["a2", "a1"]})).summary_frame(alpha=0.10)
    assert table["Fit"].tolist() == pytest.approx(ref["mean"].tolist())
    assert table["SE Fit"].tolist() == pytest.approx(ref["mean_se"].tolist())
    assert table["CI Lower"].tolist() == pytest.approx(ref["mean_ci_lower"].tolist())
    assert table["PI Upper"].tolist() == pytest.approx(ref["obs_ci_upper"].tolist())
    assert res.blocks[-1][0] == "text" and res.blocks[-1][1].endswith("x1")      # 30 is an extrapolation
    crossed = regression.predict(ws, "y", ["x1"], ["a"], True, "x1=11, a=a3")
    ref = smf.ols("y ~ x1 * C(a)", ws.df).fit().predict(pd.DataFrame({"x1": [11.0], "a": ["a3"]}))
    assert tables(crossed)[0]["Fit"].iloc[0] == pytest.approx(ref.iloc[0])
    assert crossed.blocks[-1][0] == "table"
    for bad in ["", "x1=1", "x1=1, a=zz", "x1=abc, a=a1", "x1=1, a=a1, q=2", "x1 1, a=a1"]:
        with pytest.raises(ValueError):
            regression.predict(ws, "y", ["x1"], ["a"], False, bad)


def test_binary_fitted_line(ws):
    res = regression.binary_fitted_line(ws, "bin", "x1")
    ref = smf.glm("event ~ x1", ws.df.assign(event=(ws.df["bin"] == "lo").astype(float)),
                  family=sm.families.Binomial()).fit()
    assert tables(res)[0].loc["x1", "Coef"] == pytest.approx(ref.params["x1"], rel=1e-6)
    assert len(res.figures()) == 1


def stability_data(slopes, intercepts, noise=0.15, seed=4):
    rng = np.random.default_rng(seed)
    times = np.tile([0.0, 3.0, 6.0, 9.0, 12.0, 18.0], len(slopes))
    batch = np.repeat([f"b{i + 1}" for i in range(len(slopes))], 6)
    y = np.repeat(intercepts, 6) + np.repeat(slopes, 6) * times + rng.normal(0, noise, len(times))
    return Worksheet("S", pd.DataFrame({"y": y, "t": times, "batch": batch.astype(object)}))


def test_stability_study_pooled_matches_direct_calculation():
    ws = stability_data([-0.20, -0.20, -0.20], [100.0, 100.0, 100.0])
    out = tables(regression.stability_study(ws, "y", "t", "batch", "95", ""))
    assert len(out[0]) == 2 and (out[0]["P-Value"] >= 0.25).all()          # both pooling tests pass
    assert len(out[-1]) == 1
    life = out[-1]["Shelf life"].iloc[0]
    fit = smf.ols("y ~ t", ws.df).fit()
    bound = fit.get_prediction(pd.DataFrame({"t": [life]})).summary_frame(alpha=0.10)["mean_ci_lower"].iloc[0]
    assert bound == pytest.approx(95.0, abs=1e-6)                           # the one-sided 95% bound meets the limit
    assert 20 < life < 25.5                                                 # the mean itself crosses near 25


def test_stability_study_separate_batches_and_limits():
    ws = stability_data([-0.10, -0.30, -0.20], [100.0, 101.0, 100.0])   # the means cross 95 at 50, 20 and 25
    out = tables(regression.stability_study(ws, "y", "t", "batch", "95", ""))
    assert len(out[0]) == 1 and out[0]["P-Value"].iloc[0] < 0.25            # the slopes differ
    lives = out[-1]["Shelf life"]
    assert lives.idxmin() == "b2" and lives["b1"] > lives["b3"] > lives["b2"]
    assert out[-1].loc["b2", "Slope"] == pytest.approx(-0.30, abs=0.03)
    two_sided = tables(regression.stability_study(ws, "y", "t", "batch", "95", "105"))[-1]["Shelf life"]
    assert (two_sided <= lives + 1e-9).all()                                # a two-sided interval is wider
    flat = stability_data([0.0], [100.0])
    assert np.isinf(tables(regression.stability_study(flat, "y", "t", None, "90", ""))[-1]["Shelf life"].iloc[0])
    late = tables(regression.stability_study(flat, "y", "t", None, "", "99"))[-1]
    assert late["Shelf life"].iloc[0] == 0.0                                # out of specification from the start
    with pytest.raises(ValueError):
        regression.stability_study(ws, "y", "t", "batch", "", "")


def test_games_howell_and_dunnett_critical(ws):
    groups = anova.factor_groups(ws, "y", "a")
    table = anova.games_howell(groups)
    ref = sps.tukey_hsd(*groups.values(), equal_var=False)
    assert table.loc["a2 − a1", "Adjusted P-Value"] == pytest.approx(ref.pvalue[0, 1], rel=1e-6)
    assert table.loc["a3 − a2", "Adjusted P-Value"] == pytest.approx(ref.pvalue[1, 2], rel=1e-6)
    interval = ref.confidence_interval(0.95)
    assert table.loc["a3 − a1", "CI Lower"] == pytest.approx(-interval.high[0, 2], rel=1e-6)
    assert table.loc["a3 − a1", "CI Upper"] == pytest.approx(-interval.low[0, 2], rel=1e-6)

    # one comparison is a plain one-sided t; several are checked against the multivariate t
    assert anova.one_sided_dunnett_critical([np.sqrt(0.5)], 12, 0.95) == pytest.approx(sps.t.ppf(0.95, 12), abs=1e-4)
    d = anova.one_sided_dunnett_critical([np.sqrt(0.5)] * 3, 20, 0.95)
    shape = np.full((3, 3), 0.5) + 0.5 * np.eye(3)
    assert sps.multivariate_t(shape=shape, df=20).cdf([d] * 3) == pytest.approx(0.95, abs=2e-3)
    lam = [np.sqrt(4 / 10), np.sqrt(8 / 14)]
    d = anova.one_sided_dunnett_critical(lam, 15, 0.9)
    shape = np.array([[1.0, lam[0] * lam[1]], [lam[0] * lam[1], 1.0]])
    assert sps.multivariate_t(shape=shape, df=15).cdf([d, d]) == pytest.approx(0.9, abs=2e-3)


def test_hsu_mcb_balanced(ws):
    rng = np.random.default_rng(8)
    groups = {name: rng.normal(mean, 1.0, 10) for name, mean in (("a", 10.0), ("b", 10.3), ("c", 13.0))}
    table = anova.hsu_mcb(groups)
    t = anova.one_way_table(groups)
    d = anova.one_sided_dunnett_critical([np.sqrt(0.5)] * 2, t["df_error"], 0.95)
    se = np.sqrt(2 * t["mse"] / 10)
    means = {k: v.mean() for k, v in groups.items()}
    for name in groups:
        diff = means[name] - max(v for k, v in means.items() if k != name)
        assert table.loc[name, "Difference from best"] == pytest.approx(diff)
        assert table.loc[name, "CI Upper"] == pytest.approx(max(0.0, diff + d * se))
    assert table.loc["c", "CI Lower"] == 0.0 and table.loc["a", "CI Upper"] == 0.0      # c is the best
    assert table.loc["a", "CI Lower"] == pytest.approx(means["a"] - means["c"] - d * se)
    smallest = anova.hsu_mcb(groups, largest=False)
    assert smallest.loc["c", "CI Lower"] == 0.0 and smallest.loc["c", "CI Upper"] > 0   # c is not the smallest
    assert (smallest["CI Lower"] <= smallest["Difference from best"] + 1e-12).all()
    for choice in anova.COMPARISONS[4:]:
        assert len(tables(anova.one_way(ws, "y", "a", choice, plots=False))) == 4


def test_analysis_of_means(ws):
    # with two levels the deviations from the grand mean are a t statistic
    assert anova.anom_critical([6, 6], 10, 0.05) == pytest.approx(sps.t.ppf(0.975, 10), rel=0.02)
    assert anova.anom_critical([4, 9], 11, 0.10) == pytest.approx(sps.t.ppf(0.95, 11), rel=0.02)
    h = anova.anom_critical([5] * 6, 24, 0.05)
    sidak = sps.t.ppf(1 - (1 - 0.95 ** (1 / 6)) / 2, 24)
    assert sps.t.ppf(0.975, 24) < h <= sidak * 1.005                 # between a single test and the Sidak bound
    table = tables(anova.analysis_of_means(ws, "y", "a"))[0]
    grand = ws.df["y"].mean()
    assert ((table["Lower limit"] < grand) & (grand < table["Upper limit"])).all()
    assert table.loc["a2", "Flag"] == "*"


def test_balanced_anova_fixed_and_mixed():
    rng = np.random.default_rng(15)
    a, b, _ = (g.ravel() for g in np.meshgrid(np.arange(3), np.arange(4), np.arange(2), indexing="ij"))
    y = (10 + np.array([0.0, 1.0, 2.5])[a] + rng.normal(0, 1.0, 4)[b] + rng.normal(0, 0.7, (3, 4))[a, b]
         + rng.normal(0, 0.5, len(a)))
    frame = pd.DataFrame({"y": y, "A": np.array(["p", "q", "r"], dtype=object)[a],
                          "B": np.array(["w", "x", "y", "z"], dtype=object)[b]})
    fixed = anova.balanced_anova_table(frame, "y", ["A", "B"])
    ref = sm.stats.anova_lm(smf.ols("y ~ C(A) * C(B)", frame).fit(), typ=2)
    assert fixed.loc["A", "SS"] == pytest.approx(ref.loc["C(A)", "sum_sq"])
    assert fixed.loc["A*B", "F-Value"] == pytest.approx(ref.loc["C(A):C(B)", "F"])
    assert fixed.loc["B", "P-Value"] == pytest.approx(ref.loc["C(B)", "PR(>F)"])
    assert fixed.loc["Error", "DF"] == 12 and fixed.loc["Total", "DF"] == 23

    mixed = anova.balanced_anova_table(frame, "y", ["A", "B"], ["B"])
    ms = mixed["MS"]
    assert mixed.loc["A", "F-Value"] == pytest.approx(ms["A"] / ms["A*B"]) and mixed.loc["A", "DF Den"] == 6
    assert mixed.loc["B", "F-Value"] == pytest.approx(ms["B"] / ms["A*B"])           # unrestricted model
    assert mixed.loc["A*B", "F-Value"] == pytest.approx(ms["A*B"] / ms["Error"])
    assert mixed.loc["A*B", "Variance component"] == pytest.approx(max(0.0, (ms["A*B"] - ms["Error"]) / 2))
    assert mixed.loc["B", "Variance component"] == pytest.approx(max(0.0, (ms["B"] - ms["A*B"]) / 6))
    assert np.isnan(mixed.loc["A", "Variance component"])

    single = frame.groupby(["A", "B"], as_index=False)["y"].mean()                    # one observation per cell
    table = anova.balanced_anova_table(single, "y", ["A", "B"])
    assert list(table.index) == ["A", "B", "Error", "Total"] and table.loc["Error", "DF"] == 6
    with pytest.raises(ValueError):
        anova.balanced_anova_table(frame.iloc[1:], "y", ["A", "B"])
    res = anova.balanced_anova(Worksheet("B", frame), "y", ["A", "B"], ["B"])
    assert "DF Den" in tables(res)[1].columns
    assert "DF Den" not in tables(anova.balanced_anova(Worksheet("B", frame), "y", ["A", "B"]))[1].columns


def test_balanced_anova_three_random_factors_uses_satterthwaite():
    rng = np.random.default_rng(3)
    idx = np.array(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), np.arange(2), indexing="ij")).reshape(4, -1)
    y = (rng.normal(0, 1, 3)[idx[0]] + rng.normal(0, 1, 3)[idx[1]] + rng.normal(0, 1, 3)[idx[2]]
         + rng.normal(0, 0.5, (3, 3))[idx[0], idx[1]] + rng.normal(0, 0.5, idx.shape[1]))
    frame = pd.DataFrame({"y": y, **{name: idx[i].astype(str).astype(object) for i, name in enumerate("ABC")}})
    table = anova.balanced_anova_table(frame, "y", ["A", "B", "C"], ["A", "B", "C"])
    ms = table["MS"]
    # E(MS_A) = error + ABC + AB + AC + A, so A is tested against MS_AB + MS_AC - MS_ABC
    denominator = ms["A*B"] + ms["A*C"] - ms["A*B*C"]
    if denominator > 0:
        assert table.loc["A", "F-Value"] == pytest.approx(ms["A"] / denominator)
        df = denominator ** 2 / (ms["A*B"] ** 2 / 4 + ms["A*C"] ** 2 / 4 + ms["A*B*C"] ** 2 / 8)
        assert table.loc["A", "DF Den"] == pytest.approx(df)
    else:
        assert np.isnan(table.loc["A", "F-Value"])
    assert table.loc["A*B", "F-Value"] == pytest.approx(ms["A*B"] / ms["A*B*C"])
    assert table.loc["A*B*C", "F-Value"] == pytest.approx(ms["A*B*C"] / ms["Error"])


def test_cox_regression_matches_lifelines():
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(31)
    n = 200
    x = rng.normal(size=n)
    g = np.array(["ctl", "trt"], dtype=object)[rng.integers(0, 2, n)]
    life = rng.exponential(1 / np.exp(0.7 * x - 0.5 * (g == "trt")))
    limit = rng.exponential(2.0, n)
    frame = pd.DataFrame({"t": np.round(np.minimum(life, limit), 2) + 0.01, "event": (life <= limit).astype(float),
                          "x": x, "g": g})
    out = tables(reliability.cox_regression(Worksheet("C", frame), "t", "event", "0", ["x"], ["g"]))
    ref = CoxPHFitter().fit(frame.assign(g=(frame["g"] == "trt").astype(float)), "t", "event")
    assert out[0].loc["x", "Coef"] == pytest.approx(ref.params_["x"], rel=1e-4)
    assert out[0].loc["g_trt", "Coef"] == pytest.approx(ref.params_["g"], rel=1e-4)
    assert out[0].loc["x", "SE Coef"] == pytest.approx(ref.standard_errors_["x"], rel=1e-3)
    assert out[0].loc["x", "Hazard ratio"] == pytest.approx(np.exp(ref.params_["x"]), rel=1e-4)
    assert out[1].iloc[0]["Chi-Square"] == pytest.approx(ref.log_likelihood_ratio_test().test_statistic, rel=1e-4)
    assert out[1].iloc[0]["Concordance"] == pytest.approx(ref.concordance_index_, abs=1e-6)
    breslow = tables(reliability.cox_regression(Worksheet("C", frame), "t", "event", "0", ["x"], ["g"], "Breslow"))
    assert breslow[0].loc["x", "Coef"] != out[0].loc["x", "Coef"]             # the tied times matter
