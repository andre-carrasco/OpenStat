"""Regression and ANOVA, checked against statsmodels, SciPy and NIST certified values."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import anova, regression
from openstat.stats._design import build_design


def tables(res):
    return {payload[1]: payload[0] for kind, payload in res.blocks if kind == "table"}


@pytest.fixture
def ws():
    rng = np.random.default_rng(21)
    n = 60
    x1, x2 = rng.normal(10, 2, n), rng.normal(5, 1, n)
    a = np.array(["a1", "a2", "a3"], dtype=object)[rng.integers(0, 3, n)]
    b = np.array(["b1", "b2"], dtype=object)[rng.integers(0, 2, n)]
    effect = pd.Series(a).map({"a1": 0.0, "a2": 2.0, "a3": -1.0}).to_numpy()
    y = 4 + 1.5 * x1 - 2.0 * x2 + effect + np.where(b == "b2", 1.0, 0.0) + rng.normal(0, 1, n)
    return Worksheet("W", pd.DataFrame({"y": y, "x1": x1, "x2": x2, "noise": rng.normal(size=n), "a": a, "b": b,
                                        "bin": np.where(y > np.median(y), "hi", "lo").astype(object)}))


def test_longley_nist_certified():
    # NIST StRD Longley: certified coefficients, residual SD and R-squared
    data = sm.datasets.longley.load_pandas().data
    ws = Worksheet("Longley", data)
    predictors = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
    design = build_design(ws, ["TOTEMP"], predictors)
    _, fit = regression.fit_ols(ws, "TOTEMP", design)
    assert fit.params["Constant"] == pytest.approx(-3482258.63459582, rel=1e-7)
    assert fit.params["GNPDEFL"] == pytest.approx(15.0618722713733, rel=1e-7)
    summary = regression.model_summary(fit)
    assert summary["S"] == pytest.approx(304.854073561965, rel=1e-7)
    assert summary["R-sq"] == pytest.approx(99.5479004577296, rel=1e-9)


def test_fit_regression_tables(ws):
    res = regression.fit_regression(ws, "y", ["x1", "x2"], ["a"], store=True)
    t = tables(res)
    ref = smf.ols("y ~ x1 + x2 + C(a)", ws.df).fit()
    coef = t["Coefficients"]
    assert coef.loc["x1", "Coef"] == pytest.approx(ref.params["x1"])
    assert coef.loc["a_a2", "Coef"] == pytest.approx(ref.params["C(a)[T.a2]"])
    assert coef.loc["x1", "VIF"] > 1
    aov = t["Analysis of Variance"]
    ref3 = sm.stats.anova_lm(ref, typ=3)
    assert aov.loc["a", "Adj SS"] == pytest.approx(ref3.loc["C(a)", "sum_sq"])
    assert aov.loc["a", "DF"] == 2
    assert aov.loc["x1", "F-Value"] == pytest.approx(ref.tvalues["x1"] ** 2)
    assert aov.loc["Regression", "Adj SS"] + aov.loc["Error", "Adj SS"] == pytest.approx(aov.loc["Total", "Adj SS"])
    summary = t["Model Summary"].iloc[0]
    loo = [ws.df["y"][i] - smf.ols("y ~ x1 + x2 + C(a)", ws.df.drop(i)).fit().predict(ws.df.iloc[[i]]).iloc[0]
           for i in range(5)]
    hat = ref.get_influence().hat_matrix_diag
    assert (ref.resid[:5] / (1 - hat[:5])).to_numpy() == pytest.approx(loo)
    assert summary["PRESS"] == pytest.approx(((ref.resid / (1 - hat)) ** 2).sum())
    assert np.allclose(res.stored["FITS_y"], ref.fittedvalues)


def test_regression_rejects_bad_input(ws):
    with pytest.raises(ValueError):
        regression.fit_regression(ws, "y")
    with pytest.raises(ValueError):
        regression.fit_regression(ws, "y", ["y", "x1"])
    ws.add_column("x1copy", ws.df["x1"].tolist())
    with pytest.raises(ValueError, match="linearly dependent"):
        regression.fit_regression(ws, "y", ["x1", "x1copy"])


def test_glm_type3_matches_statsmodels(ws):
    res = anova.general_linear_model(ws, "y", ["a", "b"], ["x1"], interactions=True, plots=False)
    aov = tables(res)["Analysis of Variance"]
    ref = sm.stats.anova_lm(smf.ols("y ~ x1 * C(a, Sum) + x1 * C(b, Sum) + C(a, Sum) * C(b, Sum)", ws.df).fit(),
                            typ=3)
    assert aov.loc["a", "Adj SS"] == pytest.approx(ref.loc["C(a, Sum)", "sum_sq"])
    assert aov.loc["b", "Adj SS"] == pytest.approx(ref.loc["C(b, Sum)", "sum_sq"])
    assert aov.loc["a*b", "Adj SS"] == pytest.approx(ref.loc["C(a, Sum):C(b, Sum)", "sum_sq"])
    assert aov.loc["a*b", "P-Value"] == pytest.approx(ref.loc["C(a, Sum):C(b, Sum)", "PR(>F)"])


def test_fitted_line_and_nonlinear():
    x = np.linspace(1, 10, 25)
    ws = Worksheet(df=pd.DataFrame({"x": x, "q": 2 + 3 * x - 0.5 * x ** 2, "e": 2.5 * np.exp(0.3 * x),
                                    "m": 8 * x / (2 + x)}))
    res = regression.fitted_line_plot(ws, "q", "x", "Quadratic")
    assert tables(res)["Model Summary"].iloc[0]["R-sq"] == pytest.approx(100.0)
    out = regression.fit_nonlinear(x, ws.df["e"], regression.NONLINEAR_NAMES[0])
    assert out["params"] == pytest.approx([2.5, 0.3], rel=1e-5)
    out = regression.fit_nonlinear(x, ws.df["m"], "Michaelis-Menten: a * x / (b + x)")
    assert out["params"] == pytest.approx([8.0, 2.0], rel=1e-5)


def test_stepwise_and_best_subsets(ws):
    frame = ws.df[["x1", "x2", "noise"]]
    for method in regression.STEPWISE_METHODS:
        selected, _ = regression.stepwise_select(ws.df["y"].to_numpy(), frame, method, 0.05, 0.05)
        assert sorted(selected) == ["x1", "x2"]
    table = [p[0] for k, p in regression.best_subsets(ws, "y", ["x1", "x2", "noise"]).blocks if k == "table"][0]
    full = table[table["Vars"] == 3].iloc[0]
    assert full["Mallows Cp"] == pytest.approx(4.0)
    assert table[table["Vars"] == 2].iloc[0]["Predictors"] == "x1, x2"


def test_binary_logistic_matches_statsmodels(ws):
    res = regression.binary_logistic(ws, "bin", ["x1", "x2"], store=True)
    coef = tables(res)["Coefficients"]
    ref = sm.Logit((ws.df["bin"] == "lo").astype(float), sm.add_constant(ws.df[["x1", "x2"]])).fit(disp=0)
    assert coef.loc["x1", "Coef"] == pytest.approx(ref.params["x1"], rel=1e-5)
    assert coef.loc["x1", "Odds Ratio"] == pytest.approx(np.exp(ref.params["x1"]), rel=1e-5)
    deviance = tables(res)["Deviance Table"]
    assert deviance.loc["Regression", "Chi-Square"] == pytest.approx(ref.llr, rel=1e-6)
    assert np.nanmax(res.stored["EPRO_bin"]) <= 1


def test_poisson_ordinal_nominal_run():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    counts = rng.poisson(np.exp(0.5 + 0.8 * x)).astype(float)
    latent = 1.2 * x + rng.logistic(size=n)
    grade = np.where(latent < -1, "1", np.where(latent < 1, "2", "3")).astype(object)
    ws = Worksheet(df=pd.DataFrame({"x": x, "counts": counts, "grade": grade}))
    coef = tables(regression.poisson_regression(ws, "counts", ["x"]))["Coefficients"]
    assert coef.loc["x", "Coef"] == pytest.approx(0.8, abs=0.15)
    coef = tables(regression.ordinal_logistic(ws, "grade", ["x"]))["Coefficients"]
    assert coef.loc["x", "Coef"] == pytest.approx(1.2, abs=0.4)
    res = regression.nominal_logistic(ws, "grade", ["x"])
    assert tables(res)["Test of All Slopes Equal to Zero"].iloc[0]["P-Value"] < 0.001


def test_pls_one_component_per_predictor_equals_ols(ws):
    res = regression.partial_least_squares(ws, ["y"], ["x1", "x2"], components=2)
    coef = [p[0] for k, p in res.blocks if k == "table"][1]
    ref = smf.ols("y ~ x1 + x2", ws.df).fit()
    assert coef.loc["x1", "y"] == pytest.approx(ref.params["x1"], rel=1e-6)
    assert coef.loc["Constant", "y"] == pytest.approx(ref.params["Intercept"], rel=1e-6)


def test_deming_properties():
    rng = np.random.default_rng(8)
    t = rng.normal(0, 3, 200)
    x, y = t + rng.normal(0, 1, 200), 2 + 1.5 * t + rng.normal(0, 1, 200)
    forward, reverse = regression.deming(x, y, 1.0), regression.deming(y, x, 1.0)
    assert forward["slope"] == pytest.approx(1 / reverse["slope"])
    assert forward["slope"] == pytest.approx(1.5, abs=0.1)
    assert regression.deming(x, y, 1e9)["slope"] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-6)


def test_one_way_anova(ws):
    groups = anova.factor_groups(ws, "y", "a")
    t = anova.one_way_table(groups)
    ref = sps.f_oneway(*groups.values())
    assert t["f"] == pytest.approx(ref.statistic)
    assert t["p"] == pytest.approx(ref.pvalue)
    res = anova.one_way(ws, "y", "a", comparisons="Tukey", store=True)
    assert np.allclose(np.array(res.stored["FITS_y"]) + np.array(res.stored["RESI_y"]), ws.df["y"])
    means = ws.df.groupby("a")["y"].transform("mean")
    assert np.allclose(res.stored["FITS_y"], means)


def test_tukey_fisher_dunnett(ws):
    groups = anova.factor_groups(ws, "y", "a")
    tukey = anova.pairwise_comparisons(groups, "Tukey")
    ref = sps.tukey_hsd(*groups.values())
    assert tukey.loc["a2 − a1", "Adjusted P-Value"] == pytest.approx(ref.pvalue[1, 0], abs=1e-6)
    ci = ref.confidence_interval(0.95)
    assert tukey.loc["a2 − a1", "CI Lower"] == pytest.approx(ci.low[1, 0], abs=1e-5)
    fisher = anova.pairwise_comparisons(groups, "Fisher")
    assert (fisher["P-Value"] <= tukey["Adjusted P-Value"] + 1e-12).all()
    dunnett = anova.dunnett_comparisons(groups)
    assert list(dunnett.index) == ["a2 − a1", "a3 − a1"]
    assert dunnett.loc["a2 − a1", "Difference"] == pytest.approx(groups["a2"].mean() - groups["a1"].mean())


def test_welch_anova(ws):
    from statsmodels.stats.oneway import anova_oneway

    groups = anova.factor_groups(ws, "y", "a")
    out = anova.welch_anova(groups)
    ref = anova_oneway(list(groups.values()), use_var="unequal", welch_correction=True)
    assert out["f"] == pytest.approx(ref.statistic)
    assert out["p"] == pytest.approx(ref.pvalue)


def test_equal_variances(ws):
    t = tables(anova.equal_variances(ws, "y", "a"))["Tests"]
    groups = anova.factor_groups(ws, "y", "a")
    assert t.loc["Bartlett", "P-Value"] == pytest.approx(sps.bartlett(*groups.values()).pvalue)
    assert t.loc["Levene", "P-Value"] == pytest.approx(sps.levene(*groups.values()).pvalue)


def test_nested_anova_balanced():
    rng = np.random.default_rng(6)
    a_levels, b_per_a, reps = 6, 4, 5
    rows = []
    for a in range(a_levels):
        ea = rng.normal(0, 3)
        for b in range(b_per_a):
            eb = rng.normal(0, 2)
            rows += [(f"a{a}", f"a{a}b{b}", 50 + ea + eb + rng.normal(0, 1)) for _ in range(reps)]
    frame = pd.DataFrame(rows, columns=["A", "B", "y"])
    table, balanced = anova.nested_anova(frame["y"], [frame["A"].to_numpy(), frame["B"].to_numpy()])
    assert balanced
    assert table["DF"].tolist() == [5, 18, 96]
    assert table["SS"].sum() == pytest.approx(((frame["y"] - frame["y"].mean()) ** 2).sum())
    ms = table["MS"].to_numpy()
    assert table["F-Value"].iloc[0] == pytest.approx(ms[0] / ms[1])
    assert table["Variance component"].iloc[1] == pytest.approx((ms[1] - ms[2]) / reps)
    assert table["Variance component"].iloc[0] == pytest.approx((ms[0] - ms[1]) / (reps * b_per_a))
    assert table["Variance component"].iloc[2] == pytest.approx(1.0, abs=0.4)


def test_mixed_effects_variance_components():
    rng = np.random.default_rng(9)
    lots, per = 30, 8
    lot = np.repeat([f"L{i}" for i in range(lots)], per).astype(object)
    y = 20 + np.repeat(rng.normal(0, 2, lots), per) + rng.normal(0, 1, lots * per)
    ws = Worksheet(df=pd.DataFrame({"y": y, "lot": lot}))
    comps = tables(anova.mixed_effects(ws, "y", "lot"))["Variance Components"]
    assert comps.loc["Error", "Variance"] == pytest.approx(1.0, abs=0.3)
    assert comps.loc["lot", "Variance"] == pytest.approx(4.0, abs=2.5)
    assert comps["% of total"].iloc[:2].sum() == pytest.approx(100.0)


def test_manova_matches_statsmodels(ws):
    from statsmodels.multivariate.manova import MANOVA

    res = anova.manova(ws, ["y", "x1"], ["a"])
    stat = [p[0] for k, p in res.blocks if k == "table"][0]
    ref = MANOVA.from_formula("y + x1 ~ C(a)", ws.df).mv_test().results["C(a)"]["stat"]
    assert stat.loc["Wilks' lambda", "Statistic"] == pytest.approx(float(ref.loc["Wilks' lambda", "Value"]))
    assert stat.loc["Wilks' lambda", "P-Value"] == pytest.approx(float(ref.loc["Wilks' lambda", "Pr > F"]))
