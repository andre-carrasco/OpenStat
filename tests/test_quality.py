"""Control charts and quality tools, checked against published constants and independent calculations."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from scipy.spatial.distance import mahalanobis

from openstat.core.worksheet import Worksheet
from openstat.stats import _spc, control, distfit, gage, quality


def test_unbiasing_constants_match_published_tables():
    # Montgomery, Introduction to Statistical Quality Control, Appendix VI
    assert _spc.d2(2) == pytest.approx(1.128, abs=5e-4)
    assert _spc.d2(5) == pytest.approx(2.326, abs=5e-4)
    assert _spc.d2(10) == pytest.approx(3.078, abs=5e-4)
    assert _spc.d3(2) == pytest.approx(0.853, abs=5e-4)
    assert _spc.d3(5) == pytest.approx(0.864, abs=5e-4)
    assert _spc.d3(10) == pytest.approx(0.797, abs=5e-4)
    assert _spc.c4(2) == pytest.approx(0.7979, abs=5e-5)
    assert _spc.c4(5) == pytest.approx(0.9400, abs=5e-5)
    assert _spc.c4(25) == pytest.approx(0.9896, abs=5e-5)


@pytest.fixture
def groups():
    rng = np.random.default_rng(3)
    return [rng.normal(100, 2, 5) for _ in range(25)]


def test_xbar_r_limits_use_a2_d3_d4(groups):
    sigma = _spc.sigma_rbar(groups)
    r_bar = np.mean([np.ptp(g) for g in groups])
    grand = np.mean([g.mean() for g in groups])
    xbar, rng = control.xbar_limits(groups, sigma), control.range_limits(groups, sigma)
    # A2 = 0.577, D3 = 0, D4 = 2.114 for subgroups of five
    assert xbar["ucl"][0] == pytest.approx(grand + 0.577 * r_bar, rel=1e-3)
    assert xbar["lcl"][0] == pytest.approx(grand - 0.577 * r_bar, rel=1e-3)
    assert rng["ucl"][0] == pytest.approx(2.114 * r_bar, rel=1e-3)
    assert rng["lcl"][0] == 0.0
    assert rng["center"][0] == pytest.approx(r_bar)


def test_xbar_s_limits_use_a3_b4(groups):
    sigma = _spc.sigma_sbar(groups)
    s_bar = np.mean([g.std(ddof=1) for g in groups])
    xbar, sd = control.xbar_limits(groups, sigma), control.sd_limits(groups, sigma)
    grand = np.mean([g.mean() for g in groups])
    assert xbar["ucl"][0] == pytest.approx(grand + 1.427 * s_bar, rel=1e-3)   # A3
    assert sd["ucl"][0] == pytest.approx(2.089 * s_bar, rel=1e-3)             # B4
    assert sd["lcl"][0] == 0.0                                                # B3


def test_imr_limits():
    x = np.array([10.0, 12, 11, 13, 9, 10, 12, 11])
    lim = control.imr_limits(x)
    mr_bar = np.abs(np.diff(x)).mean()
    assert lim["ucl"] == pytest.approx(x.mean() + 2.66 * mr_bar, rel=1e-3)
    assert lim["mr_ucl"] == pytest.approx(3.267 * mr_bar, rel=1e-3)
    assert np.isnan(lim["mr"][0])


def test_attribute_limits():
    counts, sizes = np.array([4.0, 6, 5, 9]), np.array([100.0, 100, 200, 100])
    p = control.p_limits(counts, sizes)
    assert p["center"] == pytest.approx(24 / 500)
    assert p["ucl"][2] == pytest.approx(0.048 + 3 * np.sqrt(0.048 * 0.952 / 200))
    u = control.u_limits(np.array([4.0, 9, 5]), np.ones(3))
    assert u["center"] == pytest.approx(6.0) and u["ucl"][0] == pytest.approx(6 + 3 * np.sqrt(6))
    with pytest.raises(ValueError):
        control.p_limits(np.array([5.0]), np.array([3.0]))


def test_special_cause_tests():
    assert _spc.special_cause_tests([0, 0, 3.5, 0], (1,)) == {1: [2]}
    assert _spc.special_cause_tests([0.5] * 9, (2,)) == {2: [8]}
    assert _spc.special_cause_tests([0.5] * 8, (2,)) == {}
    assert _spc.special_cause_tests([0, .1, .2, .3, .4, .5], (3,)) == {3: [5]}
    assert _spc.special_cause_tests([0, .1, .2, .3, .4], (3,)) == {}
    assert _spc.special_cause_tests([1, -1] * 7, (4,)) == {4: [13]}
    assert _spc.special_cause_tests([2.5, 0, 2.5], (5,)) == {5: [2]}
    assert _spc.special_cause_tests([1.5, 1.5, 0, 1.5, 1.5], (6,)) == {6: [4]}
    assert _spc.special_cause_tests([0.1] * 15, (7,)) == {7: [14]}
    assert _spc.special_cause_tests([1.5, -1.5] * 4, (8,)) == {8: [7]}


def test_ewma_and_cusum():
    x = np.full(200, 10.0)
    lim = control.ewma_limits(x, se=2.0, weight=0.2)
    assert lim["ucl"][-1] == pytest.approx(10 + 3 * 2 * np.sqrt(0.2 / 1.8))
    assert lim["ucl"][0] == pytest.approx(10 + 3 * 2 * 0.2)
    high, low = control.cusum_values(np.array([10.0, 12, 12, 8]), se=1.0, target=10.0, k=0.5)
    assert high.tolist() == [0.0, 1.5, 3.0, 0.5]
    assert low.tolist() == [0.0, 0.0, 0.0, -1.5]


def test_t_squared_is_mahalanobis():
    X = np.random.default_rng(1).normal(size=(30, 3))
    out = control.t_squared(X)
    inverse = np.linalg.inv(np.cov(X, rowvar=False))
    assert out["values"][4] == pytest.approx(mahalanobis(X[4], X.mean(axis=0), inverse) ** 2)
    assert out["ucl"] > out["center"] > 0


def test_box_cox_lambda():
    x = np.random.default_rng(2).lognormal(1.0, 0.6, 200)
    out = control.box_cox_lambda(x)
    assert out["lower"] <= 0 <= out["upper"] and out["rounded"] == 0.0
    assert out["lambda"] == pytest.approx(sps.boxcox(x)[1], abs=1e-4)
    with pytest.raises(ValueError):
        control.box_cox_lambda(np.array([1.0, -2.0, 3.0]))


def test_capability_normal(groups):
    out = quality.capability_normal(groups, 94.0, 106.0, 100.0)
    x = np.concatenate(groups)
    df = sum(len(g) - 1 for g in groups)
    within = np.sqrt(sum(((g - g.mean()) ** 2).sum() for g in groups) / df) / _spc.c4(df + 1)
    assert out["within"] == pytest.approx(within)
    assert out["cp"]["potential"] == pytest.approx(12 / (6 * within))
    assert out["cp"]["k"] == pytest.approx(min(x.mean() - 94, 106 - x.mean()) / (3 * within))
    assert out["pp"]["potential"] == pytest.approx(12 / (6 * x.std(ddof=1)))
    assert out["cpm"] == pytest.approx(12 / (6 * np.sqrt(((x - 100) ** 2).sum() / (len(x) - 1))))
    lo, hi = out["ppm"]["Expected overall"]
    assert lo == pytest.approx(1e6 * sps.norm.cdf(94, x.mean(), x.std(ddof=1)))
    assert hi == pytest.approx(1e6 * sps.norm.sf(106, x.mean(), x.std(ddof=1)))
    one_sided = quality.capability_normal(groups, None, 106.0)
    assert np.isnan(one_sided["cp"]["potential"]) and one_sided["cp"]["k"] == one_sided["cp"]["upper"]


def test_spec_limit_validation():
    assert quality.spec_limits("1,5", "") == (1.5, None)
    for bad in (("", ""), ("5", "1"), ("abc", "2")):
        with pytest.raises(ValueError):
            quality.spec_limits(*bad)


def test_capability_nonnormal_percentile_method():
    x = sps.weibull_min(2.0, scale=10).rvs(500, random_state=4)
    f = distfit.fit(x, "Weibull")
    assert f.params["Shape"] == pytest.approx(2.0, abs=0.25) and f.params["Scale"] == pytest.approx(10, abs=1)
    out = quality.capability_nonnormal(x, f.dist, 1.0, 25.0)
    q = f.dist.ppf([0.00135, 0.5, 0.99865])
    assert out["Pp"] == pytest.approx(24 / (q[2] - q[0]))
    assert out["Ppk"] == pytest.approx(min((q[1] - 1) / (q[1] - q[0]), (25 - q[1]) / (q[2] - q[1])))


def test_distribution_identification_ranks_true_family_high():
    x = sps.lognorm(0.8, scale=5).rvs(400, random_state=7)
    fits = distfit.fit_all(x)
    names = [f.name for f in fits]
    assert names.index("Lognormal") <= 2 and names.index("Normal") > names.index("Lognormal")
    normal = next(f for f in fits if f.name == "Normal")
    assert normal.ad == pytest.approx(sps.anderson(x, "norm", method="interpolate").statistic, rel=1e-6)
    assert all(f.name not in ("Lognormal", "Weibull") for f in distfit.fit_all(x - 100))


def test_johnson_transformation_improves_normality():
    x = np.random.default_rng(5).gamma(2.0, 3.0, 300)
    best = quality.johnson_transform(x)
    assert best["p"] > 0.05
    assert best["ad"] < quality.anderson_darling_normal(x)[0]


def test_run_chart_and_pareto():
    out = quality.run_chart_tests(np.arange(20.0))
    assert out["runs_median"] == 2 and out["p_clustering"] < 0.01
    assert out["runs_updown"] == 1 and out["p_trends"] < 0.01
    table = quality.pareto_table(list("aaaabbbcd"), combine_after=None)
    assert table["Count"].tolist() == [4, 3, 1, 1] and table["Cum %"].iloc[-1] == pytest.approx(100)
    weighted = quality.pareto_table(["x", "y"], [1, 9], None)
    assert list(weighted.index) == ["y", "x"]


def test_tolerance_factor_against_tables():
    # Two-sided 95%/95% normal tolerance factors: n=10 -> 3.379, n=30 -> 2.549
    assert quality.tolerance_factor(10, 0.95, 0.95) == pytest.approx(3.379, abs=0.02)
    assert quality.tolerance_factor(30, 0.95, 0.95) == pytest.approx(2.549, abs=0.01)


def test_sampling_plan_meets_both_risks():
    n, c = quality.sampling_plan(0.01, 0.05, 0.05, 0.10)
    assert sps.binom.cdf(c, n, 0.01) >= 0.95
    assert sps.binom.cdf(c, n, 0.05) <= 0.10
    assert sps.binom.cdf(c, n - 1, 0.05) > 0.10
    with pytest.raises(ValueError):
        quality.sampling_plan(0.05, 0.01)


def gage_frame(seed=0, parts=10, operators=3, reps=3, sd_part=3.0, sd_oper=1.0, sd_rep=0.5):
    rng = np.random.default_rng(seed)
    part_effect, oper_effect = rng.normal(0, sd_part, parts), rng.normal(0, sd_oper, operators)
    rows = [(f"P{p:02d}", f"O{o}", 50 + part_effect[p] + oper_effect[o] + rng.normal(0, sd_rep))
            for p in range(parts) for o in range(operators) for _ in range(reps)]
    return pd.DataFrame(rows, columns=["part", "operator", "y"])


def test_gage_crossed_anova_matches_statsmodels():
    frame = gage_frame()
    out = gage.crossed_anova(frame, alpha_interaction=1.0)   # keep the interaction
    ref = sm.stats.anova_lm(smf.ols("y ~ C(part) * C(operator)", frame).fit(), typ=2)
    table = out["anova"]
    assert table.loc["part", "SS"] == pytest.approx(ref.loc["C(part)", "sum_sq"])
    assert table.loc["operator", "SS"] == pytest.approx(ref.loc["C(operator)", "sum_sq"])
    assert table.loc["interaction", "SS"] == pytest.approx(ref.loc["C(part):C(operator)", "sum_sq"])
    assert table.loc["Repeatability", "SS"] == pytest.approx(ref.loc["Residual", "sum_sq"])
    ms = table["MS"]
    assert out["components"]["part"] == pytest.approx((ms["part"] - ms["interaction"]) / 9)
    assert out["components"]["repeatability"] == pytest.approx(ms["Repeatability"])


def test_gage_tables_and_pooling():
    frame = gage_frame(seed=1)
    out = gage.crossed_anova(frame)
    assert not out["with_interaction"]          # no interaction was simulated
    table, ndc = gage.gage_tables(out["components"], 6.0, tolerance=30.0)
    assert table.loc["Total Variation", "% Contribution"] == pytest.approx(100.0)
    grr, part = table.loc["Total Gage R&R", "VarComp"], table.loc["Part-To-Part", "VarComp"]
    assert table.loc["Total Variation", "VarComp"] == pytest.approx(grr + part)
    assert table.loc["Total Gage R&R", "% Tolerance"] == pytest.approx(100 * 6 * np.sqrt(grr) / 30)
    assert ndc == int(np.floor(1.41 * np.sqrt(part / grr)))
    assert out["components"]["repeatability"] == pytest.approx(0.25, abs=0.1)
    with pytest.raises(ValueError, match="balanced"):
        gage.crossed_anova(frame.iloc[:-1])


def test_type1_gage_study():
    x = np.array([10.01, 10.02, 9.99, 10.00, 10.03, 9.98, 10.01, 10.00])
    out = gage.type1_study(x, reference=10.0, tolerance=0.5)
    sd, bias = x.std(ddof=1), x.mean() - 10
    assert out["cg"] == pytest.approx(0.2 * 0.5 / (6 * sd))
    assert out["cgk"] == pytest.approx((0.1 * 0.5 - abs(bias)) / (3 * sd))
    assert out["p"] == pytest.approx(sps.ttest_1samp(x, 10.0).pvalue)


def test_fleiss_kappa_reference_example():
    # Fleiss (1971) worked example reproduced in standard references: kappa = 0.210
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert gage.fleiss_kappa(np.array(table)) == pytest.approx(0.210, abs=5e-4)


def test_attribute_agreement():
    rows = []
    truth = {"s1": "ok", "s2": "ng", "s3": "ok", "s4": "ng"}
    calls = {"A": {"s1": ["ok", "ok"], "s2": ["ng", "ng"], "s3": ["ok", "ng"], "s4": ["ng", "ng"]},
             "B": {"s1": ["ok", "ok"], "s2": ["ok", "ok"], "s3": ["ok", "ok"], "s4": ["ng", "ng"]}}
    for appraiser, samples in calls.items():
        for sample, ratings in samples.items():
            rows += [(r, sample, appraiser, truth[sample]) for r in ratings]
    ws = Worksheet(df=pd.DataFrame(rows, columns=["rating", "sample", "appraiser", "std"]).astype(object))
    res = gage.attribute_agreement(ws, "rating", "sample", "appraiser", "std")
    t = {p[1]: p[0] for k, p in res.blocks if k == "table"}
    assert t["Within Appraisers"]["Matched"].tolist() == [3, 4]
    assert t["Each Appraiser vs Standard"]["Matched"].tolist() == [3, 3]
    assert t["Between Appraisers"]["Matched"].iloc[0] == 2
    assert t["All Appraisers vs Standard"]["Matched"].iloc[0] == 2
