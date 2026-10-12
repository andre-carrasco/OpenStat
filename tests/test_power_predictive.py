"""Power calculations checked against statsmodels and exact definitions; predictive models on known structure."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower, TTestPower

from openstat.core.worksheet import Worksheet
from openstat.stats import power, predictive
from openstat.stats._tests import GREATER, LESS, NOT_EQUAL


def table(res, i=0):
    return [payload[0] for kind, payload in res.blocks if kind == "table"][i]


def test_t_power_matches_statsmodels():
    assert power.t_power(0.5 * np.sqrt(30), 29, 0.05, NOT_EQUAL) == pytest.approx(
        TTestPower().power(0.5, 30, 0.05, alternative="two-sided"))
    assert power.t_power(0.5 * np.sqrt(30), 29, 0.05, GREATER) == pytest.approx(
        TTestPower().power(0.5, 30, 0.05, alternative="larger"))
    assert power.t_power(-0.5 * np.sqrt(30), 29, 0.05, LESS) == pytest.approx(
        TTestPower().power(-0.5, 30, 0.05, alternative="smaller"))
    assert power.t_power(0.8 / np.sqrt(2 / 20), 38, 0.05, NOT_EQUAL) == pytest.approx(
        TTestIndPower().power(0.8, 20, 0.05))
    assert power.z_power(0.4 * np.sqrt(25), 0.05, NOT_EQUAL) == pytest.approx(
        NormalIndPower().power(0.4, 25, 0.05, ratio=0))


def test_poisson_rate_power():
    za, zb = sps.norm.isf(0.05), sps.norm.isf(0.1)
    # one sample, one-sided: n = ((za sqrt(r0) + zb sqrt(r1)) / (r1 - r0))^2
    n1 = int(np.ceil(((za * np.sqrt(2.0) + zb * np.sqrt(3.0)) / (3.0 - 2.0)) ** 2))
    row = table(power.power_one_rate(None, 0, 3.0, 0.9, 2.0, 0.05, GREATER)).iloc[0]
    assert row["Sample size"] == n1 and row["Power"] >= 0.9
    assert power.one_rate_power(n1 - 1, 2.0, 3.0, 0.05, GREATER) < 0.9
    assert power.one_rate_power(20, 2.0, 2.0, 0.05, NOT_EQUAL) == pytest.approx(0.05)
    assert power.one_rate_power(20, 2.0, 1.2, 0.05, LESS) > power.one_rate_power(20, 2.0, 1.6, 0.05, LESS)
    # two samples, one-sided: n = (za + zb)^2 (r1 + r2) / (r1 - r2)^2 per group
    n2 = int(np.ceil((za + zb) ** 2 * (3.0 + 2.0) / (3.0 - 2.0) ** 2))
    row = table(power.power_two_rates(None, 0, 3.0, 0.9, 2.0, 0.05, GREATER)).iloc[0]
    assert row["Sample size per group"] == n2
    solved = table(power.power_two_rates(None, n2, 0.0, 0.9, 2.0, 0.05, GREATER)).iloc[0]
    assert 2.9 < solved["Comparison rate"] <= 3.0
    with pytest.raises(ValueError):
        power.power_one_rate(None, 0, 2.0, 0.9, 2.0)


def test_design_power_variants():
    assert table(power.power_plackett_burman(None, 12, 11, 2, 2.0, 0.0)).iloc[0]["Power"] == pytest.approx(
        power.factorial_power(2, 12, 11, 2.0, 1.0, 0.05))
    with pytest.raises(ValueError):
        power.power_plackett_burman(None, 10, 5, 2, 2.0, 0.0)
    # a single factor with main effects only is the one-way ANOVA
    assert power.general_factorial_power(6, [3], 1, 1.2, 1.0, 0.05) == [pytest.approx(power.anova_power(6, 3, 1.2, 1.0, 0.05))]
    by_factor = power.general_factorial_power(3, [2, 4], 2, 1.0, 1.0, 0.05)
    assert by_factor[0] > by_factor[1]                        # fewer levels: more runs behind each mean
    assert power.general_factorial_power(1, [2, 4], 2, 1.0, 1.0, 0.05) == [0.0, 0.0]   # no error df
    res = power.power_general_factorial(None, "2, 4", 2, 0, 1.0, 0.9)
    row = table(res).iloc[0]
    assert row["Power"] >= 0.9 and table(res, 1)["Power"].min() == pytest.approx(row["Power"])
    assert min(power.general_factorial_power(int(row["Replicates"]) - 1, [2, 4], 2, 1.0, 1.0, 0.05)) < 0.9
    assert power.parse_levels("2;3 4") == [2, 3, 4]
    for bad in ["", "1, 3", "a"]:
        with pytest.raises(ValueError):
            power.parse_levels(bad)


def test_equivalence_power_crossover():
    n, sd, alpha = 12, 1.0, 0.05
    crit = sps.t.isf(alpha, 2 * n - 2)
    se = sd / np.sqrt(n)
    expected = sps.nct.cdf(-crit, 2 * n - 2, (0.1 - 1) / se) - sps.nct.cdf(crit, 2 * n - 2, (0.1 + 1) / se)
    assert power.equivalence_power(n, 0.1, -1, 1, sd, alpha, False, crossover=True) == pytest.approx(expected)
    row = table(power.power_equivalence(None, power.EQUIVALENCE_DESIGNS[2], 0, 0.1, 0.9, -1, 1, sd, alpha)).iloc[0]
    assert row["Power"] >= 0.9 and "Subjects per sequence" in row.index


def test_sample_sizes_match_textbook_values():
    # difference of one standard deviation, power 0.9, alpha 0.05, two-sided
    assert table(power.power_two_sample_t(None, 0, 1.0, 0.9))["Sample size per group"].iloc[0] == 23
    assert table(power.power_one_sample_t(None, 0, 1.0, 0.9))["Sample size"].iloc[0] == 13
    assert table(power.power_one_sample_z(None, 0, 1.0, 0.9))["Sample size"].iloc[0] == 11
    assert table(power.power_paired_t(None, 0, 0.5, 0.8))["Number of pairs"].iloc[0] == 34


def test_solve_for_each_unknown_is_consistent():
    row = table(power.power_two_sample_t(None, 0, 0.75, 0.85)).iloc[0]
    n = int(row["Sample size per group"])
    assert row["Power"] >= 0.85
    assert table(power.power_two_sample_t(None, n - 1, 0.75, 0.0)).iloc[0]["Power"] < 0.85
    solved = table(power.power_two_sample_t(None, n, 0.0, 0.85)).iloc[0]
    assert solved["Difference"] <= 0.75 and solved["Power"] == pytest.approx(0.85, abs=1e-6)
    for bad in ((0, 0.0, 0.9), (10, 1.0, 0.9), (0, 0.0, 0.0)):
        with pytest.raises(ValueError):
            power.power_two_sample_t(None, *bad)


def test_anova_power_matches_statsmodels():
    k, n, d, sd = 4, 12, 1.5, 1.2
    effect = np.sqrt(d ** 2 / (2 * k * sd ** 2))
    assert power.anova_power(n, k, d, sd, 0.05) == pytest.approx(
        FTestAnovaPower().power(effect, n * k, 0.05, k_groups=k))
    assert power.anova_power(n, k, 0.0, sd, 0.05) == pytest.approx(0.05)


def test_factorial_power_is_a_noncentral_f():
    value = power.factorial_power(2, 8, 7, 2.0, 1.0, 0.05)
    assert value == pytest.approx(sps.ncf.sf(sps.f.isf(0.05, 1, 8), 1, 8, 16 * 4 / 4))
    assert power.factorial_power(1, 8, 7, 2.0, 1.0, 0.05) == 0.0     # saturated: no error degrees of freedom
    with pytest.raises(ValueError):
        power.power_factorial(None, runs=8, terms=8, difference=2.0)


def test_proportion_power():
    assert power.one_proportion_power(100, 0.5, 0.5, 0.05, NOT_EQUAL) == pytest.approx(0.05)
    assert power.two_proportion_power(100, 0.4, 0.4, 0.05, NOT_EQUAL) == pytest.approx(0.05)
    assert power.one_proportion_power(100, 0.5, 0.65, 0.05, GREATER) > power.one_proportion_power(
        100, 0.5, 0.65, 0.05, NOT_EQUAL)
    row = table(power.power_two_proportions(None, 0, 0.7, 0.9, p2=0.5)).iloc[0]
    n = int(row["Sample size per group"])
    assert power.two_proportion_power(n, 0.7, 0.5, 0.05, NOT_EQUAL) >= 0.9
    assert power.two_proportion_power(n - 1, 0.7, 0.5, 0.05, NOT_EQUAL) < 0.9
    assert 120 <= n <= 130        # the usual formula gives about 124 per group
    back = table(power.power_two_proportions(None, n, 0.0, 0.9, p2=0.5)).iloc[0]
    assert 0.5 < back["Comparison p"] <= 0.7


def test_variance_power():
    assert power.variance_power(20, 1.0, 0.05, NOT_EQUAL, False) == pytest.approx(0.05)
    assert power.variance_power(20, 1.0, 0.05, GREATER, True) == pytest.approx(0.05)
    crit = sps.chi2.isf(0.05, 19)
    assert power.variance_power(20, 1.5, 0.05, GREATER, False) == pytest.approx(sps.chi2.sf(crit / 2.25, 19))
    row = table(power.power_two_variances(None, 0, 1.5, 0.8)).iloc[0]
    assert row["Power"] >= 0.8 and row["Sample size per group"] > 20


def test_equivalence_power_behaviour():
    small = power.equivalence_power(10, 0.0, -1, 1, 1.0, 0.05, False)
    large = power.equivalence_power(40, 0.0, -1, 1, 1.0, 0.05, False)
    assert 0 <= small < large <= 1
    assert power.equivalence_power(40, 0.9, -1, 1, 1.0, 0.05, False) < large
    assert power.equivalence_power(40, 0.0, -1, 1, 1.0, 0.05, True) < large
    with pytest.raises(ValueError):
        power.power_equivalence(None, n=0, difference=2.0, power=0.9)


def test_estimation_and_tolerance_sample_sizes():
    n = power.mean_estimation_size(0.5, 1.0, 0.95)
    half = lambda m: sps.t.ppf(0.975, m - 1) / np.sqrt(m)   # noqa: E731
    assert half(n) <= 0.5 < half(n - 1) and n == 18
    assert power.proportion_estimation_size(0.05, 0.5, 0.95) == 385
    # classic result: 93 observations for a 95%/95% distribution-free two-sided tolerance interval
    assert power.tolerance_sample_size(0.95, 0.95) == 93
    assert power.tolerance_sample_size(0.99, 0.95) == 473


@pytest.fixture
def data():
    rng = np.random.default_rng(15)
    n = 400
    x1, x2, noise = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.normal(size=n)
    kind = np.array(["a", "b"], dtype=object)[rng.integers(0, 2, n)]
    y = np.sin(x1) * 3 + (x2 > 0) * 2 + np.where(kind == "b", 4.0, 0.0) + rng.normal(0, 0.3, n)
    label = np.where((x1 > 0) & (x2 > 0), "odd", "even").astype(object)   # one quadrant against the rest
    return Worksheet("D", pd.DataFrame({"y": y, "x1": x1, "x2": x2, "noise": noise, "kind": kind, "label": label}))


def tables(res):
    return {payload[1]: payload[0] for kind, payload in res.blocks if kind == "table"}


def test_regression_models_learn_structure(data):
    for analysis, kwargs in ((predictive.cart_regression, {"max_depth": 6}),
                             (predictive.forest_regression, {"trees": 60}),
                             (predictive.boosting_regression, {"trees": 120})):
        res = analysis(data, "y", ["x1", "x2", "noise"], ["kind"], folds=3, store=True, **kwargs)
        t = tables(res)
        summary = t["Model Summary"]
        assert summary.loc["R-sq", "3-fold"] > 80
        assert summary.loc["R-sq", summary.columns[0]] >= summary.loc["R-sq", "3-fold"]
        importance = t["Variable Importance"]["Relative importance (%)"]
        assert importance["noise"] < 15 and importance.max() == pytest.approx(100.0)
        assert len(res.stored["PFITS_y"]) == data.nrows


def test_classification_models_learn_quadrant(data):
    for analysis, kwargs in ((predictive.cart_classification, {"max_depth": 4}),
                             (predictive.forest_classification, {"trees": 60}),
                             (predictive.boosting_classification, {"trees": 80})):
        res = analysis(data, "label", ["x1", "x2", "noise"], folds=3, store=True, **kwargs)
        t = tables(res)
        summary = t["Model Summary"]
        assert summary.loc["Accuracy (%)", "3-fold"] > 90
        assert summary.loc["Area under ROC curve", "3-fold"] > 0.9
        confusion = t["Confusion Matrix (cross-validation)"]
        assert confusion.to_numpy().sum() == 400 and list(confusion.index) == ["even", "odd"]
        assert set(res.stored["PFITS_label"]) == {"even", "odd"}


def test_predictive_input_checks(data):
    with pytest.raises(ValueError):
        predictive.cart_regression(data, "y")
    with pytest.raises(ValueError):
        predictive.cart_regression(data, "y", ["y"])
    one_class = Worksheet(df=pd.DataFrame({"c": np.array(["a"] * 20, dtype=object), "x": np.arange(20.0)}))
    with pytest.raises(ValueError):
        predictive.cart_classification(one_class, "c", ["x"])
    assert len(predictive.cart_regression(data, "y", ["x1"], folds=0).figures()) == 2
