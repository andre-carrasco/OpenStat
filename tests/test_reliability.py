"""Life-data analysis checked against SciPy, lifelines and closed-form results."""
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, WeibullAFTFitter, WeibullFitter
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import _life, reliability


@pytest.fixture
def censored():
    rng = np.random.default_rng(12)
    life = sps.weibull_min(1.8, scale=100).rvs(150, random_state=rng)
    limit = rng.uniform(40, 220, 150)
    return np.minimum(life, limit), life <= limit


def test_complete_data_matches_scipy_and_closed_forms():
    x = sps.weibull_min(2.2, scale=50).rvs(200, random_state=1)
    fit = _life.fit_life(x, x, name="Weibull")
    shape, _, scale = sps.weibull_min.fit(x, floc=0)
    assert 1 / fit.sigma == pytest.approx(shape, rel=1e-4)
    assert np.exp(fit.beta[0]) == pytest.approx(scale, rel=1e-4)
    fit = _life.fit_life(x, x, name="Lognormal")
    assert fit.beta[0] == pytest.approx(np.log(x).mean(), abs=1e-5)
    assert fit.sigma == pytest.approx(np.log(x).std(ddof=0), rel=1e-4)
    assert fit.mean() == pytest.approx(np.exp(fit.beta[0] + fit.sigma ** 2 / 2))


def test_exponential_with_censoring_closed_form(censored):
    t, failed = censored
    fit = _life.fit_life(*_life.right_censored(t, failed), name="Exponential")
    assert np.exp(fit.beta[0]) == pytest.approx(t.sum() / failed.sum(), rel=1e-5)
    assert fit.se_beta()[0] == pytest.approx(1 / np.sqrt(failed.sum()), rel=1e-3)
    assert fit.mean() == pytest.approx(t.sum() / failed.sum(), rel=1e-5)


def test_censored_weibull_matches_lifelines(censored):
    t, failed = censored
    fit = _life.fit_life(*_life.right_censored(t, failed), name="Weibull")
    ref = WeibullFitter().fit(t, failed)
    assert np.exp(fit.beta[0]) == pytest.approx(ref.lambda_, rel=1e-3)
    assert 1 / fit.sigma == pytest.approx(ref.rho_, rel=1e-3)
    assert fit.loglik == pytest.approx(ref.log_likelihood_, rel=1e-6)
    table = reliability.parameter_table(fit)
    assert table.loc["Scale", "Standard error"] == pytest.approx(ref.summary.loc["lambda_", "se(coef)"], rel=2e-2)
    assert table.loc["Shape", "Standard error"] == pytest.approx(ref.summary.loc["rho_", "se(coef)"], rel=2e-2)
    median, lo, hi = fit.quantile(0.5)
    assert median == pytest.approx(ref.median_survival_time_, rel=1e-3) and lo < median < hi
    assert fit.survival(median) == pytest.approx(0.5, abs=1e-9)


def test_interval_censoring_matches_lifelines():
    life = sps.weibull_min(1.5, scale=30).rvs(200, random_state=5)
    lower, upper = np.floor(life / 5) * 5, np.floor(life / 5) * 5 + 5
    upper[life > 60] = np.inf
    lower[life > 60] = 60
    fit = _life.fit_life(lower, upper, name="Weibull")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = WeibullFitter().fit_interval_censoring(np.where(lower == 0, 1e-9, lower), upper)
    assert np.exp(fit.beta[0]) == pytest.approx(ref.lambda_, rel=5e-3)
    assert 1 / fit.sigma == pytest.approx(ref.rho_, rel=5e-3)


def test_life_regression_matches_lifelines_aft():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(size=n)
    life = np.exp(4.0 - 0.6 * x) * sps.weibull_min(2.0).rvs(n, random_state=rng)
    limit = rng.uniform(30, 300, n)
    t, failed = np.minimum(life, limit), life <= limit
    fit = _life.fit_life(*_life.right_censored(t, failed), X=np.column_stack([np.ones(n), x]), name="Weibull")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = WeibullAFTFitter().fit(pd.DataFrame({"t": t, "e": failed, "x": x}), "t", "e")
    assert fit.beta[1] == pytest.approx(ref.params_[("lambda_", "x")], abs=2e-3)
    assert fit.beta[0] == pytest.approx(ref.params_[("lambda_", "Intercept")], abs=2e-3)
    assert 1 / fit.sigma == pytest.approx(np.exp(ref.params_[("rho_", "Intercept")]), rel=2e-3)
    assert fit.beta[1] == pytest.approx(-0.6, abs=0.1)


def test_kaplan_meier_matches_lifelines(censored):
    t, failed = censored
    table = _life.kaplan_meier(t, failed)
    ref = KaplanMeierFitter().fit(t, failed)
    expected = ref.survival_function_at_times(table["Time"]).to_numpy()
    assert table["Survival probability"].to_numpy() == pytest.approx(expected)
    assert _life.km_median(table) == pytest.approx(ref.median_survival_time_)
    simple = _life.kaplan_meier([1, 2, 3, 4], [True, False, True, True])
    assert simple["Survival probability"].tolist() == pytest.approx([0.75, 0.375, 0.0])


def test_plotting_positions_without_censoring():
    times, probs = _life.plotting_positions([3.0, 1.0, 2.0], [True, True, True])
    assert times.tolist() == [1.0, 2.0, 3.0]
    assert probs == pytest.approx([1 / 4, 2 / 4, 3 / 4])     # Herd-Johnson reduces to i / (n + 1)


def test_reports_run_with_text_censoring_column(censored):
    t, failed = censored
    ws = Worksheet("L", pd.DataFrame({"t": t, "status": np.where(failed, "F", "C").astype(object),
                                      "g": np.array(["a", "b"], dtype=object)[np.arange(len(t)) % 2]}))
    res = reliability.parametric_analysis(ws, "t", "status", "C")
    tables = {p[1]: p[0] for k, p in res.blocks if k == "table"}
    assert tables["Parameter Estimates"].loc["Shape", "Estimate"] == pytest.approx(1.8, abs=0.35)
    assert tables["Table of Percentiles"]["Percentile"].is_monotonic_increasing
    res = reliability.nonparametric_analysis(ws, "t", "status", "C", by="g")
    tables = {p[1]: p[0] for k, p in res.blocks if k == "table"}
    assert tables["Test Statistics"].iloc[0]["DF"] == 1
    assert tables["Censoring Information"]["Failures"].sum() == failed.sum()
    all_censored = Worksheet(df=pd.DataFrame({"t": [1.0, 2.0], "c": [0.0, 0.0]}))
    with pytest.raises(ValueError):
        reliability.life_data(all_censored, "t", "c", "0")


def test_power_law_process():
    t = np.array([5.0, 40, 43, 175, 389, 712, 747, 795, 1299, 1478])
    out = reliability.power_law_process(t)
    assert out["shape"] == pytest.approx(len(t) / np.log(t[-1] / t).sum())
    assert (t[-1] / out["scale"]) ** out["shape"] == pytest.approx(len(t))
    assert out["shape"] < 1
    truncated = reliability.power_law_process(t, end=2000.0)
    assert truncated["mil_df"] == 20 and truncated["total"] == 2000.0
    hpp = np.cumsum(np.random.default_rng(3).exponential(10, 200))
    assert reliability.power_law_process(hpp)["mil_p"] > 0.01


def test_demonstration_plan_zero_failures():
    # success-run formula: n = ln(1 - C) / ln(R)
    assert reliability.demonstration_sample_size(0.90, 1000, 1000, 1.0, 0.95) == 29
    assert reliability.demonstration_sample_size(0.90, 1000, 2000, 2.0, 0.95) == 8
    n1 = reliability.demonstration_sample_size(0.90, 1000, 1000, 1.0, 0.95, failures=1)
    assert sps.binom.cdf(1, n1, 0.1) <= 0.05 < sps.binom.cdf(1, n1 - 1, 0.1)


def test_probit_percentiles():
    dose = np.array([1.0, 2, 3, 4, 5, 6])
    trials = np.full(6, 50.0)
    events = np.array([3.0, 9, 20, 31, 42, 47])
    ws = Worksheet(df=pd.DataFrame({"d": dose, "n": trials, "k": events}))
    res = reliability.probit(ws, "k", "n", "d")
    tables = {p[1]: p[0] for k, p in res.blocks if k == "table"}
    ref = sm.GLM(np.column_stack([events, trials - events]), sm.add_constant(dose),
                 family=sm.families.Binomial(sm.families.links.Probit())).fit()
    assert tables["Regression Table"].loc["d", "Coef"] == pytest.approx(ref.params[1])
    assert tables["Table of Percentiles"].loc[50, "Percentile"] == pytest.approx(-ref.params[0] / ref.params[1])
    row = tables["Table of Percentiles"].loc[50]
    assert row["CI Lower"] < row["Percentile"] < row["CI Upper"]


def test_stress_transformations():
    assert reliability.transform_stress([25.0], reliability.RELATIONSHIPS[0])[0] == pytest.approx(11604.83 / 298.16)
    assert reliability.transform_stress([np.e], reliability.RELATIONSHIPS[2])[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        reliability.transform_stress([-1.0], reliability.RELATIONSHIPS[2])
