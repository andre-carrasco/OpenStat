"""Test plans, failure modes, repairable systems, warranty data, unit-root test and ARIMA selection,
checked against closed forms, statsmodels and direct calculation."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import _life, reliability, timeseries


def tables(res):
    return [payload[0] for kind, payload in res.blocks if kind == "table"]


def test_weighted_fit_equals_repeated_rows():
    lower = np.array([10.0, 25.0, 40.0, 60.0, 30.0])
    upper = np.array([10.0, 25.0, 50.0, np.inf, np.inf])
    counts = np.array([3, 1, 4, 6, 2])
    repeated = _life.fit_life(np.repeat(lower, counts), np.repeat(upper, counts), name="Weibull")
    weighted = _life.fit_life(lower, upper, name="Weibull", weights=counts)
    assert weighted.beta[0] == pytest.approx(repeated.beta[0], rel=1e-5)
    assert weighted.sigma == pytest.approx(repeated.sigma, rel=1e-5)
    assert weighted.loglik == pytest.approx(repeated.loglik, rel=1e-8)
    assert (weighted.n, weighted.failures) == (16, 4) == (repeated.n, repeated.failures)
    assert weighted.se_beta()[0] == pytest.approx(repeated.se_beta()[0], rel=1e-3)


def test_fisher_information_of_censored_observations():
    assert _life.censored_information("Normal") == pytest.approx(np.diag([1.0, 2.0]), abs=1e-6)
    # smallest extreme value: 1, 1 - gamma and pi^2 / 6 + (1 - gamma)^2
    g = 1 - np.euler_gamma
    assert _life.censored_information("Weibull") == pytest.approx(np.array([[1, g], [g, np.pi ** 2 / 6 + g ** 2]]),
                                                                    abs=1e-5)
    full, half = _life.censored_information("Lognormal"), _life.censored_information("Lognormal", 0.0)
    assert np.all(np.linalg.eigvalsh(full - half) > 0)                 # censoring loses information
    # the large-sample covariance of a censored fit is the inverse of this information
    rng = np.random.default_rng(10)
    n = 4000
    y = rng.normal(5.0, 0.5, n)
    t, failed = np.exp(np.minimum(y, 5.25)), y <= 5.25
    fit = _life.fit_life(*_life.right_censored(t, failed), name="Lognormal")
    expected = np.linalg.inv(_life.censored_information("Lognormal", 0.5)) * 0.5 ** 2 / n
    assert fit.se_beta()[0] ** 2 == pytest.approx(expected[0, 0], rel=0.08)
    assert fit.se_sigma() ** 2 == pytest.approx(expected[1, 1], rel=0.08)


def test_estimation_test_plan():
    z = sps.norm.ppf(0.975)
    # the mean of a normal distribution: n = (z sigma / precision)^2
    assert reliability.estimation_plan("Normal", 50.0, 4.0, 50.0, 1.0)["n"] == int(np.ceil((z * 4.0) ** 2))
    # the median of a lognormal: the same on the log scale, where the bound is exp(-z se) times the estimate
    out = reliability.estimation_plan("Lognormal", np.log(100.0), 0.8, 50.0, 20.0)
    assert out["percentile"] == pytest.approx(100.0)
    assert out["n"] == int(np.ceil((0.8 * z / np.log(1 - 20.0 / 100.0)) ** 2))
    censored = reliability.estimation_plan("Lognormal", np.log(100.0), 0.8, 50.0, 20.0, censor_time=100.0)
    assert censored["n"] > out["n"] and censored["failing"] == pytest.approx(0.5)
    assert reliability.planning_parameters("Weibull", 2.0, 100.0) == (pytest.approx(np.log(100.0)), 0.5)
    exponential = reliability.estimation_plan("Exponential", np.log(500.0), 1.0, 50.0, 100.0)
    assert exponential["n"] == int(np.ceil((z / np.log(1 - 100.0 / (500.0 * np.log(2)))) ** 2))
    row = tables(reliability.estimation_test_plan(None, "Weibull", 2.0, 100.0, 10.0, 10.0, 80.0))[0].iloc[0]
    assert row["Planning percentile"] == pytest.approx(100 * (-np.log(0.9)) ** 0.5)
    assert row["Expected failures (%)"] == pytest.approx(100 * (1 - np.exp(-0.8 ** 2)))
    with pytest.raises(ValueError):
        reliability.estimation_plan("Weibull", np.log(100.0), 0.5, 10.0, 50.0)   # more than the percentile itself


def test_multiple_failure_modes():
    rng = np.random.default_rng(14)
    n = 3000
    wear, shock = rng.exponential(200.0, n), rng.exponential(300.0, n)
    limit = np.full(n, 400.0)
    t = np.minimum.reduce([wear, shock, limit])
    mode = np.where(t == limit, None, np.where(wear < shock, "wear", "shock")).astype(object)
    ws = Worksheet("F", pd.DataFrame({"t": t, "mode": mode}))
    out = tables(reliability.failure_modes(ws, "t", "mode", "Exponential"))
    assert out[0].loc["Mean", "Estimate"] == pytest.approx(300.0, rel=0.08)     # shock comes first alphabetically
    assert out[1].loc["Mean", "Estimate"] == pytest.approx(200.0, rel=0.08)
    direct = _life.fit_life(*_life.right_censored(t, mode == "wear"), name="Exponential")
    assert out[1].loc["Mean", "Estimate"] == pytest.approx(np.exp(direct.beta[0]))
    rate = 1 / out[0].loc["Mean", "Estimate"] + 1 / out[1].loc["Mean", "Estimate"]   # the hazards add up
    assert out[2].loc[50, "Percentile"] == pytest.approx(np.log(2) / rate, rel=1e-6)
    assert out[2].loc[50, "Percentile"] == pytest.approx(np.log(2) * 120.0, rel=0.06)
    with pytest.raises(ValueError):
        reliability.failure_modes(Worksheet("N", pd.DataFrame({"t": [1.0, 2.0], "mode": [None, None]})), "t", "mode")


def test_mean_cumulative_function():
    time, system = [5.0, 8.0, 12.0], ["a", "a", "b"]
    table = reliability.mean_cumulative_function(time, system)
    assert table["Mean cumulative function"].tolist() == [0.5, 1.0, 1.5] and table["Systems at risk"].tolist() == [2, 2, 2]
    assert table["Standard error"].iloc[-1] == pytest.approx(np.sqrt(3 / 4))
    # system a leaves at 10, so only b is watched when it fails at 12
    table = reliability.mean_cumulative_function(time + [10.0], system + ["a"], [False, False, False, True])
    assert table["Mean cumulative function"].tolist() == [0.5, 1.0, 2.0] and table["Systems at risk"].tolist() == [2, 2, 1]
    ws = Worksheet("M", pd.DataFrame({"t": time + [10.0], "id": np.array(system + ["a"], dtype=object),
                                      "kind": np.array(["f", "f", "f", "end"], dtype=object)}))
    out = tables(reliability.nonparametric_growth(ws, "t", "id", "kind", "end"))[0]
    assert out["Mean cumulative function"].tolist() == [0.5, 1.0, 2.0] and (out["CI Lower"] >= 0).all()
    assert tables(reliability.nonparametric_growth(ws, "t", "id"))[0]["Failures"].sum() == 4


def test_warranty_preprocessing_and_prediction():
    #            returns in period 1, 2, 3
    returns = [[2.0, 3.0, 4.0],      # shipped in period 1: 100 units
               [np.nan, 1.0, 5.0],   # period 2: 80 units
               [np.nan, np.nan, 2.0]]
    frame = reliability.warranty_intervals([100.0, 80.0, 60.0], returns)
    assert frame.iloc[:4].to_numpy().tolist()[:3] == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    assert len(frame) == 9 and frame["Frequencies"].sum() == 240.0
    assert frame[frame["End time"].isna()][["Start time", "Frequencies"]].to_numpy().tolist() == [
        [3.0, 91.0], [2.0, 74.0], [1.0, 58.0]]
    assert frame[(frame["Start time"] == 0) & frame["End time"].notna()]["Frequencies"].sum() == 5.0
    with pytest.raises(ValueError):
        reliability.warranty_intervals([5.0], [[9.0]])

    ws = Worksheet("W", pd.DataFrame({"ship": [100.0, 80.0, 60.0], "r1": [2.0, np.nan, np.nan],
                                      "r2": [3.0, 1.0, np.nan], "r3": [4.0, 5.0, 2.0]}))
    data = reliability.warranty_preprocess(ws, "ship", ["r1", "r2", "r3"]).worksheets[0][1]
    assert data.equals(frame)
    out = tables(reliability.warranty_prediction(Worksheet("D", data), "Start time", "End time", "Frequencies",
                                                 "Exponential", 4))
    assert out[0].iloc[0].tolist() == [240.0, 17.0, 223.0]
    mean = out[1].loc["Mean", "Estimate"]
    # the exponential forgets its age: each unit in service fails within m periods with 1 - exp(-m / mean)
    assert out[2]["Predicted failures"].tolist() == pytest.approx([223 * (1 - np.exp(-m / mean)) for m in (1, 2, 3, 4)])
    assert (out[2]["Lower"] < out[2]["Predicted failures"]).all() and (out[2]["Upper"] > out[2]["Predicted failures"]).all()
    fit = _life.fit_life([1.0, 2.0], [1.0, 2.0], name="Weibull")
    assert reliability.warranty_forecast(fit, [], [], 2).tolist() == [0.0, 0.0]


@pytest.fixture
def ar1():
    rng = np.random.default_rng(22)
    x = np.zeros(300)
    for i in range(1, 300):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    return x


def test_augmented_dickey_fuller(ar1):
    from statsmodels.tsa.stattools import adfuller

    walk = np.cumsum(np.random.default_rng(23).normal(size=300))
    ws = Worksheet("T", pd.DataFrame({"ar": ar1, "walk": walk}))
    stationary = tables(timeseries.augmented_dickey_fuller(ws, "ar"))[0].iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ref = adfuller(ar1, regression="c", autolag="AIC")
        ref_fixed = adfuller(walk, maxlag=4, regression="ct", autolag=None)
    assert stationary["Test statistic"] == pytest.approx(ref[0]) and stationary["P-Value"] == pytest.approx(ref[1])
    assert stationary["P-Value"] < 0.01 and stationary["Lags used"] == ref[2]
    assert tables(timeseries.augmented_dickey_fuller(ws, "walk"))[0].iloc[0]["P-Value"] > 0.05
    fixed = tables(timeseries.augmented_dickey_fuller(ws, "walk", 4, timeseries.ADF_TERMS[1]))[0].iloc[0]
    assert fixed["Lags used"] == 4 and fixed["Test statistic"] == pytest.approx(ref_fixed[0])


def test_best_arima_selection(ar1):
    from statsmodels.tsa.arima.model import ARIMA

    table = timeseries.arima_candidates(ar1, d=0, max_order=2)
    assert len(table) == 9 and table["AICc"].is_monotonic_increasing
    best = table.iloc[0]
    assert best["p"] >= 1                                              # white noise does not win
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ARIMA(ar1, order=(int(best["p"]), 0, int(best["q"])), trend="c").fit()
    assert best["AICc"] == pytest.approx(ref.aicc)
    ws = Worksheet("T", pd.DataFrame({"ar": ar1}))
    res = timeseries.best_arima(ws, "ar", 0, 2, forecasts=5, store=True)
    out = tables(res)
    assert "P" not in out[0].columns and len(out[-1]) == 5 and set(res.stored) == {"FITS_ar", "RESI_ar"}
    assert any(kind == "text" and f"ARIMA({int(best['p'])},0,{int(best['q'])})" in payload for kind, payload in res.blocks)
    seasonal = timeseries.arima_candidates(ar1[:80], d=0, max_order=1, seasonal_d=0, season=4)
    assert len(seasonal) > 4 and {"P", "Q"} <= set(seasonal.columns)
    with pytest.raises(ValueError):
        timeseries.best_arima(Worksheet("S", pd.DataFrame({"x": ar1[:20]})), "x", seasonal=True)
