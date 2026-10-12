"""Multivariate and time-series analyses checked against scikit-learn, statsmodels and exact constructions."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from sklearn.decomposition import PCA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as sm_acf

from openstat.core.worksheet import Worksheet
from openstat.stats import multivariate as mv
from openstat.stats import timeseries as ts


def tables(res):
    return {payload[1]: payload[0] for kind, payload in res.blocks if kind == "table"}


@pytest.fixture
def frame():
    rng = np.random.default_rng(10)
    latent = rng.normal(size=(120, 2))
    data = latent @ rng.normal(size=(2, 5)) + 0.4 * rng.normal(size=(120, 5))
    return pd.DataFrame(data * [1, 5, 2, 10, 3], columns=list("abcde"))


def test_pca_matches_sklearn(frame):
    out = mv.principal_components(frame, "Correlation")
    z = (frame - frame.mean()) / frame.std(ddof=1)
    ref = PCA().fit(z)
    assert out["eigenvalues"] == pytest.approx(ref.explained_variance_)
    assert out["eigenvalues"].sum() == pytest.approx(5.0)
    assert np.abs(out["vectors"][:, 0]) == pytest.approx(np.abs(ref.components_[0]))
    assert np.cov(out["scores"], rowvar=False).diagonal() == pytest.approx(out["eigenvalues"])
    cov = mv.principal_components(frame, "Covariance")
    assert cov["eigenvalues"].sum() == pytest.approx(frame.var(ddof=1).sum())


def test_pca_stores_scores_aligned(frame):
    frame = frame.copy()
    frame.loc[3, "a"] = np.nan
    res = mv.pca(Worksheet("P", frame), list("abcde"), components=2, store=True)
    assert list(res.stored) == ["PC1", "PC2"] and np.isnan(res.stored["PC1"][3])
    assert tables(res)["Eigenvectors"].shape == (5, 2)


def test_varimax_preserves_communalities(frame):
    unrotated = mv.factor_loadings(frame, 2, rotation="None")
    rotated = mv.factor_loadings(frame, 2, rotation="Varimax")
    assert (rotated ** 2).sum(axis=1) == pytest.approx((unrotated ** 2).sum(axis=1))
    simplicity = lambda L: ((L ** 2).var(axis=0)).sum()   # noqa: E731
    assert simplicity(rotated) >= simplicity(unrotated) - 1e-9
    assert mv.factor_loadings(frame, 2, mv.EXTRACTION[1]).shape == (5, 2)
    with pytest.raises(ValueError):
        mv.factor_loadings(frame, 6)


def test_cronbach_alpha():
    base = np.random.default_rng(1).normal(size=100)
    identical = pd.DataFrame({"a": base, "b": base, "c": base})
    assert mv.cronbach_alpha(identical) == pytest.approx(1.0)
    items = pd.DataFrame({k: base + np.random.default_rng(i).normal(size=100) for i, k in enumerate("abcd")})
    k, cov = 4, items.cov().to_numpy()
    assert mv.cronbach_alpha(items) == pytest.approx(k / (k - 1) * (1 - np.trace(cov) / cov.sum()))


def blobs():
    rng = np.random.default_rng(2)
    data = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(6, 0.5, (30, 2)), rng.normal([0, 6], 0.5, (30, 2))])
    return Worksheet("B", pd.DataFrame({"x": data[:, 0], "y": data[:, 1],
                                        "g": np.repeat(["g1", "g2", "g3"], 30).astype(object)}))


def _same_partition(labels, truth):
    return pd.crosstab(np.asarray(labels), np.asarray(truth)).to_numpy().max(axis=0).sum() == len(truth)


def test_clustering_recovers_groups():
    ws = blobs()
    for analysis in (mv.cluster_observations, mv.k_means):
        res = analysis(ws, ["x", "y"], clusters=3, store=True)
        assert _same_partition(res.stored["Cluster"], ws.df["g"])
    with pytest.raises(ValueError):
        mv.cluster_observations(ws, ["x", "y"], linkage="Ward", distance="Manhattan")


def test_cluster_variables_groups_correlated_columns(frame):
    frame = frame.assign(f=frame["a"] + 0.01, g=np.random.default_rng(0).normal(size=len(frame)))
    res = mv.cluster_variables(Worksheet("V", frame), ["a", "f", "g"], clusters=2)
    partition = tables(res)["Final Partition"]["Cluster"]
    assert partition["a"] == partition["f"] != partition["g"]


def test_discriminant_function_reproduces_classification():
    ws = blobs()
    res = mv.discriminant(ws, "g", ["x", "y"], store=True)
    t = tables(res)
    assert t["Summary of Classification"].loc["Proportion"].tolist() == [1.0, 1.0, 1.0]
    function = t["Linear Discriminant Function for Groups"]
    scores = function.loc["Constant"].to_numpy() + ws.df[["x", "y"]].to_numpy() @ function.loc[["x", "y"]].to_numpy()
    assert (function.columns[scores.argmax(axis=1)] == np.array(res.stored["Predicted group"])).all()
    assert mv.discriminant(ws, "g", ["x", "y"], "Quadratic", cross_validate=True).blocks


def test_correspondence_inertia_is_chi_square_over_n():
    table = np.array([[20, 10, 5], [8, 15, 12], [4, 9, 22], [10, 10, 10]])
    out = mv.correspondence(table)
    chi2 = sps.chi2_contingency(table, correction=False).statistic
    assert out["total"] == pytest.approx(chi2 / table.sum())
    assert len(out["inertia"]) == 2
    assert out["row_mass"] @ out["rows"] == pytest.approx([0, 0], abs=1e-12)


# ----- time series -----

def seasonal_series(n=72, season=12):
    t = np.arange(1, n + 1)
    pattern = 1 + 0.3 * np.sin(2 * np.pi * (t - 1) / season)
    return (100 + 2 * t) * pattern, pattern[:season] / pattern[:season].mean()


def test_series_rules():
    ws = Worksheet(df=pd.DataFrame({"a": [1.0, 2, 3, np.nan], "b": [1.0, np.nan, 3, 4], "c": [1.0, np.nan, np.nan, np.nan]}))
    assert ts.series(ws, "a").tolist() == [1.0, 2.0, 3.0]
    for bad in ("b", "c"):
        with pytest.raises(ValueError):
            ts.series(ws, bad)


def test_trend_models_fit_exact_data():
    t = np.arange(1, 31, dtype=float)
    for kind, x in ((ts.TRENDS[0], 5 + 2 * t), (ts.TRENDS[1], 5 + 2 * t + 0.1 * t ** 2),
                    (ts.TRENDS[2], 3 * 1.07 ** t), (ts.TRENDS[3], 100 / (1 + 20 * np.exp(-0.3 * t)))):
        model, _ = ts.fit_trend(x, kind)
        assert model(t) == pytest.approx(x, rel=1e-5)
        assert model(np.array([31.0]))[0] > x[-1]
    assert ts.accuracy([10.0, 20.0], [9.0, 22.0]) == {"MAPE": pytest.approx(10.0), "MAD": 1.5, "MSD": 2.5}


def test_centered_moving_average():
    x = np.arange(1.0, 11.0)
    assert ts.centered_moving_average(x, 3)[1:-1] == pytest.approx(x[1:-1])
    even = ts.centered_moving_average(x, 4)
    assert np.isnan(even[:2]).all() and np.isnan(even[-2:]).all()
    assert even[2:-2] == pytest.approx(x[2:-2])      # a linear series is its own centred average


def test_decomposition_recovers_seasonal_indices():
    x, true_indices = seasonal_series()
    out = ts.decompose(x, 12, multiplicative=True)
    assert out["indices"] == pytest.approx(true_indices, abs=0.01)
    assert out["indices"].mean() == pytest.approx(1.0)
    assert out["slope"] == pytest.approx(2.0, rel=0.05)
    assert ts.accuracy(x, out["fitted"])["MAPE"] < 2.0
    assert out["predict"](np.array([73]))[0] == pytest.approx((out["intercept"] + out["slope"] * 73) * out["indices"][0])
    additive = ts.decompose(x, 12, multiplicative=False)
    assert additive["indices"].sum() == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        ts.decompose(x[:20], 12)


def test_single_exponential_recursion():
    x = np.array([10.0, 12, 11, 13, 12, 14, 13, 15])
    out = ts.single_exponential(x, 0.3)
    level = x[:6].mean()
    assert out["fitted"][0] == pytest.approx(level)
    assert out["fitted"][1] == pytest.approx(0.3 * x[0] + 0.7 * level)
    best = ts.single_exponential(x)
    assert 0 < best["alpha"] < 1
    assert np.sum((x - best["fitted"]) ** 2) <= np.sum((x - out["fitted"]) ** 2) + 1e-9


def test_smoothing_reports_and_forecasts():
    x, _ = seasonal_series()
    ws = Worksheet("T", pd.DataFrame({"y": x}))
    for analysis, kwargs in ((ts.moving_average, {"length": 3}), (ts.double_exp_smoothing, {}),
                             (ts.winters, {"season": 12}), (ts.decomposition, {"season": 12}),
                             (ts.trend_analysis, {})):
        res = analysis(ws, "y", forecasts=6, store=True, **kwargs)
        t = tables(res)
        assert len(t["Forecasts"]) == 6 and t["Forecasts"].index[0] == 73
        assert len(res.stored["FITS_y"]) == 72
    winters = tables(ts.winters(ws, "y", season=12, forecasts=12))
    assert winters["Accuracy Measures"].iloc[0]["MAPE"] < 5
    fixed = ts.double_exp_smoothing(ws, "y", alpha=0.5, gamma=0.2)
    assert "Alpha" in fixed.blocks[0][1] and "0.5" in fixed.blocks[0][1]


def test_acf_and_ljung_box_match_statsmodels():
    rng = np.random.default_rng(4)
    x = np.zeros(300)
    for i in range(1, 300):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    table = ts.autocorrelation(x, 10)
    assert table["ACF"].to_numpy() == pytest.approx(sm_acf(x, nlags=10)[1:])
    ref = acorr_ljungbox(x, lags=[5, 10])
    assert table.loc[5, "LBQ"] == pytest.approx(ref.loc[5, "lb_stat"])
    assert table.loc[10, "P-Value"] == pytest.approx(ref.loc[10, "lb_pvalue"])
    assert table.loc[1, "T"] == pytest.approx(table.loc[1, "ACF"] * np.sqrt(300))


def test_cross_correlation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    y = np.concatenate([[0.0, 0.0], x[:-2]]) + 0.1 * rng.normal(size=200)   # y lags x by two periods
    values = ts.cross_correlation(x, y, 5)
    assert values.idxmax() == 2 and values[2] > 0.9
    assert ts.cross_correlation(x, y, 0)[0] == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_arima_recovers_ar1():
    rng = np.random.default_rng(6)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = 5 + 0.6 * (x[i - 1] - 5) + rng.normal() if i > 1 else 5.0
    ws = Worksheet("A", pd.DataFrame({"y": x[1:]}))
    res = ts.arima(ws, "y", p=1, forecasts=5, store=True)
    t = tables(res)
    estimates = t["Final Estimates of Parameters"]
    assert estimates.loc["AR 1", "Coef"] == pytest.approx(0.6, abs=0.1)
    assert estimates.loc["Constant", "Coef"] == pytest.approx(5.0, abs=0.5)
    forecast = t["Forecasts (95% limits)"]
    assert (forecast["Lower"] < forecast["Forecast"]).all() and (forecast["Forecast"] < forecast["Upper"]).all()
    assert 12 in t["Modified Box-Pierce (Ljung-Box) Chi-Square Statistic"].index
    seasonal = ts.arima(ws, "y", p=0, d=1, q=1, seasonal_q=1, seasonal_d=1, season=12, constant=False)
    assert "SMA 12" in tables(seasonal)["Final Estimates of Parameters"].index
