"""Time series: trend, decomposition, smoothing, correlation functions and ARIMA."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.graphs import new_figure
from openstat.i18n import tr

MENU = ("Stat", "Time Series")
BLUE, RED, GREEN, GRAY = "#5b9bd5", "#c00000", "#00803c", "#7f7f7f"

SERIES = Param("column", "Series", "column")
FORECASTS = Param("forecasts", "Number of forecasts", "int", default=0, minimum=0, maximum=10000)
STORE = Param("store", "Store fits and residuals", "bool", default=False)
SEASON = Param("season", "Seasonal length", "int", default=12, minimum=2, maximum=10000)


def series(ws, column, minimum=3) -> np.ndarray:
    """The series from the first row; missing values are only allowed at the end of the column."""
    x = ws.df[column].to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < minimum:
        raise ValueError(f"The series needs at least {minimum} observations")
    x = x[:valid[-1] + 1]
    if np.isnan(x).any():
        raise ValueError("The series has missing values before its last observation")
    return x


def accuracy(actual, fitted) -> dict:
    actual, fitted = np.asarray(actual, dtype=float), np.asarray(fitted, dtype=float)
    ok = ~np.isnan(fitted)
    error = actual[ok] - fitted[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(error / actual[ok])
    return {"MAPE": float(100 * np.mean(ape[np.isfinite(ape)])) if np.isfinite(ape).any() else np.nan,
            "MAD": float(np.mean(np.abs(error))), "MSD": float(np.mean(error ** 2))}


def _plot(column, actual, fitted, forecast=None, title="", extra=None):
    fig = new_figure(8.0, 4.0)
    ax = fig.add_subplot()
    index = np.arange(1, len(actual) + 1)
    ax.plot(index, actual, "-o", color=BLUE, markersize=3, label=tr("Actual"))
    ax.plot(index, fitted, "-", color=RED, label=tr("Fits"))
    if extra is not None:
        ax.plot(index, extra[1], "--", color=GRAY, label=extra[0])
    if forecast is not None and len(forecast):
        ax.plot(np.arange(len(actual) + 1, len(actual) + len(forecast) + 1), forecast, "-s", color=GREEN,
                markersize=3, label=tr("Forecasts"))
    ax.set_xlabel(tr("Index")), ax.set_ylabel(column), ax.set_title(title), ax.legend(fontsize=8)
    return fig


def _report(res: Result, column, actual, fitted, forecast, title, store, extra=None) -> Result:
    res.table(pd.DataFrame([accuracy(actual, fitted)]), title="Accuracy Measures", index=False)
    if forecast is not None and len(forecast):
        res.table(pd.DataFrame({"Forecast": forecast},
                               index=pd.Index(np.arange(len(actual) + 1, len(actual) + len(forecast) + 1),
                                              name="Period")), title="Forecasts")
    res.figure(_plot(column, actual, fitted, forecast, f"{tr(title)}: {column}", extra))
    if store:
        res.store(f"FITS_{column}", list(fitted)).store(f"RESI_{column}", list(np.asarray(actual) - fitted))
    return res


# ----- trend -----

TRENDS = ["Linear", "Quadratic", "Exponential growth", "S-curve (Pearl-Reed logistic)"]


def fit_trend(x: np.ndarray, kind: str):
    """Returns (function of t giving the fitted value, text of the equation)."""
    t = np.arange(1, len(x) + 1, dtype=float)
    if kind == TRENDS[0]:
        b1, b0 = np.polyfit(t, x, 1)
        return (lambda s: b0 + b1 * s), f"Yt = {b0:.5g} + {b1:.5g} t"
    if kind == TRENDS[1]:
        b2, b1, b0 = np.polyfit(t, x, 2)
        return (lambda s: b0 + b1 * s + b2 * s ** 2), f"Yt = {b0:.5g} + {b1:.5g} t + {b2:.5g} t^2"
    if (x <= 0).any():
        raise ValueError("This trend model needs positive data")
    if kind == TRENDS[2]:
        b1, b0 = np.polyfit(t, np.log(x), 1)
        return (lambda s: np.exp(b0) * np.exp(b1) ** s), f"Yt = {np.exp(b0):.5g} * ({np.exp(b1):.5g} ^ t)"

    def curve(s, a, b, c):
        return a / (1 + b * np.exp(-c * s))

    start = [x.max() * 1.2, max(x.max() / max(x[0], 1e-9) - 1, 0.1), 4 / len(x)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (a, b, c), _ = optimize.curve_fit(curve, t, x, p0=start, maxfev=20000)
    except RuntimeError as exc:
        raise ValueError("The S-curve model did not converge") from exc
    return (lambda s: curve(s, a, b, c)), f"Yt = {a:.5g} / (1 + {b:.5g} * exp(-{c:.5g} t))"


@analysis("ts.trend", menu=MENU, title="Trend Analysis",
          params=[SERIES, Param("kind", "Model type", "choice", default="Linear", choices=TRENDS), FORECASTS, STORE])
def trend_analysis(ws, column, kind="Linear", forecasts=0, store=False):
    x = series(ws, column)
    model, text = fit_trend(x, kind)
    n = len(x)
    fitted = model(np.arange(1, n + 1, dtype=float))
    forecast = model(np.arange(n + 1, n + forecasts + 1, dtype=float)) if forecasts else None
    res = Result("Trend Analysis")
    res.text(f"{tr(kind)}:  {text}")
    return _report(res, column, x, fitted, forecast, "Trend Analysis", store)


# ----- decomposition -----

MODEL_TYPES = ["Multiplicative", "Additive"]


def centered_moving_average(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    half = length // 2
    if length % 2:
        for i in range(half, len(x) - half):
            out[i] = x[i - half:i + half + 1].mean()
    else:  # two-step average centres an even window
        for i in range(half, len(x) - half):
            window = x[i - half:i + half + 1]
            out[i] = (0.5 * window[0] + window[1:-1].sum() + 0.5 * window[-1]) / length
    return out


def seasonal_indices(x: np.ndarray, season: int, multiplicative: bool) -> np.ndarray:
    """Median of the detrended values in each seasonal period, adjusted to average 1 (or sum to 0)."""
    cma = centered_moving_average(x, season)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = x / cma if multiplicative else x - cma
    period = np.arange(len(x)) % season
    indices = np.array([np.nanmedian(raw[period == p]) for p in range(season)])
    return indices / indices.mean() if multiplicative else indices - indices.mean()


def decompose(x: np.ndarray, season: int, multiplicative=True, trend=True) -> dict:
    if len(x) < 2 * season:
        raise ValueError("The series needs at least two full seasonal cycles")
    if multiplicative and (x <= 0).any():
        raise ValueError("The multiplicative model needs positive data")
    indices = seasonal_indices(x, season, multiplicative)
    n = len(x)
    t = np.arange(1, n + 1, dtype=float)
    seasonal = indices[np.arange(n) % season]
    deseasonalized = x / seasonal if multiplicative else x - seasonal
    slope, intercept = np.polyfit(t, deseasonalized, 1) if trend else (0.0, deseasonalized.mean())

    def predict(s):
        s = np.asarray(s, dtype=float)
        level = intercept + slope * s
        season_part = indices[(s.astype(int) - 1) % season]
        return level * season_part if multiplicative else level + season_part

    return {"indices": indices, "slope": float(slope), "intercept": float(intercept), "fitted": predict(t),
            "trend": intercept + slope * t, "predict": predict, "deseasonalized": deseasonalized}


@analysis("ts.decomposition", menu=MENU, title="Decomposition",
          params=[SERIES, SEASON, Param("kind", "Model type", "choice", default="Multiplicative", choices=MODEL_TYPES),
                  Param("trend", "Model components: trend plus seasonal", "bool", default=True), FORECASTS, STORE])
def decomposition(ws, column, season=12, kind="Multiplicative", trend=True, forecasts=0, store=False):
    x = series(ws, column)
    out = decompose(x, season, kind == "Multiplicative", trend)
    n = len(x)
    forecast = out["predict"](np.arange(n + 1, n + forecasts + 1)) if forecasts else None
    res = Result("Decomposition")
    res.text(f"{tr(kind)}    {tr('Seasonal length')}: {season}")
    res.text(f"{tr('Fitted trend equation')}:  Yt = {out['intercept']:.5g} + {out['slope']:.5g} t")
    res.table(pd.DataFrame({"Index": out["indices"]}, index=pd.Index(range(1, season + 1), name="Period")),
              title="Seasonal Indices", digits=5)
    return _report(res, column, x, out["fitted"], forecast, "Decomposition", store, (tr("Trend"), out["trend"]))


# ----- smoothing -----

@analysis("ts.moving_average", menu=MENU, title="Moving Average",
          params=[SERIES, Param("length", "MA length", "int", default=3, minimum=1),
                  Param("center", "Center the moving averages", "bool", default=False), FORECASTS, STORE])
def moving_average(ws, column, length=3, center=False, forecasts=0, store=False):
    x = series(ws, column, minimum=length + 1)
    n = len(x)
    if center:
        smoothed = centered_moving_average(x, length)
        fitted = smoothed
    else:
        smoothed = np.full(n, np.nan)
        for i in range(length - 1, n):
            smoothed[i] = x[i - length + 1:i + 1].mean()
        fitted = np.concatenate([[np.nan], smoothed[:-1]])   # one-step-ahead forecasts
    forecast = np.full(forecasts, smoothed[~np.isnan(smoothed)][-1]) if forecasts else None
    res = Result("Moving Average")
    res.text(f"{tr('MA length')}: {length}")
    return _report(res, column, x, fitted, forecast, "Moving Average", store)


def single_exponential(x: np.ndarray, alpha: float | None = None) -> dict:
    """Single exponential smoothing; alpha is chosen by least squares when not given."""
    start = float(x[:min(6, len(x))].mean())

    def run(a):
        level, fitted = start, np.empty(len(x))
        for i, value in enumerate(x):
            fitted[i] = level
            level = a * value + (1 - a) * level
        return fitted, level

    if alpha is None:
        best = optimize.minimize_scalar(lambda a: np.sum((x - run(a)[0]) ** 2), bounds=(0.001, 0.999),
                                        method="bounded")
        alpha = float(best.x)
    fitted, level = run(alpha)
    return {"alpha": alpha, "fitted": fitted, "level": level}


@analysis("ts.single_exp", menu=MENU, title="Single Exponential Smoothing",
          params=[SERIES, Param("alpha", "Weight for level (0 = optimal)", "number", default=0.0, minimum=0.0,
                                maximum=1.0), FORECASTS, STORE])
def single_exp_smoothing(ws, column, alpha=0.0, forecasts=0, store=False):
    x = series(ws, column)
    out = single_exponential(x, alpha or None)
    forecast = np.full(forecasts, out["level"]) if forecasts else None
    res = Result("Single Exponential Smoothing")
    res.text(f"Alpha = {out['alpha']:.5g}")
    return _report(res, column, x, out["fitted"], forecast, "Single Exponential Smoothing", store)


def _holt_winters(x, trend, seasonal, season, weights: dict, forecasts: int):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fixed = {k: v for k, v in weights.items() if v}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(x, trend=trend, seasonal=seasonal, seasonal_periods=season if seasonal else None,
                                     initialization_method="estimated")
        if fixed:
            with model.fix_params(fixed):
                fit = model.fit()
        else:
            fit = model.fit()
    forecast = np.asarray(fit.forecast(forecasts)) if forecasts else None
    return fit, forecast


@analysis("ts.double_exp", menu=MENU, title="Double Exponential Smoothing",
          params=[SERIES, Param("alpha", "Weight for level (0 = optimal)", "number", default=0.0, minimum=0.0,
                                maximum=1.0),
                  Param("gamma", "Weight for trend (0 = optimal)", "number", default=0.0, minimum=0.0, maximum=1.0),
                  FORECASTS, STORE])
def double_exp_smoothing(ws, column, alpha=0.0, gamma=0.0, forecasts=0, store=False):
    x = series(ws, column, minimum=4)
    fit, forecast = _holt_winters(x, "add", None, None, {"smoothing_level": alpha, "smoothing_trend": gamma},
                                  forecasts)
    res = Result("Double Exponential Smoothing")
    res.text(f"Alpha ({tr('level')}) = {fit.params['smoothing_level']:.5g}    "
             f"Gamma ({tr('trend')}) = {fit.params['smoothing_trend']:.5g}")
    return _report(res, column, x, np.asarray(fit.fittedvalues), forecast, "Double Exponential Smoothing", store)


@analysis("ts.winters", menu=MENU, title="Winters' Method",
          params=[SERIES, SEASON, Param("kind", "Seasonality", "choice", default="Multiplicative", choices=MODEL_TYPES),
                  Param("alpha", "Weight for level (0 = optimal)", "number", default=0.0, minimum=0.0, maximum=1.0),
                  Param("gamma", "Weight for trend (0 = optimal)", "number", default=0.0, minimum=0.0, maximum=1.0),
                  Param("delta", "Weight for seasonal (0 = optimal)", "number", default=0.0, minimum=0.0,
                        maximum=1.0), FORECASTS, STORE])
def winters(ws, column, season=12, kind="Multiplicative", alpha=0.0, gamma=0.0, delta=0.0, forecasts=0, store=False):
    x = series(ws, column)
    if len(x) < 2 * season:
        raise ValueError("The series needs at least two full seasonal cycles")
    if kind == "Multiplicative" and (x <= 0).any():
        raise ValueError("The multiplicative model needs positive data")
    fit, forecast = _holt_winters(x, "add", "mul" if kind == "Multiplicative" else "add", season,
                                  {"smoothing_level": alpha, "smoothing_trend": gamma, "smoothing_seasonal": delta},
                                  forecasts)
    res = Result("Winters' Method")
    res.text(f"{tr(kind)}    {tr('Seasonal length')}: {season}")
    res.text(f"Alpha ({tr('level')}) = {fit.params['smoothing_level']:.5g}    "
             f"Gamma ({tr('trend')}) = {fit.params['smoothing_trend']:.5g}    "
             f"Delta ({tr('seasonal')}) = {fit.params['smoothing_seasonal']:.5g}")
    return _report(res, column, x, np.asarray(fit.fittedvalues), forecast, "Winters' Method", store)


# ----- differences and lags -----

@analysis("ts.differences", menu=MENU, title="Differences",
          params=[SERIES, Param("lag", "Lag", "int", default=1, minimum=1)])
def differences(ws, column, lag=1):
    x = ws.df[column].astype(float)
    res = Result("Differences").store(f"Diff{lag}_{column}", (x - x.shift(lag)).tolist())
    return res.text(f"{column}(t) - {column}(t-{lag})    {tr('Stored in columns')}: Diff{lag}_{column}")


@analysis("ts.lag", menu=MENU, title="Lag", params=[SERIES, Param("lag", "Lag", "int", default=1, minimum=1)])
def lag_column(ws, column, lag=1):
    res = Result("Lag").store(f"Lag{lag}_{column}", ws.df[column].astype(float).shift(lag).tolist())
    return res.text(f"{column}(t-{lag})    {tr('Stored in columns')}: Lag{lag}_{column}")


# ----- correlation functions -----

def autocorrelation(x: np.ndarray, lags: int) -> pd.DataFrame:
    """Autocorrelations with large-lag standard errors, t statistics and Ljung-Box Q."""
    n = len(x)
    centered = x - x.mean()
    denom = float(centered @ centered)
    if denom == 0:
        raise ValueError("The series has no variation")
    r = np.array([centered[k:] @ centered[:n - k] / denom for k in range(1, lags + 1)])
    se = np.sqrt((1 + 2 * np.concatenate([[0], np.cumsum(r[:-1] ** 2)])) / n)
    q = n * (n + 2) * np.cumsum(r ** 2 / (n - np.arange(1, lags + 1)))
    return pd.DataFrame({"ACF": r, "T": r / se, "LBQ": q, "P-Value": sps.chi2.sf(q, np.arange(1, lags + 1))},
                        index=pd.Index(range(1, lags + 1), name="Lag"))


def default_lags(n: int) -> int:
    return int(max(1, min(n - 1, n // 4 if n <= 240 else np.sqrt(n) + 45)))


def _correlogram(values, limit, title, ylabel, lags=None):
    fig = new_figure(8.0, 3.8)
    ax = fig.add_subplot()
    lags = np.arange(1, len(values) + 1) if lags is None else lags
    ax.vlines(lags, 0, values, color=BLUE, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    limit = np.broadcast_to(limit, (len(values),))
    ax.plot(lags, limit, "--", color=RED, linewidth=1), ax.plot(lags, -limit, "--", color=RED, linewidth=1)
    ax.set_ylim(-1.05, 1.05), ax.set_xlabel(tr("Lag")), ax.set_ylabel(ylabel), ax.set_title(title)
    return fig


LAGS = Param("lags", "Number of lags (0 = default)", "int", default=0, minimum=0)


@analysis("ts.acf", menu=MENU, title="Autocorrelation", params=[SERIES, LAGS])
def acf(ws, column, lags=0):
    x = series(ws, column)
    lags = min(lags or default_lags(len(x)), len(x) - 1)
    table = autocorrelation(x, lags)
    limit = 1.96 * (table["ACF"] / table["T"]).abs().to_numpy()
    res = Result("Autocorrelation")
    res.figure(_correlogram(table["ACF"].to_numpy(), limit, f"{tr('Autocorrelation Function')}: {column}",
                            tr("Autocorrelation")))
    return res.table(table)


@analysis("ts.pacf", menu=MENU, title="Partial Autocorrelation", params=[SERIES, LAGS])
def pacf(ws, column, lags=0):
    from statsmodels.tsa.stattools import pacf as sm_pacf

    x = series(ws, column, minimum=6)
    lags = min(lags or default_lags(len(x)), len(x) // 2 - 1)
    values = sm_pacf(x, nlags=lags, method="ywm")[1:]
    se = 1 / np.sqrt(len(x))
    res = Result("Partial Autocorrelation")
    res.figure(_correlogram(values, 1.96 * se, f"{tr('Partial Autocorrelation Function')}: {column}",
                            tr("Partial autocorrelation")))
    return res.table(pd.DataFrame({"PACF": values, "T": values / se}, index=pd.Index(range(1, lags + 1), name="Lag")))


def cross_correlation(x: np.ndarray, y: np.ndarray, lags: int) -> pd.Series:
    """Correlation between x(t) and y(t + k) for k = -lags..lags."""
    n = len(x)
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    if denom == 0:
        raise ValueError("The series has no variation")
    values = {}
    for k in range(-lags, lags + 1):
        values[k] = (xc[:n - k] @ yc[k:] if k >= 0 else xc[-k:] @ yc[:n + k]) / denom
    return pd.Series(values, name="CCF")


@analysis("ts.ccf", menu=MENU, title="Cross Correlation",
          params=[Param("first", "First series", "column"), Param("second", "Second series", "column"), LAGS])
def ccf(ws, first, second, lags=0):
    frame = ws.df[[first, second]].dropna().astype(float)
    n = len(frame)
    if n < 4:
        raise ValueError("The series needs at least 4 observations")
    lags = min(lags or int(np.sqrt(n) + 10), n - 2)
    values = cross_correlation(frame[first].to_numpy(), frame[second].to_numpy(), lags)
    res = Result("Cross Correlation")
    res.text(f"CCF: {first}(t), {second}(t+k)")
    res.figure(_correlogram(values.to_numpy(), 2 / np.sqrt(n), f"{tr('Cross Correlation Function')}: {first}, {second}",
                            tr("Cross correlation"), values.index.to_numpy()))
    table = values.to_frame()
    table.index.name = "Lag"
    return res.table(table)


# ----- ARIMA -----

@analysis("ts.arima", menu=MENU, title="ARIMA",
          params=[SERIES,
                  Param("p", "Autoregressive order (p)", "int", default=1, minimum=0, maximum=5),
                  Param("d", "Differencing order (d)", "int", default=0, minimum=0, maximum=2),
                  Param("q", "Moving average order (q)", "int", default=0, minimum=0, maximum=5),
                  Param("seasonal_p", "Seasonal autoregressive order (P)", "int", default=0, minimum=0, maximum=5),
                  Param("seasonal_d", "Seasonal differencing order (D)", "int", default=0, minimum=0, maximum=2),
                  Param("seasonal_q", "Seasonal moving average order (Q)", "int", default=0, minimum=0, maximum=5),
                  Param("season", "Seasonal length", "int", default=12, minimum=2, maximum=10000),
                  Param("constant", "Include constant term in model", "bool", default=True), FORECASTS, STORE])
def arima(ws, column, p=1, d=0, q=0, seasonal_p=0, seasonal_d=0, seasonal_q=0, season=12, constant=True,
          forecasts=0, store=False):
    from statsmodels.tsa.arima.model import ARIMA

    x = series(ws, column, minimum=10)
    if p + q + seasonal_p + seasonal_q == 0 and not constant:
        raise ValueError("The model has no terms")
    seasonal = (seasonal_p, seasonal_d, seasonal_q, season) if seasonal_p + seasonal_d + seasonal_q else (0, 0, 0, 0)
    differenced = d + seasonal_d > 0
    trend = ("t" if d + seasonal_d == 1 else "n") if differenced and constant else ("c" if constant else "n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(x, order=(p, d, q), seasonal_order=seasonal, trend=trend).fit()
    names = [n for n in fit.param_names if n != "sigma2"]
    index = [list(fit.param_names).index(n) for n in names]
    table = pd.DataFrame({"Coef": np.asarray(fit.params)[index], "SE Coef": np.asarray(fit.bse)[index],
                          "T-Value": np.asarray(fit.tvalues)[index], "P-Value": np.asarray(fit.pvalues)[index]},
                         index=pd.Index([_arima_name(n) for n in names], name="Type"))
    burn = d + seasonal_d * season
    resid = np.asarray(fit.resid)[burn:]
    k = len(names)
    res = Result("ARIMA")
    order_text = f"ARIMA({p},{d},{q})" + (f"({seasonal_p},{seasonal_d},{seasonal_q}){season}" if seasonal[3] else "")
    res.text(f"{order_text}    N = {len(x)}")
    res.table(table, title="Final Estimates of Parameters")
    df = len(resid) - k
    res.table(pd.DataFrame([{"DF": df, "SS": float(resid @ resid), "MS": float(resid @ resid / df) if df > 0 else np.nan,
                             "AIC": fit.aic, "BIC": fit.bic}]), title="Residual Sums of Squares", index=False)
    lb_rows = {}
    for lag in (12, 24, 36, 48):
        if lag < len(resid) and lag > k:
            q_stat = float(autocorrelation(resid, lag)["LBQ"].iloc[-1])
            lb_rows[lag] = {"Chi-Square": q_stat, "DF": lag - k, "P-Value": float(sps.chi2.sf(q_stat, lag - k))}
    if lb_rows:
        lb = pd.DataFrame.from_dict(lb_rows, orient="index")
        lb.index.name = "Lag"
        res.table(lb, title="Modified Box-Pierce (Ljung-Box) Chi-Square Statistic")
    forecast = None
    if forecasts:
        prediction = fit.get_forecast(forecasts)
        interval = np.asarray(prediction.conf_int(alpha=0.05))
        forecast = np.asarray(prediction.predicted_mean)
        res.table(pd.DataFrame({"Forecast": forecast, "Lower": interval[:, 0], "Upper": interval[:, 1]},
                               index=pd.Index(np.arange(len(x) + 1, len(x) + forecasts + 1), name="Period")),
                  title="Forecasts (95% limits)")
    fitted = np.asarray(fit.fittedvalues, dtype=float).copy()
    fitted[:burn] = np.nan
    res.figure(_plot(column, x, fitted, forecast, f"ARIMA: {column}"))
    if store:
        res.store(f"FITS_{column}", fitted.tolist()).store(f"RESI_{column}", (x - fitted).tolist())
    return res


MAX_ORDER = Param("max_order", "Largest AR and MA order to try", "int", default=3, minimum=1, maximum=5)


def arima_candidates(x, d=1, max_order=3, seasonal_d=0, season=0, constant=True) -> pd.DataFrame:
    """Every ARIMA(p, d, q) up to `max_order`, with seasonal orders up to 1 when `season` is given,
    ordered by the corrected Akaike criterion (AICc); models that cannot be fitted are left out."""
    from statsmodels.tsa.arima.model import ARIMA

    seasonal_orders = [(sp, sq) for sp in (0, 1) for sq in (0, 1)] if season else [(0, 0)]
    differenced = d + seasonal_d > 0
    trend = ("t" if d + seasonal_d == 1 else "n") if differenced and constant else ("c" if constant else "n")
    rows = []
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            for sp, sq in seasonal_orders:
                if p + q + sp + sq == 0 and trend == "n":
                    continue
                seasonal = (sp, seasonal_d, sq, season) if season and sp + sq + seasonal_d else (0, 0, 0, 0)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = ARIMA(x, order=(p, d, q), seasonal_order=seasonal, trend=trend).fit()
                except Exception:   # some orders cannot be estimated from a short series
                    continue
                if np.isfinite(fit.aicc):
                    rows.append({"p": p, "d": d, "q": q, "P": sp, "D": seasonal_d if season else 0, "Q": sq,
                                 "Log-likelihood": float(fit.llf), "AICc": float(fit.aicc), "AIC": float(fit.aic),
                                 "BIC": float(fit.bic)})
    if not rows:
        raise ValueError("No ARIMA model could be fitted to the series")
    return pd.DataFrame(rows).sort_values("AICc", kind="stable").reset_index(drop=True)


@analysis("ts.best_arima", menu=MENU, title="Forecast with Best ARIMA Model",
          params=[SERIES, Param("d", "Differencing order (d)", "int", default=1, minimum=0, maximum=2), MAX_ORDER,
                  Param("seasonal", "Try seasonal terms", "bool", default=False),
                  Param("seasonal_d", "Seasonal differencing order (D)", "int", default=0, minimum=0, maximum=1),
                  SEASON, Param("constant", "Include constant term in model", "bool", default=True),
                  Param("forecasts", "Number of forecasts", "int", default=6, minimum=0, maximum=10000), STORE])
def best_arima(ws, column, d=1, max_order=3, seasonal=False, seasonal_d=0, season=12, constant=True, forecasts=6,
               store=False):
    x = series(ws, column, minimum=10)
    if seasonal and len(x) < 2 * season + 5:
        raise ValueError("The series is too short for seasonal terms of this length")
    table = arima_candidates(x, d, max_order, seasonal_d if seasonal else 0, season if seasonal else 0, constant)
    best = table.iloc[0]
    res = Result("Forecast with Best ARIMA Model")
    shown = table.head(10).drop(columns=[] if seasonal else ["P", "D", "Q"])
    res.table(shown, title=f"{tr('Model selection')} (AICc, {len(table)} {tr('models')})", index=False, digits=3)
    report = arima(ws, column, int(best["p"]), d, int(best["q"]), int(best["P"]), int(best["D"]), int(best["Q"]),
                   season, constant, forecasts, store)
    res.blocks += report.blocks
    res.stored = report.stored
    return res


ADF_TERMS = ["Constant", "Constant and linear trend", "None"]


@analysis("ts.adf", menu=MENU, title="Augmented Dickey-Fuller Test",
          params=[SERIES, Param("lags", "Maximum lag order (0 = chosen by AIC)", "int", default=0, minimum=0,
                                maximum=100),
                  Param("terms", "Deterministic terms", "choice", default=ADF_TERMS[0], choices=ADF_TERMS)])
def augmented_dickey_fuller(ws, column, lags=0, terms=ADF_TERMS[0]):
    from statsmodels.tsa.stattools import adfuller

    x = series(ws, column, minimum=10)
    regression = dict(zip(ADF_TERMS, ["c", "ct", "n"]))[terms]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # the return type is changing between statsmodels versions
        # the tuple is one value longer when the lag order is chosen automatically
        statistic, p, used, n, critical = adfuller(x, maxlag=lags or None, regression=regression,
                                                   autolag=None if lags else "AIC")[:5]
    res = Result("Augmented Dickey-Fuller Test")
    res.text(f"{tr('H0: the series has a unit root (it is not stationary)')}    {tr('Deterministic terms')}: "
             f"{tr(terms)}")
    res.table(pd.DataFrame([{"Test statistic": float(statistic), "P-Value": float(p), "Lags used": int(used),
                             "N": int(n), **{f"{tr('Critical value')} {k}": v for k, v in critical.items()}}]),
              index=False)
    res.text(tr("The data give evidence that the series is stationary.") if p < 0.05
             else tr("A unit root cannot be ruled out; consider differencing the series."))
    return res


def _arima_name(name: str) -> str:
    kind, _, lag = name.partition(".L")
    labels = {"ar": "AR", "ma": "MA", "ar.S": "SAR", "ma.S": "SMA"}
    if kind in labels:
        return f"{labels[kind]} {lag}"
    return {"const": "Constant", "drift": "Constant", "x1": "Constant"}.get(name, name)
