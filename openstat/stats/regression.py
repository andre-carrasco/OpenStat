"""Regression: linear models, model selection, nonlinear, generalized linear, PLS and orthogonal."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize
from scipy import stats as sps
from statsmodels.stats.outliers_influence import variance_inflation_factor

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import Worksheet, format_value, parse_cell
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._design import CONSTANT, Design, build_design, complete_rows, report_reference
from openstat.stats._tests import CONFIDENCE
from openstat.stats._util import median_ranks
from openstat.stats.quality import LSL, USL, optional_number

MENU = ("Stat", "Regression")
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"
PALETTE = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#7030a0", "#c00000", "#00b0f0", "#7f7f7f"]

RESPONSE = Param("response", "Response", "column")
CONTINUOUS = Param("continuous", "Continuous predictors", "columns", optional=True)
CATEGORICAL = Param("categorical", "Categorical predictors", "columns", numeric=False, optional=True)
INTERACTIONS = Param("interactions", "Include two-way interactions", "bool", default=False)
STORE = Param("store", "Store residuals and fits", "bool", default=False)
PLOTS = Param("plots", "Residual plots", "bool", default=True)


# ----- ordinary least squares -----

def term_tests(fit, design: Design) -> pd.DataFrame:
    """Adjusted (partial) sums of squares and F-tests for each model term."""
    names = list(design.X.columns)
    mse, dfe = fit.mse_resid, fit.df_resid
    rows = {}
    for term, cols in design.terms:
        restriction = np.zeros((len(cols), len(names)))
        for i, col in enumerate(cols):
            restriction[i, names.index(col)] = 1.0
        test = fit.f_test(restriction)
        f = float(np.squeeze(test.fvalue))
        rows[term] = {"DF": len(cols), "Adj SS": f * len(cols) * mse, "Adj MS": f * mse, "F-Value": f,
                      "P-Value": float(test.pvalue)}
    table = pd.DataFrame.from_dict(rows, orient="index")
    regression = {"DF": fit.df_model, "Adj SS": fit.ess, "Adj MS": fit.mse_model, "F-Value": fit.fvalue,
                  "P-Value": fit.f_pvalue}
    error = {"DF": dfe, "Adj SS": fit.ssr, "Adj MS": mse, "F-Value": np.nan, "P-Value": np.nan}
    total = {"DF": fit.df_model + dfe, "Adj SS": fit.centered_tss, "Adj MS": np.nan, "F-Value": np.nan,
             "P-Value": np.nan}
    out = pd.concat([pd.DataFrame([regression], index=["Regression"]), table,
                     pd.DataFrame([error, total], index=["Error", "Total"])])
    out.index.name = "Source"
    return out


def model_summary(fit) -> dict:
    influence = fit.get_influence()
    press = float(np.sum((fit.resid / (1 - influence.hat_matrix_diag)) ** 2))
    return {"S": float(np.sqrt(fit.mse_resid)), "R-sq": 100 * fit.rsquared, "R-sq(adj)": 100 * fit.rsquared_adj,
            "PRESS": press, "R-sq(pred)": 100 * max(0.0, 1 - press / fit.centered_tss)}


def coefficient_table(fit, design: Design) -> pd.DataFrame:
    X = design.X.to_numpy()
    vif = [np.nan] + [variance_inflation_factor(X, i) if X.shape[1] > 2 else 1.0 for i in range(1, X.shape[1])]
    table = pd.DataFrame({"Coef": fit.params, "SE Coef": fit.bse, "T-Value": fit.tvalues, "P-Value": fit.pvalues,
                          "VIF": vif})
    table.index.name = "Term"
    return table


def equation(response: str, params: pd.Series) -> str:
    parts = []
    for name, value in params.items():
        if name == CONSTANT:
            parts.append(f"{value:.5g}")
        else:
            parts.append(f"{'+' if value >= 0 else '-'} {abs(value):.5g} {name}")
    return f"{response} = " + " ".join(parts)


def residual_plots(fits, residuals, title: str):
    """Four-in-one: normal probability, versus fits, histogram and versus order."""
    fits, residuals = np.asarray(fits, dtype=float), np.asarray(residuals, dtype=float)
    fig = new_figure(8.0, 6.0)
    (a, b), (c, d) = fig.subplots(2, 2)
    ordered = np.sort(residuals)
    a.plot(ordered, sps.norm.ppf(median_ranks(len(ordered))), "o", color=BLUE, markersize=4)
    sd = residuals.std(ddof=1)
    if sd > 0:
        line = np.array([ordered[0], ordered[-1]])
        a.plot(line, (line - residuals.mean()) / sd, color=RED)
    a.set_title(tr("Normal Probability Plot")), a.set_xlabel(tr("Residual")), a.set_ylabel(tr("Normal score"))
    b.plot(fits, residuals, "o", color=BLUE, markersize=4)
    b.axhline(0, color=GRAY, linestyle="--")
    b.set_title(tr("Versus Fits")), b.set_xlabel(tr("Fitted value")), b.set_ylabel(tr("Residual"))
    c.hist(residuals, bins="auto", color=BLUE, edgecolor="white")
    c.set_title(tr("Histogram")), c.set_xlabel(tr("Residual")), c.set_ylabel(tr("Frequency"))
    d.plot(np.arange(1, len(residuals) + 1), residuals, "-o", color=BLUE, markersize=4)
    d.axhline(0, color=GRAY, linestyle="--")
    d.set_title(tr("Versus Order")), d.set_xlabel(tr("Observation order")), d.set_ylabel(tr("Residual"))
    fig.suptitle(f"{tr('Residual Plots')}: {title}")
    return fig


def unusual_observations(fit, design: Design, y) -> pd.DataFrame:
    influence = fit.get_influence()
    std = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    p, n = design.X.shape[1], len(y)
    large, high = np.abs(std) > 2, leverage > min(0.99, 3 * p / n)
    keep = large | high
    flags = np.where(large & high, "R X", np.where(large, "R", "X"))
    return pd.DataFrame({"Response": np.asarray(y)[keep], "Fit": np.asarray(fit.fittedvalues)[keep],
                         "Residual": np.asarray(fit.resid)[keep], "Std Resid": std[keep], "Flag": flags[keep]},
                        index=pd.Index(design.rows[keep] + 1, name="Obs"))


def fit_ols(ws, response, design: Design):
    y = ws.df[response].iloc[design.rows].astype(float)
    if np.linalg.matrix_rank(design.X.to_numpy()) < design.X.shape[1]:
        raise ValueError("The predictors are linearly dependent; remove a redundant term")
    return y, sm.OLS(y.to_numpy(), design.X).fit()


def linear_report(res: Result, ws, response, design: Design, store=False, plots=True, show_equation=True):
    y, fit = fit_ols(ws, response, design)
    if show_equation:
        res.heading("Regression Equation").text(equation(response, fit.params))
    report_reference(res, design)
    res.table(coefficient_table(fit, design), title="Coefficients")
    res.table(pd.DataFrame([model_summary(fit)]), title="Model Summary", index=False)
    res.table(term_tests(fit, design), title="Analysis of Variance")
    unusual = unusual_observations(fit, design, y)
    if len(unusual):
        res.table(unusual.head(50), title="Fits and Diagnostics for Unusual Observations")
        res.text(tr("R: large residual    X: unusual X (high leverage)"))
    res.text(f"{tr('Durbin-Watson statistic')} = {sm.stats.durbin_watson(fit.resid):.5f}")
    if plots:
        res.figure(residual_plots(fit.fittedvalues, fit.resid, response))
    if store:
        res.store(f"FITS_{response}", design.aligned(fit.fittedvalues))
        res.store(f"RESI_{response}", design.aligned(fit.resid))
    return fit


@analysis("reg.fit", menu=MENU, title="Fit Regression Model",
          params=[RESPONSE, CONTINUOUS, CATEGORICAL, INTERACTIONS, PLOTS, STORE])
def fit_regression(ws, response, continuous=None, categorical=None, interactions=False, plots=True, store=False):
    design = build_design(ws, [response], continuous, categorical, interactions)
    res = Result("Fit Regression Model")
    linear_report(res, ws, response, design, store, plots)
    return res


# ----- prediction -----

def parse_points(text: str, names: list[str]) -> list[dict]:
    """'X=50, Group=A; X=55, Group=B' -> one {predictor: value} per point."""
    points = []
    for part in str(text).split(";"):
        if not part.strip():
            continue
        point = {}
        for item in part.split(","):
            if "=" not in item:
                raise ValueError(f"Invalid predictor value: {item.strip()}")
            name, value = (s.strip() for s in item.split("=", 1))
            if name not in names:
                raise ValueError(f"Unknown predictor: {name}")
            point[name] = value
        missing = [n for n in names if n not in point]
        if missing:
            raise ValueError(f"No value given for: {', '.join(missing)}")
        points.append(point)
    if not points:
        raise ValueError("Enter the values of the predictors")
    return points


def design_rows(ws, design: Design, response, continuous, categorical, interactions, points: list[dict],
                coding="reference") -> pd.DataFrame:
    """Rows of the model matrix for new settings of the predictors (`points`: {predictor: value})."""
    continuous, categorical = list(continuous or []), list(categorical or [])
    new = {response: [0.0] * len(points)}   # a placeholder: only the predictors of these rows are used
    for name in continuous:
        try:
            new[name] = [float(str(p[name]).replace(",", ".")) for p in points]
        except ValueError:
            raise ValueError(f"{name}: the value must be a number") from None
    for name in categorical:
        values = [parse_cell(p[name]) if ws.is_numeric(name) and isinstance(p[name], str) else p[name]
                  for p in points]
        unknown = [format_value(v) for v in values if format_value(v) not in design.levels[name]]
        if unknown:
            raise ValueError(f"{name}: unknown level {unknown[0]}")
        new[name] = values
    used = [response] + continuous + categorical
    extended = Worksheet("points", pd.concat([ws.df[used].iloc[design.rows], pd.DataFrame(new)], ignore_index=True))
    extended.value_orders = ws.value_orders   # same reference levels as the fitted model
    return build_design(extended, [response], continuous, categorical, interactions, coding).X.iloc[-len(points):]


@analysis("reg.predict", menu=MENU, title="Predict",
          params=[RESPONSE, CONTINUOUS, CATEGORICAL, INTERACTIONS,
                  Param("values", "Predictor values (e.g. X=50, Group=A; X=55, Group=B)", "text"), CONFIDENCE])
def predict(ws, response, continuous=None, categorical=None, interactions=False, values="", confidence=95.0):
    design = build_design(ws, [response], continuous, categorical, interactions)
    names = list(continuous or []) + list(categorical or [])
    points = parse_points(values, names)
    _, fit = fit_ols(ws, response, design)
    rows = design_rows(ws, design, response, continuous, categorical, interactions, points)
    frame = fit.get_prediction(rows.to_numpy()).summary_frame(alpha=1 - confidence / 100)
    table = pd.DataFrame({"Fit": frame["mean"], "SE Fit": frame["mean_se"], "CI Lower": frame["mean_ci_lower"],
                          "CI Upper": frame["mean_ci_upper"], "PI Lower": frame["obs_ci_lower"],
                          "PI Upper": frame["obs_ci_upper"]})
    table.index = pd.Index([", ".join(f"{n} = {p[n]}" for n in names) for p in points], name="Settings")
    res = Result("Predict")
    res.heading("Regression Equation").text(equation(response, fit.params))
    res.table(table, title=f"{tr('Prediction')} ({confidence:g}% {tr('CI')}, {tr('PI')})")
    used = ws.df.iloc[design.rows]
    outside = sorted({n for n in (continuous or []) for x in rows[n]
                      if not used[n].min() <= x <= used[n].max()})
    if outside:
        res.text(f"{tr('Values outside the range of the data (extrapolation)')}: {', '.join(outside)}")
    return res


@analysis("reg.binary_fitted_line", menu=MENU, title="Binary Fitted Line Plot",
          params=[Param("response", "Response (two values)", "column", numeric=False),
                  Param("predictor", "Predictor", "column"), CONFIDENCE])
def binary_fitted_line(ws, response, predictor, confidence=95.0):
    design = build_design(ws, [response], [predictor])
    labels = ws.df[response].iloc[design.rows].map(format_value)
    levels = ws.levels(response, labels)
    if len(levels) != 2:
        raise ValueError("The response must have exactly two distinct values")
    y = (labels == levels[-1]).astype(float).to_numpy()
    x = design.X[predictor].to_numpy()
    fit = sm.GLM(y, design.X, family=sm.families.Binomial()).fit()
    grid = np.linspace(x.min(), x.max(), 200)
    matrix = np.column_stack([np.ones(len(grid)), grid])
    eta = matrix @ fit.params.to_numpy()
    half = sps.norm.ppf(0.5 + confidence / 200) * np.sqrt(np.einsum("ij,jk,ik->i", matrix, fit.cov_params().to_numpy(),
                                                                      matrix))
    fig = new_figure()
    ax = fig.add_subplot()
    ax.plot(x, y, "o", color=BLUE, markersize=5, alpha=0.6)
    ax.plot(grid, sps.logistic.cdf(eta), color=RED, label=tr("Event probability"))
    ax.plot(grid, sps.logistic.cdf(eta - half), color=RED, linestyle="--", label=f"{confidence:g}% {tr('CI')}")
    ax.plot(grid, sps.logistic.cdf(eta + half), color=RED, linestyle="--")
    ax.set_ylim(-0.05, 1.05), ax.set_xlabel(predictor), ax.set_ylabel(f"P({response} = {levels[-1]})")
    ax.set_title(tr("Binary Fitted Line Plot")), ax.legend()
    res = Result("Binary Fitted Line Plot")
    res.text(f"{tr('Event')}: {response} = {levels[-1]}    {tr('Events')}: {int(y.sum())}    N = {len(y)}")
    glm_report(res, fit, "Odds Ratio", confidence)
    return res.figure(fig)


# ----- stability study -----

SHELF_LIFE_SEARCH = 10   # the shelf life is looked for up to this many times the longest time in the data


def shelf_life(mean_at, half_at, lower, upper, horizon: float) -> float:
    """First time at which the confidence bound of the mean response leaves the specification.

    `mean_at(t)` is the fitted mean and `half_at(t)` the half-width of its bound. Returns 0 when
    the response is out of specification from the start and inf when it stays inside up to `horizon`.
    """
    def margin(t):
        gaps = []
        if lower is not None:
            gaps.append(mean_at(t) - half_at(t) - lower)
        if upper is not None:
            gaps.append(upper - mean_at(t) - half_at(t))
        return min(gaps)

    if margin(0.0) <= 0:
        return 0.0
    grid = np.linspace(0.0, horizon, 2001)
    values = np.array([margin(t) for t in grid])
    crossed = np.flatnonzero(values <= 0)
    if not len(crossed):
        return float("inf")
    return float(optimize.brentq(margin, grid[crossed[0] - 1], grid[crossed[0]]))


@analysis("reg.stability", menu=MENU, title="Stability Study",
          params=[RESPONSE, Param("time", "Time", "column"),
                  Param("batch", "Batch", "column", numeric=False, optional=True), LSL, USL, CONFIDENCE,
                  Param("alpha", "Significance level for pooling batches", "number", default=0.25, minimum=0.0001,
                        maximum=0.5)])
def stability_study(ws, response, time, batch=None, lsl="", usl="", confidence=95.0, alpha=0.25):
    lower, upper = optional_number(lsl), optional_number(usl)
    if lower is None and upper is None:
        raise ValueError("Enter at least one specification limit")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("The lower limit must be less than the upper limit")
    res = Result("Stability Study")
    categorical, interactions = ([batch], True) if batch else ([], False)
    design = build_design(ws, [response], [time], categorical, interactions)
    y, fit = fit_ols(ws, response, design)
    if batch:
        # the batches share a slope, and then an intercept, unless the data say otherwise
        tests = term_tests(fit, design)
        selection = [{"Term": f"{time}*{batch}", "P-Value": tests.loc[f"{time}*{batch}", "P-Value"]}]
        if selection[0]["P-Value"] >= alpha:
            interactions = False
            design = build_design(ws, [response], [time], categorical)
            y, fit = fit_ols(ws, response, design)
            selection.append({"Term": batch, "P-Value": term_tests(fit, design).loc[batch, "P-Value"]})
            if selection[1]["P-Value"] >= alpha:
                categorical = []
                design = build_design(ws, [response], [time])
                y, fit = fit_ols(ws, response, design)
        res.table(pd.DataFrame(selection), title=f"{tr('Model selection')} (α = {alpha:g})", index=False)
        model = [tr("Separate slopes and intercepts for the batches"), tr("Common slope, separate intercepts"),
                 tr("Common slope and intercept (batches pooled)")][0 if interactions else 1 if categorical else 2]
        res.text(f"{tr('Model')}: {model}")
    res.heading("Regression Equation").text(equation(response, fit.params))
    report_reference(res, design)
    res.table(coefficient_table(fit, design), title="Coefficients")
    res.table(pd.DataFrame([model_summary(fit)]), title="Model Summary", index=False)

    # a single limit is checked with a one-sided bound, two limits with a two-sided interval
    level = confidence / 100 if lower is None or upper is None else 0.5 + confidence / 200
    crit = sps.t.ppf(level, fit.df_resid)
    beta, cov = fit.params.to_numpy(), fit.cov_params().to_numpy()
    used = ws.df.iloc[design.rows]
    longest = float(used[time].max())
    groups = design.levels[batch] if categorical else [tr("All batches")]
    fig = new_figure()
    ax = fig.add_subplot()
    rows = {}
    for i, group in enumerate(groups):
        ends = [{time: 0.0, **({batch: group} if categorical else {})}, {time: 1.0, **({batch: group} if categorical else {})}]
        x0, x1 = design_rows(ws, design, response, [time], categorical, interactions, ends).to_numpy()

        def row(t, x0=x0, x1=x1):   # the model matrix is linear in time
            return x0 + t * (x1 - x0)

        def mean_at(t, row=row):
            return float(row(t) @ beta)

        def half_at(t, row=row):
            return float(crit * np.sqrt(row(t) @ cov @ row(t)))

        life = shelf_life(mean_at, half_at, lower, upper, SHELF_LIFE_SEARCH * longest)
        rows[group] = {"Intercept": mean_at(0.0), "Slope": mean_at(1.0) - mean_at(0.0), "Shelf life": life}
        color = PALETTE[i % len(PALETTE)]
        mask = (used[batch].map(format_value) == group).to_numpy() if categorical else np.ones(len(used), dtype=bool)
        ax.plot(used[time][mask], y[mask], "o", color=color, markersize=4)
        end = min(life, SHELF_LIFE_SEARCH * longest) if life > 0 else longest
        grid = np.linspace(0.0, max(longest, min(end * 1.1, 3 * longest)), 100)
        ax.plot(grid, [mean_at(t) for t in grid], color=color, label=group)
        for sign in ([-1] if upper is None else [1] if lower is None else [-1, 1]):
            ax.plot(grid, [mean_at(t) + sign * half_at(t) for t in grid], color=color, linestyle="--", linewidth=0.8)
        if 0 < life < np.inf:
            ax.axvline(life, color=color, linestyle=":", linewidth=0.8)
    for limit in (lower, upper):
        if limit is not None:
            ax.axhline(limit, color=RED)
    ax.set_xlabel(time), ax.set_ylabel(response), ax.legend()
    ax.set_title(f"{tr('Stability Study')}: {response}")
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = batch if categorical else tr("Batch")
    spec = ", ".join(f"{tr(name)} = {value:g}" for name, value in (("Lower spec", lower), ("Upper spec", upper))
                     if value is not None)
    res.table(table, title=f"{tr('Shelf life')} ({spec}; {confidence:g}% {tr('confidence')})")
    overall = table["Shelf life"].min()
    if np.isinf(overall):
        res.text(f"{tr('The response stays within specification up to the end of the search')}: "
                 f"{SHELF_LIFE_SEARCH * longest:g}")
    else:
        res.text(f"{tr('Overall shelf life (shortest of the batches)')} = {overall:.5g}")
    return res.figure(fig)


ORDERS = ["Linear", "Quadratic", "Cubic"]


@analysis("reg.fitted_line", menu=MENU, title="Fitted Line Plot",
          params=[RESPONSE, Param("predictor", "Predictor", "column"),
                  Param("order", "Type of regression model", "choice", default="Linear", choices=ORDERS),
                  Param("bands", "Display confidence and prediction intervals", "bool", default=True), CONFIDENCE])
def fitted_line_plot(ws, response, predictor, order="Linear", bands=True, confidence=95.0):
    if response == predictor:
        raise ValueError("The response cannot also be a predictor")
    frame = ws.df[[response, predictor]].dropna().astype(float)
    degree = ORDERS.index(order) + 1
    if len(frame) <= degree + 1:
        raise ValueError("Not enough observations to fit the model")
    x, y = frame[predictor].to_numpy(), frame[response].to_numpy()

    def matrix(values):
        return pd.DataFrame({CONSTANT: 1.0, **{predictor + ("" if d == 1 else f"^{d}"): values ** d
                                               for d in range(1, degree + 1)}})

    fit = sm.OLS(y, matrix(x)).fit()
    grid = np.linspace(x.min(), x.max(), 200)
    pred = fit.get_prediction(matrix(grid)).summary_frame(alpha=1 - confidence / 100)
    fig = new_figure()
    ax = fig.add_subplot()
    ax.plot(x, y, "o", color=BLUE, markersize=5)
    ax.plot(grid, pred["mean"], color=RED, label=tr("Regression"))
    if bands:
        ax.plot(grid, pred["mean_ci_lower"], color=RED, linestyle="--", label=f"{confidence:g}% {tr('CI')}")
        ax.plot(grid, pred["mean_ci_upper"], color=RED, linestyle="--")
        ax.plot(grid, pred["obs_ci_lower"], color="#7030a0", linestyle=":", label=f"{confidence:g}% {tr('PI')}")
        ax.plot(grid, pred["obs_ci_upper"], color="#7030a0", linestyle=":")
        ax.legend()
    ax.set_xlabel(predictor), ax.set_ylabel(response)
    ax.set_title(f"{tr('Fitted Line Plot')}\n{equation(response, fit.params)}", fontsize=9)
    res = Result("Fitted Line Plot")
    res.text(equation(response, fit.params))
    res.table(pd.DataFrame([{"S": np.sqrt(fit.mse_resid), "R-sq": 100 * fit.rsquared,
                             "R-sq(adj)": 100 * fit.rsquared_adj}]), title="Model Summary", index=False)
    anova = pd.DataFrame({"DF": [fit.df_model, fit.df_resid, fit.df_model + fit.df_resid],
                          "SS": [fit.ess, fit.ssr, fit.centered_tss], "MS": [fit.mse_model, fit.mse_resid, np.nan],
                          "F-Value": [fit.fvalue, np.nan, np.nan], "P-Value": [fit.f_pvalue, np.nan, np.nan]},
                         index=pd.Index(["Regression", "Error", "Total"], name="Source"))
    return res.table(anova, title="Analysis of Variance").figure(fig)


# ----- model selection -----

def _ols_columns(y, frame: pd.DataFrame, cols):
    return sm.OLS(y, sm.add_constant(frame[list(cols)], has_constant="add")).fit()


STEPWISE_METHODS = ["Stepwise", "Forward selection", "Backward elimination"]


def stepwise_select(y, frame: pd.DataFrame, method=STEPWISE_METHODS[0], alpha_enter=0.15, alpha_remove=0.15):
    """P-value based selection; returns chosen columns and the log of steps."""
    candidates = list(frame.columns)
    selected = list(candidates) if method == STEPWISE_METHODS[2] else []
    steps = []
    for _ in range(4 * len(candidates) + 4):
        changed = False
        if method != STEPWISE_METHODS[2]:
            best, best_p = None, alpha_enter
            for name in candidates:
                if name in selected:
                    continue
                p = _ols_columns(y, frame, selected + [name]).pvalues[name]
                if p < best_p:
                    best, best_p = name, p
            if best is not None:
                selected.append(best)
                steps.append((tr("Add"), best, best_p))
                changed = True
        if method != STEPWISE_METHODS[1] and selected:
            pvalues = _ols_columns(y, frame, selected).pvalues.drop("const")
            worst = pvalues.idxmax()
            if pvalues[worst] > alpha_remove:
                selected.remove(worst)
                steps.append((tr("Remove"), worst, float(pvalues[worst])))
                changed = True
        if not changed:
            break
    return selected, steps


@analysis("reg.stepwise", menu=MENU, title="Stepwise Regression",
          params=[RESPONSE, Param("predictors", "Candidate predictors", "columns"),
                  Param("method", "Method", "choice", default=STEPWISE_METHODS[0], choices=STEPWISE_METHODS),
                  Param("alpha_enter", "Alpha to enter", "number", default=0.15, minimum=0.0001, maximum=1.0),
                  Param("alpha_remove", "Alpha to remove", "number", default=0.15, minimum=0.0001, maximum=1.0)])
def stepwise(ws, response, predictors, method=STEPWISE_METHODS[0], alpha_enter=0.15, alpha_remove=0.15):
    predictors = [p for p in predictors if p != response]
    if not predictors:
        raise ValueError("Select at least one predictor")
    frame = ws.df[[response] + predictors].dropna().astype(float)
    y = frame[response].to_numpy()
    selected, steps = stepwise_select(y, frame[predictors], method, alpha_enter, alpha_remove)
    res = Result("Stepwise Regression")
    res.text(f"{tr(method)}    α {tr('to enter')} = {alpha_enter:g}    α {tr('to remove')} = {alpha_remove:g}")
    if steps:
        res.table(pd.DataFrame(steps, columns=["Action", "Term", "P-Value"],
                               index=pd.Index(range(1, len(steps) + 1), name="Step")), title="Steps")
    if not selected:
        return res.text(tr("No predictor entered the model."))
    design = build_design(ws, [response], selected)
    linear_report(res, ws, response, design, plots=False)
    return res


@analysis("reg.best_subsets", menu=MENU, title="Best Subsets Regression",
          params=[RESPONSE, Param("predictors", "Predictors", "columns"),
                  Param("models", "Models to print for each size", "int", default=2, minimum=1, maximum=5)])
def best_subsets(ws, response, predictors, models=2):
    predictors = [p for p in predictors if p != response]
    if not 1 <= len(predictors) <= 12:
        raise ValueError("Best subsets accepts between 1 and 12 predictors")
    frame = ws.df[[response] + predictors].dropna().astype(float)
    y, n = frame[response].to_numpy(), len(frame)
    if n <= len(predictors) + 1:
        raise ValueError("Not enough observations to fit the model")
    full_mse = _ols_columns(y, frame, predictors).mse_resid
    rows = []
    for size in range(1, len(predictors) + 1):
        fits = []
        for subset in combinations(predictors, size):
            fit = _ols_columns(y, frame, subset)
            fits.append((fit.rsquared, subset, fit))
        for r2, subset, fit in sorted(fits, key=lambda t: -t[0])[:models]:
            rows.append({"Vars": size, "R-sq": 100 * r2, "R-sq(adj)": 100 * fit.rsquared_adj,
                         "Mallows Cp": fit.ssr / full_mse - (n - 2 * (size + 1)),
                         "S": np.sqrt(fit.mse_resid), "Predictors": ", ".join(subset)})
    res = Result("Best Subsets Regression")
    res.text(f"{tr('Response')}: {response}")
    return res.table(pd.DataFrame(rows), index=False, digits=3)


# ----- nonlinear -----

def _exp(x, a, b):
    return a * np.exp(b * x)


def _power(x, a, b):
    return a * np.power(x, b)


def _michaelis(x, a, b):
    return a * x / (b + x)


def _logistic(x, a, b, c):
    return a / (1 + np.exp(-(x - b) / c))


def _asymptotic(x, a, b, c):
    return a - b * np.exp(-c * x)


def _gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))


def _guess_loglinear(x, y, log_x):
    ok = (y > 0) & ((x > 0) if log_x else True)
    if ok.sum() < 2:
        return [1.0, 0.1]
    slope, intercept = np.polyfit(np.log(x[ok]) if log_x else x[ok], np.log(y[ok]), 1)
    return [np.exp(intercept), slope]


NONLINEAR = {
    "Exponential: a * exp(b * x)": (_exp, ["a", "b"], lambda x, y: _guess_loglinear(x, y, False)),
    "Power: a * x ^ b": (_power, ["a", "b"], lambda x, y: _guess_loglinear(x, y, True)),
    "Michaelis-Menten: a * x / (b + x)": (_michaelis, ["a", "b"], lambda x, y: [y.max(), np.median(x)]),
    "Logistic: a / (1 + exp(-(x - b) / c))": (_logistic, ["a", "b", "c"],
                                              lambda x, y: [y.max(), np.median(x), (np.ptp(x) or 1.0) / 4]),
    "Asymptotic: a - b * exp(-c * x)": (_asymptotic, ["a", "b", "c"],
                                        lambda x, y: [y.max(), np.ptp(y) or 1.0, 1 / (abs(x.mean()) or 1.0)]),
    "Gompertz: a * exp(-b * exp(-c * x))": (_gompertz, ["a", "b", "c"],
                                            lambda x, y: [y.max() * 1.05, 1.0, 1 / (abs(x.mean()) or 1.0)]),
}
NONLINEAR_NAMES = list(NONLINEAR)


def fit_nonlinear(x, y, model: str, confidence=95.0) -> dict:
    func, names, guess = NONLINEAR[model]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) <= len(names):
        raise ValueError("Not enough observations to fit the model")
    try:
        with np.errstate(all="ignore"):
            params, cov = optimize.curve_fit(func, x, y, p0=guess(x, y), maxfev=20000)
    except RuntimeError as exc:
        raise ValueError("The nonlinear fit did not converge") from exc
    df = len(x) - len(names)
    resid = y - func(x, *params)
    sse = float(resid @ resid)
    se = np.sqrt(np.diag(cov))
    half = sps.t.ppf(0.5 + confidence / 200, df) * se
    return {"func": func, "names": names, "params": params, "se": se, "lower": params - half,
            "upper": params + half, "sse": sse, "df": df, "s": float(np.sqrt(sse / df)), "resid": resid}


@analysis("reg.nonlinear", menu=MENU, title="Nonlinear Regression",
          params=[RESPONSE, Param("predictor", "Predictor", "column"),
                  Param("model", "Expectation function", "choice", default=NONLINEAR_NAMES[0],
                        choices=NONLINEAR_NAMES), CONFIDENCE])
def nonlinear(ws, response, predictor, model=NONLINEAR_NAMES[0], confidence=95.0):
    frame = ws.df[[response, predictor]].dropna().astype(float)
    x, y = frame[predictor].to_numpy(), frame[response].to_numpy()
    out = fit_nonlinear(x, y, model, confidence)
    res = Result("Nonlinear Regression")
    res.text(f"{response} = {model.split(': ', 1)[1].replace('x', predictor)}")
    res.table(pd.DataFrame({"Estimate": out["params"], "SE Estimate": out["se"], "CI Lower": out["lower"],
                            "CI Upper": out["upper"]}, index=pd.Index(out["names"], name="Parameter")),
              title="Parameter Estimates")
    res.table(pd.DataFrame([{"DF": out["df"], "SSE": out["sse"], "MSE": out["sse"] / out["df"], "S": out["s"]}]),
              title="Summary", index=False)
    grid = np.linspace(x.min(), x.max(), 200)
    fig = new_figure()
    ax = fig.add_subplot()
    ax.plot(x, y, "o", color=BLUE, markersize=5)
    ax.plot(grid, out["func"](grid, *out["params"]), color=RED)
    ax.set_xlabel(predictor), ax.set_ylabel(response), ax.set_title(tr("Fitted Line Plot"))
    return res.figure(fig)


# ----- generalized linear models -----

def hosmer_lemeshow(y, fitted, groups=10) -> tuple[float, int, float]:
    order = np.argsort(fitted)
    chi2, used = 0.0, 0
    for idx in np.array_split(order, groups):
        if not len(idx):
            continue
        expected, observed, n = fitted[idx].sum(), y[idx].sum(), len(idx)
        if 0 < expected < n:
            chi2 += (observed - expected) ** 2 / (expected * (1 - expected / n))
            used += 1
    df = max(used - 2, 1)
    return float(chi2), df, float(sps.chi2.sf(chi2, df))


def glm_report(res: Result, fit, ratio_label: str, confidence=95.0):
    z = sps.norm.ppf(0.5 + confidence / 200)
    table = pd.DataFrame({"Coef": fit.params, "SE Coef": fit.bse, "Z-Value": fit.tvalues, "P-Value": fit.pvalues,
                          ratio_label: np.exp(fit.params), "CI Lower": np.exp(fit.params - z * fit.bse),
                          "CI Upper": np.exp(fit.params + z * fit.bse)})
    table.loc[CONSTANT, [ratio_label, "CI Lower", "CI Upper"]] = np.nan
    table.index.name = "Term"
    res.table(table, title="Coefficients")
    df_model = int(fit.df_model)
    regression = fit.null_deviance - fit.deviance
    deviance = pd.DataFrame({"DF": [df_model, fit.df_resid, df_model + fit.df_resid],
                             "Deviance": [regression, fit.deviance, fit.null_deviance],
                             "Chi-Square": [regression, np.nan, np.nan],
                             "P-Value": [sps.chi2.sf(regression, df_model), np.nan, np.nan]},
                            index=pd.Index(["Regression", "Error", "Total"], name="Source"))
    res.table(deviance, title="Deviance Table")
    res.table(pd.DataFrame([{"Deviance R-sq": 100 * (1 - fit.deviance / fit.null_deviance), "AIC": fit.aic,
                             "Log-likelihood": fit.llf}]), title="Model Summary", index=False)


@analysis("reg.logistic", menu=MENU, title="Binary Logistic Regression",
          params=[Param("response", "Response (two values)", "column", numeric=False), CONTINUOUS, CATEGORICAL,
                  INTERACTIONS, CONFIDENCE, Param("store", "Store event probabilities", "bool", default=False)])
def binary_logistic(ws, response, continuous=None, categorical=None, interactions=False, confidence=95.0,
                    store=False):
    design = build_design(ws, [response], continuous, categorical, interactions)
    labels = ws.df[response].iloc[design.rows].map(format_value)
    levels = ws.levels(response, labels)
    if len(levels) != 2:
        raise ValueError("The response must have exactly two distinct values")
    y = (labels == levels[-1]).astype(float).to_numpy()
    fit = sm.GLM(y, design.X, family=sm.families.Binomial()).fit()
    res = Result("Binary Logistic Regression")
    res.text(f"{tr('Event')}: {response} = {levels[-1]}    {tr('Events')}: {int(y.sum())}    N = {len(y)}")
    report_reference(res, design)
    glm_report(res, fit, "Odds Ratio", confidence)
    chi2, df, p = hosmer_lemeshow(y, np.asarray(fit.fittedvalues))
    res.table(pd.DataFrame([{"DF": df, "Chi-Square": chi2, "P-Value": p}],
                           index=pd.Index(["Hosmer-Lemeshow"], name="Test")), title="Goodness-of-Fit Test")
    if store:
        res.store(f"EPRO_{response}", design.aligned(fit.fittedvalues))
    return res


@analysis("reg.poisson", menu=MENU, title="Poisson Regression",
          params=[Param("response", "Response (counts)", "column"), CONTINUOUS, CATEGORICAL, INTERACTIONS,
                  CONFIDENCE])
def poisson_regression(ws, response, continuous=None, categorical=None, interactions=False, confidence=95.0):
    design = build_design(ws, [response], continuous, categorical, interactions)
    y = ws.df[response].iloc[design.rows].astype(float).to_numpy()
    if (y < 0).any():
        raise ValueError("Counts must be non-negative")
    fit = sm.GLM(y, design.X, family=sm.families.Poisson()).fit()
    res = Result("Poisson Regression")
    report_reference(res, design)
    glm_report(res, fit, "Rate Ratio", confidence)
    pearson = float(fit.pearson_chi2)
    res.table(pd.DataFrame({"DF": [fit.df_resid, fit.df_resid], "Chi-Square": [fit.deviance, pearson],
                            "P-Value": [sps.chi2.sf(fit.deviance, fit.df_resid),
                                        sps.chi2.sf(pearson, fit.df_resid)]},
                           index=pd.Index(["Deviance", "Pearson"], name="Test")), title="Goodness-of-Fit Test")
    return res


def _category_codes(ws, response, rows):
    labels = ws.df[response].iloc[rows].map(format_value)
    levels = ws.levels(response, labels)
    if response not in ws.value_orders:   # a text column of numbers keeps its numeric order
        try:
            levels = sorted(levels, key=float)
        except ValueError:
            pass
    if len(levels) < 3:
        raise ValueError("The response needs at least three distinct values")
    return labels.map({lv: i for i, lv in enumerate(levels)}).to_numpy(), levels


@analysis("reg.ordinal", menu=MENU, title="Ordinal Logistic Regression",
          params=[Param("response", "Response (ordered categories)", "column", numeric=False), CONTINUOUS,
                  CATEGORICAL])
def ordinal_logistic(ws, response, continuous=None, categorical=None):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    design = build_design(ws, [response], continuous, categorical)
    codes, levels = _category_codes(ws, response, design.rows)
    exog = design.X.drop(columns=CONSTANT)
    fit = OrderedModel(codes, exog, distr="logit").fit(method="bfgs", disp=0, maxiter=500)
    k = exog.shape[1]
    counts = np.bincount(codes)
    ll_null = float((counts * np.log(counts / counts.sum())).sum())
    g = 2 * (fit.llf - ll_null)
    res = Result("Ordinal Logistic Regression")
    res.text(f"{tr('Response order')}: {' < '.join(levels)}    N = {len(codes)}")
    report_reference(res, design)
    coefs = pd.DataFrame({"Coef": fit.params[:k], "SE Coef": fit.bse[:k], "Z-Value": fit.tvalues[:k],
                          "P-Value": fit.pvalues[:k], "Odds Ratio": np.exp(fit.params[:k])})
    coefs.index.name = "Term"
    res.table(coefs, title="Coefficients")
    cuts = fit.model.transform_threshold_params(np.asarray(fit.params))[1:-1]
    res.table(pd.DataFrame({"Threshold": cuts}, index=pd.Index(
        [f"{a} | {b}" for a, b in zip(levels[:-1], levels[1:])], name="Cut point")), title="Thresholds")
    return res.table(pd.DataFrame([{"DF": k, "G": g, "P-Value": float(sps.chi2.sf(g, k)),
                                    "Log-likelihood": fit.llf}]),
                     title="Test of All Slopes Equal to Zero", index=False)


@analysis("reg.nominal", menu=MENU, title="Nominal Logistic Regression",
          params=[Param("response", "Response (categories)", "column", numeric=False), CONTINUOUS, CATEGORICAL])
def nominal_logistic(ws, response, continuous=None, categorical=None):
    design = build_design(ws, [response], continuous, categorical)
    codes, levels = _category_codes(ws, response, design.rows)
    fit = sm.MNLogit(codes, design.X).fit(disp=0, maxiter=200)
    res = Result("Nominal Logistic Regression")
    res.text(f"{tr('Reference category')}: {levels[0]}    N = {len(codes)}")
    report_reference(res, design)
    params, bse = np.asarray(fit.params), np.asarray(fit.bse)
    z, p = np.asarray(fit.tvalues), np.asarray(fit.pvalues)
    for j, level in enumerate(levels[1:]):
        table = pd.DataFrame({"Coef": params[:, j], "SE Coef": bse[:, j], "Z-Value": z[:, j], "P-Value": p[:, j],
                              "Odds Ratio": np.exp(params[:, j])}, index=pd.Index(design.X.columns, name="Term"))
        res.heading(f"Logit: {level} / {levels[0]}").table(table)
    return res.table(pd.DataFrame([{"DF": int(fit.df_model), "G": float(fit.llr), "P-Value": float(fit.llr_pvalue),
                                    "Log-likelihood": float(fit.llf)}]),
                     title="Test of All Slopes Equal to Zero", index=False)


# ----- PLS and orthogonal -----

@analysis("reg.pls", menu=MENU, title="Partial Least Squares",
          params=[Param("responses", "Responses", "columns"), Param("predictors", "Predictors", "columns"),
                  Param("components", "Number of components", "int", default=2, minimum=1, maximum=50)])
def partial_least_squares(ws, responses, predictors, components=2):
    from sklearn.cross_decomposition import PLSRegression

    predictors = [p for p in predictors if p not in responses]
    if not predictors:
        raise ValueError("Select at least one predictor")
    frame = ws.df[list(responses) + predictors].dropna().astype(float)
    X, Y = frame[predictors].to_numpy(), frame[list(responses)].to_numpy()
    limit = min(len(predictors), len(frame) - 1)
    components = min(components, limit)
    if components < 1:
        raise ValueError("Not enough observations to fit the model")
    total = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    selection = {}
    for m in range(1, components + 1):
        model = PLSRegression(n_components=m, scale=True).fit(X, Y)
        resid = Y - model.predict(X).reshape(Y.shape)
        selection[m] = dict(zip(responses, 100 * (1 - (resid ** 2).sum(axis=0) / total)))
    coef = np.asarray(model.coef_).reshape(len(responses), len(predictors))
    table = pd.DataFrame(coef.T, index=pd.Index(predictors, name="Term"), columns=list(responses))
    # the fitted surface passes through the means, which gives the constant in original units
    intercept = pd.DataFrame([Y.mean(axis=0) - X.mean(axis=0) @ coef.T], index=[CONSTANT], columns=list(responses))
    res = Result("Partial Least Squares")
    selection_table = pd.DataFrame.from_dict(selection, orient="index")
    selection_table.index.name = "Components"
    res.table(selection_table, title="R-sq by number of components (%)")
    table = pd.concat([intercept, table])
    table.index.name = "Term"
    return res.table(table, title=f"{tr('Coefficients')} ({components})")


def deming(x, y, ratio=1.0) -> dict:
    """Orthogonal (Deming) regression; `ratio` = error variance of Y / error variance of X."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("At least three complete rows are needed")
    sxx, syy = x.var(ddof=1), y.var(ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    if sxy == 0:
        raise ValueError("The variables are uncorrelated; the slope is undefined")
    slope = (syy - ratio * sxx + np.sqrt((syy - ratio * sxx) ** 2 + 4 * ratio * sxy ** 2)) / (2 * sxy)
    intercept = y.mean() - slope * x.mean()
    # Jackknife standard errors
    idx = np.arange(n)
    reps = np.array([_deming_point(x[idx != i], y[idx != i], ratio) for i in range(n)])
    pseudo = n * np.array([intercept, slope]) - (n - 1) * reps
    se = pseudo.std(axis=0, ddof=1) / np.sqrt(n)
    return {"intercept": float(intercept), "slope": float(slope), "se_intercept": float(se[0]),
            "se_slope": float(se[1]), "n": n}


def _deming_point(x, y, ratio):
    sxx, syy = x.var(ddof=1), y.var(ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    slope = (syy - ratio * sxx + np.sqrt((syy - ratio * sxx) ** 2 + 4 * ratio * sxy ** 2)) / (2 * sxy)
    return y.mean() - slope * x.mean(), slope


@analysis("reg.orthogonal", menu=MENU, title="Orthogonal Regression",
          params=[RESPONSE, Param("predictor", "Predictor", "column"),
                  Param("ratio", "Error variance ratio (Y/X)", "number", default=1.0, minimum=1e-9), CONFIDENCE])
def orthogonal(ws, response, predictor, ratio=1.0, confidence=95.0):
    rows = complete_rows(ws, [response, predictor])
    x = ws.df[predictor].iloc[rows].to_numpy(dtype=float)
    y = ws.df[response].iloc[rows].to_numpy(dtype=float)
    out = deming(x, y, ratio)
    t = sps.t.ppf(0.5 + confidence / 200, out["n"] - 2)
    est = np.array([out["intercept"], out["slope"]])
    se = np.array([out["se_intercept"], out["se_slope"]])
    res = Result("Orthogonal Regression")
    res.text(f"{response} = {est[0]:.5g} {'+' if est[1] >= 0 else '-'} {abs(est[1]):.5g} {predictor}"
             f"    {tr('Error variance ratio (Y/X)')}: {ratio:g}")
    res.table(pd.DataFrame({"Coef": est, "SE Coef": se, "CI Lower": est - t * se, "CI Upper": est + t * se},
                           index=pd.Index([CONSTANT, predictor], name="Term")), title="Coefficients")
    ols = np.polyfit(x, y, 1)
    line = np.array([x.min(), x.max()])
    fig = new_figure()
    ax = fig.add_subplot()
    ax.plot(x, y, "o", color=BLUE, markersize=5)
    ax.plot(line, est[0] + est[1] * line, color=RED, label=tr("Orthogonal"))
    ax.plot(line, ols[1] + ols[0] * line, color=GRAY, linestyle="--", label=tr("Least squares"))
    ax.set_xlabel(predictor), ax.set_ylabel(response), ax.legend()
    ax.set_title(tr("Orthogonal Regression"))
    return res.figure(fig)
