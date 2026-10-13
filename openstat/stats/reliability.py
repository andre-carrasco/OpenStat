"""Reliability and survival: distribution analysis with censoring, life regression, probit, growth and test plans."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats import _life
from openstat.stats._design import CONSTANT, build_design
from openstat.stats._tests import CONFIDENCE

MENU = ("Stat", "Reliability/Survival")
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"
PALETTE = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#7030a0", "#c00000", "#00b0f0", "#7f7f7f"]

TIME = Param("time", "Failure or censoring times", "column")
CENSOR = Param("censor", "Censoring column", "column", numeric=False, optional=True)
CENSOR_VALUE = Param("censor_value", "Value that marks a censored observation", "text", default="0")
DISTRIBUTION = Param("distribution", "Distribution", "choice", default="Weibull", choices=_life.NAMES)
PERCENTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def life_data(ws, time, censor=None, censor_value="0", extra=()) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows with a time, their times and failure indicator (True = failed)."""
    cols = [time] + ([censor] if censor else []) + list(extra)
    frame = ws.df[cols].dropna()
    t = frame[time].to_numpy(dtype=float)
    if censor:
        failed = (frame[censor].map(format_value) != format_value(_number_or_text(censor_value))).to_numpy()
    else:
        failed = np.ones(len(frame), dtype=bool)
    if not failed.any():
        raise ValueError("At least one failure (uncensored observation) is needed")
    return frame, t, failed


def _number_or_text(text):
    text = str(text).strip()
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return text


def parameter_table(fit: _life.LifeFit, confidence=95.0) -> pd.DataFrame:
    """Parameters in the usual form of each distribution, with normal-approximation intervals."""
    z = sps.norm.ppf(0.5 + confidence / 200)
    mu, se_mu = float(fit.beta[0]), float(fit.se_beta()[0])
    sigma, se_sigma = fit.sigma, fit.se_sigma()
    rows = {}

    def positive(estimate, se_log):
        return {"Estimate": estimate, "Standard error": estimate * se_log, "CI Lower": estimate * np.exp(-z * se_log),
                "CI Upper": estimate * np.exp(z * se_log)}

    def plain(estimate, se):
        return {"Estimate": estimate, "Standard error": se, "CI Lower": estimate - z * se, "CI Upper": estimate + z * se}

    se_log_sigma = se_sigma / sigma if sigma else 0.0
    if fit.name == "Weibull":
        rows["Shape"] = positive(1 / sigma, se_log_sigma)
        rows["Scale"] = positive(np.exp(mu), se_mu)
    elif fit.name == "Exponential":
        rows["Mean"] = positive(np.exp(mu), se_mu)
    else:
        rows["Location"] = plain(mu, se_mu)
        rows["Scale"] = positive(sigma, se_log_sigma)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Parameter"
    return table


def percentile_table(fit: _life.LifeFit, confidence=95.0) -> pd.DataFrame:
    rows = {p: dict(zip(("Percentile", "CI Lower", "CI Upper"), fit.quantile(p / 100, confidence=confidence)))
            for p in PERCENTS}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Percent"
    return table


def probability_plot(ax, fit: _life.LifeFit, t, failed, title: str) -> float:
    """Probability plot on the fitted family's scale; returns the plot correlation coefficient."""
    times, probs = _life.plotting_positions(t, failed)
    x = np.log(times) if fit.log_time else times
    y = fit.base.ppf(probs)
    ax.plot(x, y, "o", color=BLUE, markersize=4)
    line = np.array([x.min(), x.max()])
    ax.plot(line, (line - fit.beta[0]) / fit.sigma, color=RED)
    ticks = np.array([1, 5, 10, 30, 50, 70, 90, 99])
    ax.set_yticks(fit.base.ppf(ticks / 100), [str(v) for v in ticks])
    if fit.log_time:
        marks = np.linspace(x.min(), x.max(), 5)
        ax.set_xticks(marks, [f"{m:.3g}" for m in np.exp(marks)])
    ax.set_title(title, fontsize=9), ax.set_ylabel(tr("Percent"))
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


@analysis("rel.parametric", menu=MENU, title="Parametric Distribution Analysis (Right Censoring)",
          params=[TIME, CENSOR, CENSOR_VALUE, DISTRIBUTION, CONFIDENCE])
def parametric_analysis(ws, time, censor=None, censor_value="0", distribution="Weibull", confidence=95.0):
    _, t, failed = life_data(ws, time, censor, censor_value)
    fit = _life.fit_life(*_life.right_censored(t, failed), name=distribution)
    res = Result("Parametric Distribution Analysis (Right Censoring)")
    res.text(f"{tr('Distribution')}: {tr(distribution)}    {tr('Failures')}: {int(failed.sum())}    "
             f"{tr('Censored')}: {int((~failed).sum())}    {tr('Estimation method: maximum likelihood')}")
    res.table(parameter_table(fit, confidence), title="Parameter Estimates")
    res.table(pd.DataFrame([{"Log-likelihood": fit.loglik, "AIC": fit.aic, "Mean (MTTF)": fit.mean(),
                             "Median": fit.quantile(0.5)[0]}]), title="Characteristics of Distribution", index=False)
    res.table(percentile_table(fit, confidence), title="Table of Percentiles")
    fig = new_figure(10.0, 4.2)
    left, right = fig.subplots(1, 2)
    corr = probability_plot(left, fit, t, failed, f"{tr('Probability Plot')} ({tr(distribution)})")
    left.set_xlabel(time)
    grid = np.linspace(max(t.min() * 0.5, 1e-9) if fit.log_time else t.min(), t.max() * 1.1, 300)
    right.plot(grid, 100 * fit.survival(grid), color=RED)
    km = _life.kaplan_meier(t, failed)
    right.step(np.concatenate([[grid[0]], km["Time"]]), np.concatenate([[100], 100 * km["Survival probability"]]),
               where="post", color=BLUE, linewidth=1)
    right.set_xlabel(time), right.set_ylabel(tr("Percent")), right.set_title(tr("Survival Plot"), fontsize=9)
    res.figure(fig)
    res.text(f"{tr('Probability plot correlation')} = {corr:.4f}")
    return res


@analysis("rel.arbitrary", menu=MENU, title="Parametric Distribution Analysis (Arbitrary Censoring)",
          params=[Param("start", "Start times (blank = left censored)", "column"),
                  Param("end", "End times (blank = right censored)", "column"),
                  Param("frequency", "Frequencies", "column", optional=True), DISTRIBUTION, CONFIDENCE])
def arbitrary_censoring(ws, start, end, frequency=None, distribution="Weibull", confidence=95.0):
    cols = [start, end] + ([frequency] if frequency else [])
    frame = ws.df[cols]
    frame = frame[frame[[start, end]].notna().any(axis=1)]
    if frequency:
        frame = frame[frame[frequency].notna() & (frame[frequency] > 0)]
        frame = frame.loc[frame.index.repeat(frame[frequency].round().astype(int))]
    lower = frame[start].fillna(0.0 if _life.FAMILIES[distribution][1] else -np.inf).to_numpy(dtype=float)
    upper = frame[end].fillna(np.inf).to_numpy(dtype=float)
    if (upper < lower).any():
        raise ValueError("An end time is earlier than its start time")
    fit = _life.fit_life(lower, upper, name=distribution)
    exact = np.isclose(lower, upper)
    res = Result("Parametric Distribution Analysis (Arbitrary Censoring)")
    res.table(pd.DataFrame([{"Exact": int(exact.sum()), "Right censored": int(np.isinf(upper).sum()),
                             "Left censored": int((~np.isfinite(frame[start].to_numpy(dtype=float))).sum()),
                             "Interval censored": int((~exact & np.isfinite(upper)
                                                       & frame[start].notna().to_numpy()).sum())}]),
              title="Censoring Information", index=False)
    res.table(parameter_table(fit, confidence), title="Parameter Estimates")
    res.table(pd.DataFrame([{"Log-likelihood": fit.loglik, "AIC": fit.aic, "Mean (MTTF)": fit.mean(),
                             "Median": fit.quantile(0.5)[0]}]), title="Characteristics of Distribution", index=False)
    return res.table(percentile_table(fit, confidence), title="Table of Percentiles")


@analysis("rel.identify", menu=MENU, title="Distribution ID Plot (Right Censoring)",
          params=[TIME, CENSOR, CENSOR_VALUE])
def distribution_id(ws, time, censor=None, censor_value="0"):
    _, t, failed = life_data(ws, time, censor, censor_value)
    fits = []
    for name in _life.NAMES:
        try:
            fits.append(_life.fit_life(*_life.right_censored(t, failed), name=name))
        except ValueError:
            continue
    fits.sort(key=lambda f: f.aic)
    fig = new_figure(9.0, 6.5)
    axes = fig.subplots(2, 2).ravel()
    correlations = {}
    for ax, fit in zip(axes, fits[:4]):
        correlations[fit.name] = probability_plot(ax, fit, t, failed, tr(fit.name))
    for ax in axes[len(fits[:4]):]:
        ax.set_axis_off()
    rows = {tr(f.name): {"Log-likelihood": f.loglik, "AIC": f.aic, "Median": f.quantile(0.5)[0],
                         "Mean (MTTF)": f.mean()} for f in fits}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Distribution"
    res = Result("Distribution ID Plot (Right Censoring)")
    res.text(tr("Distributions are ordered from best to worst fit (smallest AIC)."))
    return res.table(table, title="Goodness-of-Fit").figure(fig)


@analysis("rel.kaplan_meier", menu=MENU, title="Nonparametric Distribution Analysis (Kaplan-Meier)",
          params=[TIME, CENSOR, CENSOR_VALUE, Param("by", "By variable", "column", numeric=False, optional=True),
                  CONFIDENCE])
def nonparametric_analysis(ws, time, censor=None, censor_value="0", by=None, confidence=95.0):
    frame, t, failed = life_data(ws, time, censor, censor_value, [by] if by else [])
    labels = frame[by].map(format_value).to_numpy() if by else np.full(len(t), time, dtype=object)
    res = Result("Nonparametric Distribution Analysis (Kaplan-Meier)")
    fig = new_figure()
    ax = fig.add_subplot()
    summary = {}
    for i, level in enumerate(sorted(set(labels))):
        mask = labels == level
        if not failed[mask].any():
            continue
        table = _life.kaplan_meier(t[mask], failed[mask], confidence)
        summary[level] = {"N": int(mask.sum()), "Failures": int(failed[mask].sum()),
                          "Censored": int((~failed[mask]).sum()), "Median": _life.km_median(table)}
        res.table(table.head(100), title=f"{tr('Kaplan-Meier Estimates')}: {level}", index=False)
        ax.step(np.concatenate([[0], table["Time"]]), np.concatenate([[100], 100 * table["Survival probability"]]),
                where="post", color=PALETTE[i % len(PALETTE)], label=level)
    ax.set_xlabel(time), ax.set_ylabel(tr("Percent")), ax.set_ylim(0, 105)
    ax.set_title(tr("Survival Plot"))
    if by:
        ax.legend(title=by)
    table = pd.DataFrame.from_dict(summary, orient="index")
    table.index.name = by or "Variable"
    res.blocks.insert(0, ("table", (table, "Censoring Information", 4, True)))
    if by and len(summary) > 1:
        from lifelines.statistics import multivariate_logrank_test

        test = multivariate_logrank_test(t, labels, failed)
        res.table(pd.DataFrame([{"Chi-Square": float(test.test_statistic), "DF": len(summary) - 1,
                                 "P-Value": float(test.p_value)}], index=pd.Index(["Log-Rank"], name="Method")),
                  title="Test Statistics")
    return res.figure(fig)


# ----- regression with life data -----

def _life_regression(res: Result, ws, time, failed, design, distribution, confidence):
    t = ws.df[time].iloc[design.rows].to_numpy(dtype=float)
    fit = _life.fit_life(*_life.right_censored(t, failed), X=design.X.to_numpy(), name=distribution,
                         columns=list(design.X.columns))
    z = sps.norm.ppf(0.5 + confidence / 200)
    se = fit.se_beta()
    table = pd.DataFrame({"Coef": fit.beta, "Standard error": se, "Z-Value": fit.beta / se,
                          "P-Value": 2 * sps.norm.sf(np.abs(fit.beta / se)), "CI Lower": fit.beta - z * se,
                          "CI Upper": fit.beta + z * se}, index=pd.Index(fit.columns, name="Predictor"))
    res.table(table, title="Regression Table")
    if not fit.fixed_scale:
        label = "Shape" if distribution == "Weibull" else "Scale"
        value = 1 / fit.sigma if distribution == "Weibull" else fit.sigma
        res.table(pd.DataFrame([{"Estimate": value, "Standard error": value * fit.se_sigma() / fit.sigma}],
                               index=pd.Index([label], name="Parameter")))
    res.text(f"{tr('Log-likelihood')} = {fit.loglik:.4f}    AIC = {fit.aic:.4f}")
    return fit


@analysis("rel.regression", menu=MENU, title="Regression with Life Data",
          params=[TIME, CENSOR, CENSOR_VALUE,
                  Param("continuous", "Continuous predictors", "columns", optional=True),
                  Param("categorical", "Categorical predictors", "columns", numeric=False, optional=True),
                  DISTRIBUTION, CONFIDENCE])
def life_regression(ws, time, censor=None, censor_value="0", continuous=None, categorical=None,
                    distribution="Weibull", confidence=95.0):
    categorical = [c for c in (categorical or []) if c != censor]
    design = build_design(ws, [time], continuous, categorical, extra_columns=[censor] if censor else [])
    failed = np.ones(len(design.rows), dtype=bool)
    if censor:
        marks = ws.df[censor].iloc[design.rows].map(format_value)
        failed = (marks != format_value(_number_or_text(censor_value))).to_numpy()
    res = Result("Regression with Life Data")
    res.text(f"{tr('Distribution')}: {tr(distribution)}    {tr('Failures')}: {int(failed.sum())}    "
             f"{tr('Censored')}: {int((~failed).sum())}")
    _life_regression(res, ws, time, failed, design, distribution, confidence)
    return res


TIES = ["Efron", "Breslow"]


@analysis("rel.cox", menu=MENU, title="Cox Regression",
          params=[TIME, CENSOR, CENSOR_VALUE,
                  Param("continuous", "Continuous predictors", "columns", optional=True),
                  Param("categorical", "Categorical predictors", "columns", numeric=False, optional=True),
                  Param("ties", "Method for tied times", "choice", default=TIES[0], choices=TIES), CONFIDENCE])
def cox_regression(ws, time, censor=None, censor_value="0", continuous=None, categorical=None, ties=TIES[0],
                   confidence=95.0):
    from lifelines.utils import concordance_index

    categorical = [c for c in (categorical or []) if c != censor]
    design = build_design(ws, [time], continuous, categorical, extra_columns=[censor] if censor else [])
    failed = np.ones(len(design.rows), dtype=bool)
    if censor:
        marks = ws.df[censor].iloc[design.rows].map(format_value)
        failed = (marks != format_value(_number_or_text(censor_value))).to_numpy()
    if not failed.any():
        raise ValueError("At least one failure (uncensored observation) is needed")
    t = ws.df[time].iloc[design.rows].to_numpy(dtype=float)
    exog = design.X.drop(columns=CONSTANT)   # the baseline hazard takes the place of the constant
    model = sm.PHReg(t, exog, status=failed.astype(float), ties=ties.lower())
    fit = model.fit()
    k = exog.shape[1]
    z = sps.norm.ppf(0.5 + confidence / 200)
    params, se = np.asarray(fit.params), np.asarray(fit.bse)
    res = Result("Cox Regression")
    res.text(f"{tr('Failures')}: {int(failed.sum())}    {tr('Censored')}: {int((~failed).sum())}    "
             f"{tr('Method for tied times')}: {ties}")
    res.table(pd.DataFrame({"Coef": params, "SE Coef": se, "Z-Value": params / se,
                            "P-Value": 2 * sps.norm.sf(np.abs(params / se)), "Hazard ratio": np.exp(params),
                            "CI Lower": np.exp(params - z * se), "CI Upper": np.exp(params + z * se)},
                           index=pd.Index(exog.columns, name="Term")), title="Coefficients")
    g = 2 * (float(fit.llf) - float(model.loglike(np.zeros(k))))
    res.table(pd.DataFrame([{"DF": k, "Chi-Square": g, "P-Value": float(sps.chi2.sf(g, k)),
                             "Log-likelihood": float(fit.llf),
                             "Concordance": concordance_index(t, -(exog.to_numpy() @ params), failed)}]),
              title="Likelihood ratio test of the model", index=False)
    return res


RELATIONSHIPS = ["Arrhenius (temperature in °C)", "Inverse temperature (°C)", "Ln (power)", "Linear"]


def transform_stress(x, relationship: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if relationship == RELATIONSHIPS[0]:
        return 11604.83 / (x + 273.16)
    if relationship == RELATIONSHIPS[1]:
        return 1 / (x + 273.16)
    if relationship == RELATIONSHIPS[2]:
        if (x <= 0).any():
            raise ValueError("The accelerating variable must be positive for this relationship")
        return np.log(x)
    return x


@analysis("rel.accelerated", menu=MENU, title="Accelerated Life Testing",
          params=[TIME, CENSOR, CENSOR_VALUE, Param("stress", "Accelerating variable", "column"),
                  Param("relationship", "Relationship", "choice", default=RELATIONSHIPS[0], choices=RELATIONSHIPS),
                  Param("design_stress", "Design (use) value of the accelerating variable", "number", default=0.0),
                  DISTRIBUTION, CONFIDENCE])
def accelerated_life(ws, time, censor=None, censor_value="0", stress=None, relationship=RELATIONSHIPS[0],
                     design_stress=0.0, distribution="Weibull", confidence=95.0):
    frame, t, failed = life_data(ws, time, censor, censor_value, [stress])
    raw = frame[stress].to_numpy(dtype=float)
    x = transform_stress(raw, relationship)
    if len(np.unique(x)) < 2:
        raise ValueError("At least two levels of the accelerating variable are needed")
    X = np.column_stack([np.ones(len(x)), x])
    fit = _life.fit_life(*_life.right_censored(t, failed), X=X, name=distribution, columns=["Intercept", stress])
    z = sps.norm.ppf(0.5 + confidence / 200)
    se = fit.se_beta()
    res = Result("Accelerated Life Testing")
    res.text(f"{tr('Distribution')}: {tr(distribution)}    {tr('Relationship')}: {tr(relationship)}")
    res.table(pd.DataFrame({"Coef": fit.beta, "Standard error": se, "Z-Value": fit.beta / se,
                            "P-Value": 2 * sps.norm.sf(np.abs(fit.beta / se)), "CI Lower": fit.beta - z * se,
                            "CI Upper": fit.beta + z * se}, index=pd.Index(fit.columns, name="Predictor")),
              title="Regression Table")
    levels = sorted(set(raw.tolist()) | {float(design_stress)})
    rows = {}
    for level in levels:
        row = np.array([1.0, float(transform_stress([level], relationship)[0])])
        rows[f"{level:g}"] = {f"{int(100 * p)}%": fit.quantile(p, row)[0] for p in (0.1, 0.5, 0.9)}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = stress
    res.table(table, title="Table of Percentiles")
    fig = new_figure()
    ax = fig.add_subplot()
    ax.semilogy(raw[failed], t[failed], "o", color=BLUE, markersize=4, label=tr("Failures"))
    if (~failed).any():
        ax.semilogy(raw[~failed], t[~failed], "^", color=GRAY, markersize=4, label=tr("Censored"))
    grid = np.linspace(min(levels), max(levels), 100)
    for p, style in ((0.1, ":"), (0.5, "-"), (0.9, "--")):
        curve = [fit.quantile(p, np.array([1.0, float(transform_stress([g], relationship)[0])]))[0] for g in grid]
        ax.semilogy(grid, curve, color=RED, linestyle=style, label=f"{int(100 * p)}%")
    ax.set_xlabel(stress), ax.set_ylabel(time), ax.legend(fontsize=8)
    ax.set_title(tr("Relation Plot"))
    return res.figure(fig)


LINKS = ["Normal (probit)", "Logistic (logit)"]


@analysis("rel.probit", menu=MENU, title="Probit Analysis",
          params=[Param("events", "Number of events", "column"), Param("trials", "Number of trials", "column"),
                  Param("stress", "Stress (stimulus)", "column"),
                  Param("link", "Assumed distribution", "choice", default=LINKS[0], choices=LINKS),
                  Param("log_stress", "Use the logarithm of the stress", "bool", default=False), CONFIDENCE])
def probit(ws, events, trials, stress, link=LINKS[0], log_stress=False, confidence=95.0):
    frame = ws.df[[events, trials, stress]].dropna().astype(float)
    k, n, x = frame[events].to_numpy(), frame[trials].to_numpy(), frame[stress].to_numpy()
    if (k < 0).any() or (k > n).any() or (n <= 0).any():
        raise ValueError("Events must be between 0 and the number of trials")
    if log_stress:
        if (x <= 0).any():
            raise ValueError("The stress must be positive to take its logarithm")
        x = np.log(x)
    family = sm.families.Binomial(sm.families.links.Probit() if link == LINKS[0] else sm.families.links.Logit())
    X = pd.DataFrame({"Constant": 1.0, stress: x})
    fit = sm.GLM(np.column_stack([k, n - k]), X, family=family).fit()
    dist = sps.norm if link == LINKS[0] else sps.logistic
    b0, b1 = fit.params["Constant"], fit.params[stress]
    if b1 == 0:
        raise ValueError("The stress has no effect; percentiles are undefined")
    cov = fit.cov_params().to_numpy()
    z = sps.norm.ppf(0.5 + confidence / 200)
    rows = {}
    for p in PERCENTS:
        q = dist.ppf(p / 100)
        estimate = (q - b0) / b1
        gradient = np.array([-1 / b1, -(q - b0) / b1 ** 2])
        se = float(np.sqrt(gradient @ cov @ gradient))
        values = np.array([estimate, estimate - z * se, estimate + z * se])
        rows[p] = dict(zip(("Percentile", "CI Lower", "CI Upper"), np.exp(values) if log_stress else values))
    res = Result("Probit Analysis")
    res.text(f"{tr('Assumed distribution')}: {tr(link)}" + (f"    ln({stress})" if log_stress else ""))
    res.table(pd.DataFrame({"Coef": fit.params, "Standard error": fit.bse, "Z-Value": fit.tvalues,
                            "P-Value": fit.pvalues}), title="Regression Table")
    pearson = float(fit.pearson_chi2)
    res.table(pd.DataFrame({"Chi-Square": [pearson, fit.deviance], "DF": [fit.df_resid, fit.df_resid],
                            "P-Value": [sps.chi2.sf(pearson, fit.df_resid), sps.chi2.sf(fit.deviance, fit.df_resid)]},
                           index=pd.Index(["Pearson", "Deviance"], name="Method")), title="Goodness-of-Fit Tests")
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Percent"
    res.table(table, title="Table of Percentiles")
    fig = new_figure()
    ax = fig.add_subplot()
    shown = np.exp(x) if log_stress else x
    ax.plot(shown, 100 * k / n, "o", color=BLUE)
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(np.exp(grid) if log_stress else grid, 100 * dist.cdf(b0 + b1 * grid), color=RED)
    ax.set_xlabel(stress), ax.set_ylabel(tr("Percent")), ax.set_title(tr("Probit Analysis"))
    return res.figure(fig)


# ----- repairable systems and test plans -----

def power_law_process(times, end=None) -> dict:
    """Power-law (Crow-AMSAA) NHPP for one system; `end` given means a time-truncated test."""
    t = np.sort(np.asarray(times, dtype=float))
    n = len(t)
    if n < 3 or (t <= 0).any():
        raise ValueError("At least three positive failure times are needed")
    time_truncated = end is not None and end > t[-1]
    total = float(end) if time_truncated else float(t[-1])
    logs = np.log(total / t)
    shape = n / logs.sum()
    scale = total / n ** (1 / shape)
    # MIL-HDBK-189 trend test: 2 * sum(ln(T / t_i)) is chi-square under a constant failure rate
    df = 2 * n if time_truncated else 2 * (n - 1)
    statistic = 2 * logs.sum()
    p = 2 * min(sps.chi2.cdf(statistic, df), sps.chi2.sf(statistic, df))
    m = n if time_truncated else n - 1
    laplace = (t[:m].sum() / m - total / 2) / (total * np.sqrt(1 / (12 * m)))
    return {"n": n, "total": total, "shape": float(shape), "scale": float(scale), "mil_statistic": float(statistic),
            "mil_df": df, "mil_p": float(min(p, 1.0)), "laplace": float(laplace),
            "laplace_p": float(2 * sps.norm.sf(abs(laplace))),
            "mtbf_end": float(total / (n * shape))}


@analysis("rel.growth", menu=MENU, title="Parametric Growth Curve (Repairable System)",
          params=[Param("time", "Cumulative failure times", "column"),
                  Param("end", "End of test time (0 = last failure)", "number", default=0.0, minimum=0.0)])
def growth_curve(ws, time, end=0.0):
    t = np.sort(ws.df[time].dropna().to_numpy(dtype=float))
    out = power_law_process(t, end or None)
    res = Result("Parametric Growth Curve (Repairable System)")
    res.text(tr("Power-law process: expected failures by time t = (t / scale) ^ shape"))
    res.table(pd.DataFrame([{"Failures": out["n"], "Shape": out["shape"], "Scale": out["scale"],
                             "MTBF at end of test": out["mtbf_end"]}]), title="Parameter Estimates", index=False)
    res.table(pd.DataFrame({"Statistic": [out["mil_statistic"], out["laplace"]],
                            "P-Value": [out["mil_p"], out["laplace_p"]]},
                           index=pd.Index(["MIL-Hdbk-189", "Laplace"], name="Trend test")), title="Trend Tests")
    trend = "improving (failures become less frequent)" if out["shape"] < 1 else \
        "deteriorating (failures become more frequent)"
    res.text(f"{tr('Shape')} {'<' if out['shape'] < 1 else '>'} 1: {tr(trend)}")
    fig = new_figure()
    ax = fig.add_subplot()
    ax.step(np.concatenate([[0], t]), np.arange(len(t) + 1), where="post", color=BLUE)
    grid = np.linspace(0, out["total"], 200)
    ax.plot(grid, (grid / out["scale"]) ** out["shape"], color=RED)
    ax.set_xlabel(time), ax.set_ylabel(tr("Cumulative failures")), ax.set_title(tr("Mean Cumulative Function"))
    return res.figure(fig)


def demonstration_sample_size(reliability, mission_time, test_time, shape=1.0, confidence=0.95, failures=0) -> int:
    """Units needed to demonstrate `reliability` at `mission_time`, testing each for `test_time` (Weibull)."""
    if not 0 < reliability < 1 or mission_time <= 0 or test_time <= 0 or shape <= 0:
        raise ValueError("Reliability must be between 0 and 1 and times must be positive")
    survive = reliability ** ((test_time / mission_time) ** shape)   # reliability at the test time
    def enough(n):
        return sps.binom.cdf(failures, n, 1 - survive) <= 1 - confidence

    high = failures + 1
    while not enough(high):
        high *= 2
        if high > 10 ** 9:
            raise ValueError("No test plan found; increase the test time")
    low = failures + 1
    while low < high:   # the acceptance probability decreases with n
        mid = (low + high) // 2
        if enough(mid):
            high = mid
        else:
            low = mid + 1
    return low


@analysis("rel.demonstration", menu=MENU, title="Demonstration Test Plan",
          params=[Param("reliability", "Reliability to demonstrate (%)", "number", default=90.0, minimum=0.001,
                        maximum=99.9999),
                  Param("mission_time", "At time", "number", default=1000.0, minimum=1e-12),
                  Param("test_time", "Testing time for each unit", "number", default=1000.0, minimum=1e-12),
                  Param("shape", "Weibull shape", "number", default=1.0, minimum=0.01),
                  Param("failures", "Maximum number of failures allowed", "int", default=0, minimum=0, maximum=50),
                  CONFIDENCE])
def demonstration_plan(ws, reliability=90.0, mission_time=1000.0, test_time=1000.0, shape=1.0, failures=0,
                       confidence=95.0):
    n = demonstration_sample_size(reliability / 100, mission_time, test_time, shape, confidence / 100, failures)
    survive = (reliability / 100) ** ((test_time / mission_time) ** shape)
    res = Result("Demonstration Test Plan")
    res.table(pd.DataFrame([{"Sample size": n, "Testing time": test_time, "Failures allowed": failures,
                             "Actual confidence (%)": 100 * sps.binom.sf(failures, n, 1 - survive)}]),
              title="Test Plan", index=False, digits=2)
    res.text(f"{tr('If no more than')} {failures} {tr('units fail, the reliability at time')} {mission_time:g} "
             f"{tr('is at least')} {reliability:g}% ({confidence:g}% {tr('confidence')}).")
    return res
