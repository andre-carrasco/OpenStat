"""Analysis of variance: one-way, general linear model, nested, mixed, MANOVA and related plots."""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._design import CONSTANT, build_design, complete_rows
from openstat.stats._tests import CONFIDENCE
from openstat.stats.regression import PLOTS, STORE, linear_report, residual_plots

MENU = ("Stat", "ANOVA")
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"
PALETTE = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#7030a0", "#c00000", "#00b0f0", "#7f7f7f"]

RESPONSE = Param("response", "Response", "column")
FACTOR = Param("factor", "Factor", "column", numeric=False)
FACTORS = Param("factors", "Factors", "columns", numeric=False)


def factor_groups(ws, response, factor) -> dict[str, np.ndarray]:
    frame = ws.df[[response, factor]].dropna()
    labels = frame[factor].map(format_value)
    groups = {level: frame[response][labels == level].to_numpy(dtype=float) for level in sorted(labels.unique())}
    if len(groups) < 2:
        raise ValueError("The factor needs at least two levels")
    return groups


def one_way_table(groups: dict[str, np.ndarray]) -> dict:
    values = np.concatenate(list(groups.values()))
    n, k = len(values), len(groups)
    if n - k < 1:
        raise ValueError("Not enough observations for the error term")
    grand = values.mean()
    ss_factor = sum(len(g) * (g.mean() - grand) ** 2 for g in groups.values())
    ss_error = sum(((g - g.mean()) ** 2).sum() for g in groups.values())
    ms_factor, mse = ss_factor / (k - 1), ss_error / (n - k)
    f = ms_factor / mse if mse > 0 else np.nan
    return {"n": n, "k": k, "df_factor": k - 1, "df_error": n - k, "ss_factor": ss_factor, "ss_error": ss_error,
            "ms_factor": ms_factor, "mse": mse, "f": f, "p": float(sps.f.sf(f, k - 1, n - k))}


COMPARISONS = ["None", "Tukey", "Fisher", "Dunnett (first level is the control)"]


def pairwise_comparisons(groups: dict[str, np.ndarray], method: str, confidence=95.0) -> pd.DataFrame:
    """Simultaneous (Tukey) or individual (Fisher) intervals for all pairwise differences."""
    table = one_way_table(groups)
    mse, dfe, k = table["mse"], table["df_error"], table["k"]
    level = confidence / 100
    rows = {}
    for a, b in combinations(groups, 2):
        diff = groups[b].mean() - groups[a].mean()
        se = np.sqrt(mse * (1 / len(groups[a]) + 1 / len(groups[b])))
        if method == "Tukey":
            q = abs(diff) / (se / np.sqrt(2))
            p = float(sps.studentized_range.sf(q, k, dfe))
            half = sps.studentized_range.ppf(level, k, dfe) / np.sqrt(2) * se
        else:
            p = float(2 * sps.t.sf(abs(diff) / se, dfe))
            half = sps.t.ppf(0.5 + level / 2, dfe) * se
        rows[f"{b} − {a}"] = {"Difference": diff, "SE": se, "CI Lower": diff - half, "CI Upper": diff + half,
                              "T-Value": diff / se, "Adjusted P-Value" if method == "Tukey" else "P-Value": p}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "Difference of levels"
    return out


def dunnett_comparisons(groups: dict[str, np.ndarray], confidence=95.0) -> pd.DataFrame:
    names = list(groups)
    control, treatments = groups[names[0]], [groups[n] for n in names[1:]]
    test = sps.dunnett(*treatments, control=control)
    ci = test.confidence_interval(confidence_level=confidence / 100)
    out = pd.DataFrame({"Difference": [t.mean() - control.mean() for t in treatments],
                        "CI Lower": np.atleast_1d(ci.low), "CI Upper": np.atleast_1d(ci.high),
                        "T-Value": np.atleast_1d(test.statistic), "Adjusted P-Value": np.atleast_1d(test.pvalue)},
                       index=pd.Index([f"{n} − {names[0]}" for n in names[1:]], name="Difference of levels"))
    return out


def welch_anova(groups: dict[str, np.ndarray]) -> dict:
    k = len(groups)
    n = np.array([len(g) for g in groups.values()], dtype=float)
    means = np.array([g.mean() for g in groups.values()])
    variances = np.array([g.var(ddof=1) for g in groups.values()])
    if (n < 2).any() or (variances == 0).any():
        raise ValueError("Welch's test needs at least two observations and non-zero variance in each level")
    w = n / variances
    grand = (w * means).sum() / w.sum()
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f = ((w * (means - grand) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) * lam / (k ** 2 - 1))
    df2 = (k ** 2 - 1) / (3 * lam)
    return {"f": float(f), "df1": k - 1, "df2": float(df2), "p": float(sps.f.sf(f, k - 1, df2))}


@analysis("anova.one_way", menu=MENU, title="One-Way ANOVA",
          params=[RESPONSE, FACTOR,
                  Param("comparisons", "Comparisons", "choice", default="None", choices=COMPARISONS),
                  Param("welch", "Do not assume equal variances (Welch)", "bool", default=False),
                  CONFIDENCE, PLOTS, STORE])
def one_way(ws, response, factor, comparisons="None", welch=False, confidence=95.0, plots=True, store=False):
    groups = factor_groups(ws, response, factor)
    t = one_way_table(groups)
    res = Result("One-Way ANOVA")
    res.text(f"{tr('Factor')}: {factor}    {tr('Levels')}: {', '.join(groups)}")
    if welch:
        w = welch_anova(groups)
        res.table(pd.DataFrame([{"DF Num": w["df1"], "DF Den": w["df2"], "F-Value": w["f"], "P-Value": w["p"]}],
                               index=pd.Index([factor], name="Source")), title="Welch's Test")
    else:
        anova = pd.DataFrame({"DF": [t["df_factor"], t["df_error"], t["n"] - 1],
                              "Adj SS": [t["ss_factor"], t["ss_error"], t["ss_factor"] + t["ss_error"]],
                              "Adj MS": [t["ms_factor"], t["mse"], np.nan], "F-Value": [t["f"], np.nan, np.nan],
                              "P-Value": [t["p"], np.nan, np.nan]},
                             index=pd.Index([factor, "Error", "Total"], name="Source"))
        res.table(anova, title="Analysis of Variance")
        total = t["ss_factor"] + t["ss_error"]
        res.table(pd.DataFrame([{"S": np.sqrt(t["mse"]), "R-sq": 100 * t["ss_factor"] / total,
                                 "R-sq(adj)": 100 * (1 - t["mse"] / (total / (t["n"] - 1)))}]),
                  title="Model Summary", index=False)
    crit = sps.t.ppf(0.5 + confidence / 200, t["df_error"])
    pooled = np.sqrt(t["mse"])
    means = pd.DataFrame({"N": [len(g) for g in groups.values()], "Mean": [g.mean() for g in groups.values()],
                          "StDev": [g.std(ddof=1) if len(g) > 1 else np.nan for g in groups.values()]},
                         index=pd.Index(list(groups), name=factor))
    means["CI Lower"] = means["Mean"] - crit * pooled / np.sqrt(means["N"])
    means["CI Upper"] = means["Mean"] + crit * pooled / np.sqrt(means["N"])
    res.table(means, title="Means")
    res.text(f"{tr('Pooled StDev')} = {pooled:.5g}")
    if comparisons == "Tukey":
        res.table(pairwise_comparisons(groups, "Tukey", confidence), title="Tukey Pairwise Comparisons")
    elif comparisons == "Fisher":
        res.table(pairwise_comparisons(groups, "Fisher", confidence), title="Fisher Pairwise Comparisons")
    elif comparisons == COMPARISONS[3]:
        res.table(dunnett_comparisons(groups, confidence), title="Dunnett Comparisons with a Control")

    fits = np.concatenate([np.full(len(g), g.mean()) for g in groups.values()])
    resid = np.concatenate(list(groups.values())) - fits
    fig = new_figure(max(6.0, 1.0 + 0.9 * len(groups)), 4.0)
    ax = fig.add_subplot()
    for i, (lo, mean, hi) in enumerate(zip(means["CI Lower"], means["Mean"], means["CI Upper"])):
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color=BLUE, capsize=6, marker="o")
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel(factor), ax.set_ylabel(response)
    ax.set_title(f"{tr('Interval Plot')} ({confidence:g}% {tr('CI for the mean')}, {tr('Pooled StDev')})")
    res.figure(fig)
    if plots:
        res.figure(residual_plots(fits, resid, response))
    if store:
        rows = complete_rows(ws, [response, factor])
        order = np.argsort(ws.df[factor].iloc[rows].map(format_value).to_numpy(), kind="stable")
        fit_col, resid_col = np.full(ws.nrows, np.nan), np.full(ws.nrows, np.nan)
        fit_col[rows[order]], resid_col[rows[order]] = fits, resid
        res.store(f"FITS_{response}", fit_col.tolist()).store(f"RESI_{response}", resid_col.tolist())
    return res


@analysis("anova.glm", menu=MENU, title="General Linear Model",
          params=[RESPONSE, FACTORS, Param("covariates", "Covariates", "columns", optional=True),
                  Param("interactions", "Include two-way interactions", "bool", default=False), PLOTS, STORE])
def general_linear_model(ws, response, factors, covariates=None, interactions=False, plots=True, store=False):
    design = build_design(ws, [response], covariates, factors, interactions, coding="effect")
    res = Result("General Linear Model")
    info = pd.DataFrame({"Levels": [len(v) for v in design.levels.values()],
                         "Values": [", ".join(v) for v in design.levels.values()]},
                        index=pd.Index(list(design.levels), name="Factor"))
    res.table(info, title="Factor Information")
    linear_report(res, ws, response, design, store, plots, show_equation=False)
    frame = ws.df.iloc[design.rows]
    for factor in factors:
        grouped = frame.groupby(frame[factor].map(format_value))[response]
        means = pd.DataFrame({"N": grouped.count(), "Mean": grouped.mean(), "StDev": grouped.std(ddof=1)})
        means.index.name = factor
        res.table(means, title=f"{tr('Means')}: {factor}")
    return res


@analysis("anova.equal_variances", menu=MENU, title="Test for Equal Variances",
          params=[RESPONSE, FACTOR, CONFIDENCE])
def equal_variances(ws, response, factor, confidence=95.0):
    groups = factor_groups(ws, response, factor)
    if any(len(g) < 2 for g in groups.values()):
        raise ValueError("Each level needs at least two observations")
    k = len(groups)
    alpha = (1 - confidence / 100) / k  # Bonferroni
    rows = {}
    for level, g in groups.items():
        n, var = len(g), g.var(ddof=1)
        rows[level] = {"N": n, "StDev": np.sqrt(var),
                       "CI Lower": np.sqrt((n - 1) * var / sps.chi2.ppf(1 - alpha / 2, n - 1)),
                       "CI Upper": np.sqrt((n - 1) * var / sps.chi2.ppf(alpha / 2, n - 1))}
    bartlett = sps.bartlett(*groups.values())
    levene = sps.levene(*groups.values(), center="median")
    res = Result("Test for Equal Variances")
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = factor
    res.table(table, title=f"{confidence:g}% {tr('Bonferroni confidence intervals for standard deviations')}")
    return res.table(pd.DataFrame({"Statistic": [bartlett.statistic, levene.statistic],
                                   "P-Value": [bartlett.pvalue, levene.pvalue]},
                                  index=pd.Index(["Bartlett", "Levene"], name="Method")), title="Tests")


def nested_anova(y, factors: list[np.ndarray]) -> tuple[pd.DataFrame, bool]:
    """Sequential sums of squares for a fully nested hierarchy; returns the table and whether it is balanced."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    frame = pd.DataFrame({"y": y})
    keys = []
    for i, f in enumerate(factors):
        frame[f"f{i}"] = f
        keys.append(f"f{i}")
    parent_mean = np.full(n, y.mean())
    stats = []
    cells_prev = 1
    balanced = True
    for depth in range(1, len(keys) + 1):
        cell = frame.groupby(keys[:depth])["y"]
        mean = cell.transform("mean").to_numpy()
        cells = cell.ngroups
        ss = float(((mean - parent_mean) ** 2).sum())
        sizes = cell.size().to_numpy()
        balanced = balanced and len(set(sizes)) == 1
        stats.append({"df": cells - cells_prev, "ss": ss, "per_cell": n / cells})
        parent_mean, cells_prev = mean, cells
    ss_error, df_error = float(((y - parent_mean) ** 2).sum()), n - cells_prev
    if df_error < 1:
        raise ValueError("Not enough observations for the error term")
    if any(s["df"] < 1 for s in stats):
        raise ValueError("Each factor needs more than one level within the level above it")
    stats.append({"df": df_error, "ss": ss_error, "per_cell": 1.0})
    ms = [s["ss"] / s["df"] for s in stats]
    rows = []
    for i, s in enumerate(stats):
        last = i == len(stats) - 1
        f = np.nan if last else (ms[i] / ms[i + 1] if ms[i + 1] > 0 else np.nan)
        p = np.nan if last else float(sps.f.sf(f, s["df"], stats[i + 1]["df"]))
        component = ms[i] if last else max(0.0, (ms[i] - ms[i + 1]) / s["per_cell"])
        rows.append({"DF": s["df"], "SS": s["ss"], "MS": ms[i], "F-Value": f, "P-Value": p,
                     "Variance component": component})
    table = pd.DataFrame(rows)
    table["% of total"] = 100 * table["Variance component"] / table["Variance component"].sum()
    return table, balanced


@analysis("anova.nested", menu=MENU, title="Fully Nested ANOVA",
          params=[RESPONSE, Param("factor_a", "Factor A", "column", numeric=False),
                  Param("factor_b", "Factor B (nested in A)", "column", numeric=False),
                  Param("factor_c", "Factor C (nested in B)", "column", numeric=False, optional=True)])
def fully_nested(ws, response, factor_a, factor_b, factor_c=None):
    names = [f for f in (factor_a, factor_b, factor_c) if f]
    if len(set(names)) != len(names):
        raise ValueError("Select a different column for each factor")
    frame = ws.df[[response] + names].dropna()
    table, balanced = nested_anova(frame[response], [frame[f].map(format_value).to_numpy() for f in names])
    table.index = pd.Index(names + ["Error"], name="Source")
    res = Result("Fully Nested ANOVA")
    res.table(table, title="Analysis of Variance")
    if not balanced:
        res.text(tr("The design is unbalanced; variance components are approximate."))
    return res


@analysis("anova.mixed", menu=MENU, title="Mixed Effects Model",
          params=[RESPONSE, Param("random", "Random factor", "column", numeric=False),
                  Param("factors", "Fixed factors", "columns", numeric=False, optional=True),
                  Param("covariates", "Covariates", "columns", optional=True)])
def mixed_effects(ws, response, random, factors=None, covariates=None):
    factors = [f for f in (factors or []) if f != random]
    covariates = list(covariates or [])
    if factors or covariates:
        design = build_design(ws, [response], covariates, factors, extra_columns=[random])
        X, rows = design.X, design.rows
    else:
        rows = complete_rows(ws, [response, random])
        X = pd.DataFrame({CONSTANT: np.ones(len(rows))}, index=ws.df.index[rows])
    y = ws.df[response].iloc[rows].astype(float)
    groups = ws.df[random].iloc[rows].map(format_value)
    if groups.nunique() < 2:
        raise ValueError("The random factor needs at least two levels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence is reported in the output instead
        fit = sm.MixedLM(y, X, groups=groups).fit(reml=True)
    k = X.shape[1]
    res = Result("Mixed Effects Model")
    res.text(f"{tr('Random factor')}: {random} ({groups.nunique()} {tr('Levels')})    N = {len(y)}    REML")
    if not fit.converged:
        res.text(tr("Warning: the estimation did not converge; interpret the results with caution."))
    fixed = pd.DataFrame({"Coef": fit.params[:k], "SE Coef": fit.bse[:k], "Z-Value": fit.tvalues[:k],
                          "P-Value": fit.pvalues[:k]})
    fixed.index.name = "Term"
    res.table(fixed, title="Fixed Effects")
    between, within = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    total = between + within
    return res.table(pd.DataFrame({"Variance": [between, within, total],
                                   "% of total": [100 * between / total, 100 * within / total, 100.0],
                                   "StDev": np.sqrt([between, within, total])},
                                  index=pd.Index([random, "Error", "Total"], name="Source")),
                     title="Variance Components")


@analysis("anova.manova", menu=MENU, title="General MANOVA",
          params=[Param("responses", "Responses", "columns"), FACTORS,
                  Param("interactions", "Include two-way interactions", "bool", default=False)])
def manova(ws, responses, factors, interactions=False):
    from statsmodels.multivariate.manova import MANOVA

    if len(responses) < 2:
        raise ValueError("Select at least two responses")
    design = build_design(ws, list(responses), [], factors, interactions, coding="effect")
    Y = ws.df[list(responses)].iloc[design.rows].astype(float)
    names = list(design.X.columns)
    hypotheses = []
    for term, cols in design.terms:
        L = np.zeros((len(cols), len(names)))
        for i, col in enumerate(cols):
            L[i, names.index(col)] = 1.0
        hypotheses.append((term, L))
    tests = MANOVA(Y, design.X).mv_test(hypotheses)
    res = Result("General MANOVA")
    res.text(f"{tr('Responses')}: {', '.join(responses)}    N = {len(Y)}")
    for term, _ in design.terms:
        stat = tests.results[term]["stat"].astype(float)
        stat.columns = ["Statistic", "DF Num", "DF Den", "F-Value", "P-Value"]
        stat.index.name = "Criterion"
        res.table(stat, title=f"{tr('MANOVA tests for')} {term}")
    return res


# ----- factorial plots -----

@analysis("anova.main_effects", menu=MENU, title="Main Effects Plot", params=[RESPONSE, FACTORS])
def main_effects_plot(ws, response, factors):
    frame = ws.df[[response] + list(factors)].dropna()
    grand = frame[response].astype(float).mean()
    fig = new_figure(max(5.0, 2.6 * len(factors)), 3.8)
    axes = fig.subplots(1, len(factors), sharey=True, squeeze=False)[0]
    for ax, factor in zip(axes, factors):
        means = frame.groupby(frame[factor].map(format_value))[response].mean()
        ax.plot(list(means.index), means.to_numpy(), "-o", color=BLUE)
        ax.axhline(grand, color=GRAY, linestyle="--")
        ax.set_xlabel(factor)
    axes[0].set_ylabel(f"{tr('Mean')} ({response})")
    fig.suptitle(f"{tr('Main Effects Plot')}: {response}")
    return Result("Main Effects Plot").figure(fig)


@analysis("anova.interaction", menu=MENU, title="Interaction Plot",
          params=[RESPONSE, Param("factor1", "Factor on the X axis", "column", numeric=False),
                  Param("factor2", "Factor for the lines", "column", numeric=False)])
def interaction_plot(ws, response, factor1, factor2):
    if factor1 == factor2:
        raise ValueError("Select a different column for each factor")
    frame = ws.df[[response, factor1, factor2]].dropna()
    means = frame.groupby([frame[factor2].map(format_value), frame[factor1].map(format_value)])[response].mean()
    wide = means.unstack()
    fig = new_figure()
    ax = fig.add_subplot()
    for i, (level, row) in enumerate(wide.iterrows()):
        ax.plot(list(wide.columns), row.to_numpy(), "-o", color=PALETTE[i % len(PALETTE)], label=level)
    ax.legend(title=factor2)
    ax.set_xlabel(factor1), ax.set_ylabel(f"{tr('Mean')} ({response})")
    ax.set_title(f"{tr('Interaction Plot')}: {response}")
    return Result("Interaction Plot").figure(fig)
