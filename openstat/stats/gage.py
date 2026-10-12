"""Measurement systems analysis: gage R&R, type 1 study, linearity and bias, attribute agreement."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._tests import CONFIDENCE
from openstat.stats.anova import nested_anova
from openstat.stats.basic import clopper_pearson
from openstat.stats.quality import MENU as QUALITY_MENU
from openstat.stats.quality import optional_number

MENU = QUALITY_MENU + ("Gage Study",)
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"
PALETTE = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#7030a0", "#c00000", "#00b0f0", "#7f7f7f"]

PART = Param("part", "Part numbers", "column", numeric=False)
OPERATOR = Param("operator", "Operators", "column", numeric=False)
MEASUREMENT = Param("measurement", "Measurement data", "column")
TOLERANCE = Param("tolerance", "Process tolerance (blank = none)", "text", default="")
MULTIPLIER = Param("multiplier", "Study variation (number of standard deviations)", "number", default=6.0,
                   minimum=0.1)


def _frame(ws, part, operator, measurement) -> pd.DataFrame:
    frame = ws.df[[part, operator, measurement]].dropna()
    return pd.DataFrame({"part": frame[part].map(format_value), "operator": frame[operator].map(format_value),
                         "y": frame[measurement].astype(float)})


def crossed_anova(frame: pd.DataFrame, alpha_interaction: float = 0.05) -> dict:
    """Balanced two-way random-effects ANOVA and the gage R&R variance components."""
    cells = frame.groupby(["part", "operator"])["y"]
    counts = cells.size().unstack()
    if counts.isna().any().any() or counts.to_numpy().min() != counts.to_numpy().max():
        raise ValueError("The study must be balanced: every operator measures every part the same number of times")
    p, o, r = counts.shape[0], counts.shape[1], int(counts.iloc[0, 0])
    if p < 2 or o < 2 or r < 2:
        raise ValueError("A crossed study needs at least two parts, two operators and two replicates")
    grand = frame["y"].mean()
    part_means = frame.groupby("part")["y"].mean()
    oper_means = frame.groupby("operator")["y"].mean()
    cell_means = cells.mean().unstack()
    ss_part = o * r * ((part_means - grand) ** 2).sum()
    ss_oper = p * r * ((oper_means - grand) ** 2).sum()
    ss_cells = r * ((cell_means - grand) ** 2).to_numpy().sum()
    ss_inter = ss_cells - ss_part - ss_oper
    ss_total = ((frame["y"] - grand) ** 2).sum()
    ss_error = ss_total - ss_cells
    df = {"part": p - 1, "operator": o - 1, "interaction": (p - 1) * (o - 1), "error": p * o * (r - 1)}
    ms = {"part": ss_part / df["part"], "operator": ss_oper / df["operator"],
          "interaction": ss_inter / df["interaction"], "error": ss_error / df["error"]}
    f_inter = ms["interaction"] / ms["error"] if ms["error"] > 0 else np.inf
    p_inter = float(sps.f.sf(f_inter, df["interaction"], df["error"]))
    with_interaction = p_inter <= alpha_interaction
    rows = []
    if with_interaction:
        denom, df_denom = ms["interaction"], df["interaction"]
        repeat = ms["error"]
        inter = max(0.0, (ms["interaction"] - ms["error"]) / r)
    else:
        df_pooled = df["interaction"] + df["error"]
        denom = (ss_inter + ss_error) / df_pooled
        df_denom, repeat, inter = df_pooled, denom, 0.0
    for name, ss in (("part", ss_part), ("operator", ss_oper)):
        f = ms[name] / denom if denom > 0 else np.inf
        rows.append({"Source": name, "DF": df[name], "SS": ss, "MS": ms[name], "F-Value": f,
                     "P-Value": float(sps.f.sf(f, df[name], df_denom))})
    if with_interaction:
        rows.append({"Source": "interaction", "DF": df["interaction"], "SS": ss_inter, "MS": ms["interaction"],
                     "F-Value": f_inter, "P-Value": p_inter})
        rows.append({"Source": "Repeatability", "DF": df["error"], "SS": ss_error, "MS": ms["error"],
                     "F-Value": np.nan, "P-Value": np.nan})
    else:
        rows.append({"Source": "Repeatability", "DF": df_denom, "SS": ss_inter + ss_error, "MS": denom,
                     "F-Value": np.nan, "P-Value": np.nan})
    rows.append({"Source": "Total", "DF": p * o * r - 1, "SS": ss_total, "MS": np.nan, "F-Value": np.nan,
                 "P-Value": np.nan})
    oper = max(0.0, (ms["operator"] - denom) / (p * r))
    part = max(0.0, (ms["part"] - denom) / (o * r))
    return {"anova": pd.DataFrame(rows).set_index("Source"), "with_interaction": with_interaction,
            "p_interaction": p_inter, "parts": p, "operators": o, "replicates": r,
            "components": {"repeatability": repeat, "operator": oper, "interaction": inter, "part": part}}


def gage_tables(components: dict, multiplier=6.0, tolerance=None) -> tuple[pd.DataFrame, int]:
    """Variance components, study variation and number of distinct categories."""
    repeat = components["repeatability"]
    reproducibility = components["operator"] + components.get("interaction", 0.0)
    grr = repeat + reproducibility
    total = grr + components["part"]
    rows = [("Total Gage R&R", grr), ("  Repeatability", repeat), ("  Reproducibility", reproducibility),
            ("    Operator", components["operator"])]
    if components.get("interaction", 0.0) > 0:
        rows.append(("    Operator*Part", components["interaction"]))
    rows += [("Part-To-Part", components["part"]), ("Total Variation", total)]
    table = pd.DataFrame({"VarComp": [v for _, v in rows]}, index=pd.Index([n for n, _ in rows], name="Source"))
    table["% Contribution"] = 100 * table["VarComp"] / total if total > 0 else np.nan
    table["StdDev"] = np.sqrt(table["VarComp"])
    table["Study Var"] = multiplier * table["StdDev"]
    table["% Study Var"] = 100 * table["StdDev"] / np.sqrt(total) if total > 0 else np.nan
    if tolerance:
        table["% Tolerance"] = 100 * table["Study Var"] / tolerance
    ndc = int(max(1, np.floor(1.41 * np.sqrt(components["part"] / grr)))) if grr > 0 else 0
    return table, ndc


def _gage_figure(frame: pd.DataFrame, table: pd.DataFrame, title: str, crossed: bool):
    fig = new_figure(10.0, 7.0)
    (a, b), (c, d) = fig.subplots(2, 2)
    main = table.loc[["Total Gage R&R", "  Repeatability", "  Reproducibility", "Part-To-Part"]]
    pos = np.arange(len(main))
    a.bar(pos - 0.2, main["% Contribution"], 0.4, color=BLUE, label=tr("% Contribution"))
    a.bar(pos + 0.2, main["% Study Var"], 0.4, color="#ed7d31", label=tr("% Study Var"))
    a.set_xticks(pos, [tr(s.strip()) for s in main.index], fontsize=7)
    a.set_title(tr("Components of Variation"), fontsize=9), a.legend(fontsize=7)
    parts = sorted(frame["part"].unique())
    b.plot([parts.index(v) for v in frame["part"]], frame["y"], "o", color=GRAY, markersize=3)
    b.plot(range(len(parts)), frame.groupby("part")["y"].mean().reindex(parts), "-o", color=BLUE)
    b.set_xticks(range(len(parts)), parts, fontsize=7, rotation=45 if len(parts) > 10 else 0)
    b.set_title(tr("Measurement by Part"), fontsize=9)
    operators = sorted(frame["operator"].unique())
    c.boxplot([frame["y"][frame["operator"] == o] for o in operators], tick_labels=operators)
    c.set_title(tr("Measurement by Operator"), fontsize=9)
    if crossed:
        means = frame.groupby(["operator", "part"])["y"].mean().unstack()
        for i, (oper, row) in enumerate(means.iterrows()):
            d.plot(range(len(parts)), row.reindex(parts), "-o", color=PALETTE[i % len(PALETTE)], markersize=4,
                   label=oper)
        d.set_xticks(range(len(parts)), parts, fontsize=7, rotation=45 if len(parts) > 10 else 0)
        d.legend(fontsize=7, title=tr("Operator"))
        d.set_title(tr("Part * Operator Interaction"), fontsize=9)
    else:
        ranges = frame.groupby(["operator", "part"])["y"].agg(lambda v: v.max() - v.min())
        d.plot(range(len(ranges)), ranges.to_numpy(), "-o", color=BLUE, markersize=4)
        d.set_title(tr("Range by Part (within Operator)"), fontsize=9)
    fig.suptitle(title)
    return fig


def _gage_report(res: Result, frame, components, multiplier, tolerance, measurement, crossed) -> Result:
    table, ndc = gage_tables(components, multiplier, optional_number(tolerance))
    shown = table.copy()
    shown.index = pd.Index([s.replace(s.strip(), tr(s.strip())) for s in table.index], name="Source")
    res.table(shown[["VarComp", "% Contribution"]], title="Variance Components")
    res.table(shown.drop(columns=["VarComp", "% Contribution"]), title="Gage Evaluation")
    res.text(f"{tr('Number of distinct categories')} = {ndc}")
    return res.figure(_gage_figure(frame, table, f"{tr('Gage R&R')}: {measurement}", crossed))


@analysis("gage.crossed", menu=MENU, title="Gage R&R Study (Crossed)",
          params=[PART, OPERATOR, MEASUREMENT, TOLERANCE, MULTIPLIER])
def gage_crossed(ws, part, operator, measurement, tolerance="", multiplier=6.0):
    frame = _frame(ws, part, operator, measurement)
    out = crossed_anova(frame)
    res = Result("Gage R&R Study (Crossed)")
    res.text(f"{tr('Parts')}: {out['parts']}    {tr('Operators')}: {out['operators']}    "
             f"{tr('Replicates')}: {out['replicates']}")
    anova = out["anova"].rename(index={"part": part, "operator": operator, "interaction": f"{part} * {operator}"})
    title = "Two-Way ANOVA Table With Interaction" if out["with_interaction"] \
        else "Two-Way ANOVA Table Without Interaction"
    res.table(anova, title=title)
    if not out["with_interaction"]:
        res.text(f"{tr('The interaction was removed from the model')} (P = {out['p_interaction']:.3f} > 0.05).")
    return _gage_report(res, frame, out["components"], multiplier, tolerance, measurement, True)


@analysis("gage.nested", menu=MENU, title="Gage R&R Study (Nested)",
          params=[PART, OPERATOR, MEASUREMENT, TOLERANCE, MULTIPLIER])
def gage_nested(ws, part, operator, measurement, tolerance="", multiplier=6.0):
    frame = _frame(ws, part, operator, measurement)
    # parts are nested within operators: each operator measures a different set of parts
    nested_part = (frame["operator"] + " / " + frame["part"]).to_numpy()
    table, balanced = nested_anova(frame["y"], [frame["operator"].to_numpy(), nested_part])
    table.index = pd.Index([operator, f"{part} ({operator})", "Repeatability"], name="Source")
    res = Result("Gage R&R Study (Nested)")
    res.table(table[["DF", "SS", "MS", "F-Value", "P-Value"]], title="Nested ANOVA Table")
    if not balanced:
        res.text(tr("The design is unbalanced; variance components are approximate."))
    comps = table["Variance component"].to_numpy()
    components = {"repeatability": comps[2], "operator": comps[0], "interaction": 0.0, "part": comps[1]}
    return _gage_report(res, frame, components, multiplier, tolerance, measurement, False)


def type1_study(x, reference: float, tolerance: float, percent=20.0, multiplier=6.0) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        raise ValueError("At least two non-identical observations are needed")
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    bias = mean - reference
    t = bias / (sd / np.sqrt(n))
    k = percent / 100
    cg = k * tolerance / (multiplier * sd)
    cgk = (k / 2 * tolerance - abs(bias)) / (multiplier / 2 * sd)
    return {"n": n, "mean": mean, "sd": sd, "bias": bias, "t": t, "p": float(2 * sps.t.sf(abs(t), n - 1)),
            "cg": cg, "cgk": cgk, "var_repeat": percent / cg, "var_repeat_bias": percent / cgk if cgk > 0 else np.nan}


@analysis("gage.type1", menu=MENU, title="Type 1 Gage Study",
          params=[MEASUREMENT, Param("reference", "Reference", "number", default=0.0),
                  Param("tolerance", "Tolerance (upper spec - lower spec)", "number", default=1.0, minimum=1e-12)])
def gage_type1(ws, measurement, reference=0.0, tolerance=1.0):
    x = ws.df[measurement].dropna().to_numpy(dtype=float)
    out = type1_study(x, reference, tolerance)
    fig = new_figure(8.0, 4.0)
    ax = fig.add_subplot()
    ax.plot(np.arange(1, len(x) + 1), x, "-o", color=BLUE, markersize=4)
    for level, label in ((reference, tr("Reference")), (reference + 0.1 * tolerance, "Ref + 0.10 Tol"),
                         (reference - 0.1 * tolerance, "Ref - 0.10 Tol")):
        ax.axhline(level, color="#00803c" if level == reference else RED, linewidth=1)
        ax.annotate(label, (1.005, level), xycoords=("axes fraction", "data"), fontsize=7, va="center")
    ax.set_xlabel(tr("Observation")), ax.set_ylabel(measurement)
    ax.set_title(f"{tr('Type 1 Gage Study')}: {measurement}")
    res = Result("Type 1 Gage Study").figure(fig)
    res.table(pd.DataFrame([{"Reference": reference, "Mean": out["mean"], "StDev": out["sd"],
                             "Study Var": 6 * out["sd"], "Tolerance": tolerance}]), title="Basic Statistics",
              index=False)
    res.table(pd.DataFrame([{"Bias": out["bias"], "T-Value": out["t"], "P-Value": out["p"]}]), title="Bias",
              index=False)
    return res.table(pd.DataFrame([{"Cg": out["cg"], "Cgk": out["cgk"], "% Var (Repeatability)": out["var_repeat"],
                                    "% Var (Repeatability and Bias)": out["var_repeat_bias"]}]),
                     title="Capability", index=False, digits=2)


@analysis("gage.linearity", menu=MENU, title="Gage Linearity and Bias Study",
          params=[Param("reference", "Reference values", "column"), MEASUREMENT,
                  Param("variation", "Process variation (blank = none)", "text", default="")])
def gage_linearity(ws, reference, measurement, variation=""):
    frame = ws.df[[reference, measurement]].dropna().astype(float)
    ref = frame[reference].to_numpy()
    bias = frame[measurement].to_numpy() - ref
    if len(np.unique(ref)) < 2:
        raise ValueError("At least two different reference values are needed")
    fit = sps.linregress(ref, bias)
    n = len(ref)
    t_intercept = fit.intercept / fit.intercept_stderr
    process = optional_number(variation)
    rows = {}
    for value in np.unique(ref):
        b = bias[ref == value]
        p = float(sps.ttest_1samp(b, 0).pvalue) if len(b) > 1 and b.std(ddof=1) > 0 else np.nan
        rows[format_value(value)] = {"N": len(b), "Bias": b.mean(), "P-Value": p}
    overall_p = float(sps.ttest_1samp(bias, 0).pvalue) if bias.std(ddof=1) > 0 else np.nan
    rows[tr("Average")] = {"N": n, "Bias": bias.mean(), "P-Value": overall_p}
    res = Result("Gage Linearity and Bias Study")
    res.table(pd.DataFrame({"Coef": [fit.intercept, fit.slope], "SE Coef": [fit.intercept_stderr, fit.stderr],
                            "P-Value": [float(2 * sps.t.sf(abs(t_intercept), n - 2)), fit.pvalue]},
                           index=pd.Index(["Constant", "Slope"], name="Predictor")), title="Gage Linearity")
    summary = {"S": float(np.sqrt(((bias - fit.intercept - fit.slope * ref) ** 2).sum() / (n - 2))),
               "R-sq": 100 * fit.rvalue ** 2, "% Linearity": 100 * abs(fit.slope)}
    if process:
        summary["Linearity"] = abs(fit.slope) * process
        summary["% Bias"] = 100 * abs(bias.mean()) / process
    res.table(pd.DataFrame([summary]), index=False)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Reference"
    res.table(table, title="Gage Bias")
    fig = new_figure()
    ax = fig.add_subplot()
    ax.plot(ref, bias, "o", color=GRAY, markersize=4)
    means = pd.Series(bias).groupby(ref).mean()
    ax.plot(means.index, means.to_numpy(), "s", color=RED, label=tr("Average bias"))
    line = np.array([ref.min(), ref.max()])
    ax.plot(line, fit.intercept + fit.slope * line, color=BLUE, label=tr("Regression"))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(tr("Reference")), ax.set_ylabel(tr("Bias")), ax.legend()
    ax.set_title(tr("Gage Linearity and Bias Study"))
    return res.figure(fig)


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa from a subjects x categories table of rating counts (equal raters per subject)."""
    table = np.asarray(table, dtype=float)
    raters = table.sum(axis=1)
    if len(table) < 2 or raters.min() != raters.max() or raters[0] < 2:
        return float("nan")
    m = raters[0]
    p_cat = table.sum(axis=0) / table.sum()
    p_subject = ((table ** 2).sum(axis=1) - m) / (m * (m - 1))
    expected = (p_cat ** 2).sum()
    return float((p_subject.mean() - expected) / (1 - expected)) if expected < 1 else float("nan")


def _agreement_row(matched: int, inspected: int, confidence: float) -> dict:
    lo, hi = clopper_pearson(matched, inspected, confidence)
    return {"Inspected": inspected, "Matched": matched, "Percent": 100 * matched / inspected,
            "CI Lower": 100 * lo, "CI Upper": 100 * hi}


@analysis("gage.attribute", menu=QUALITY_MENU, title="Attribute Agreement Analysis",
          params=[Param("rating", "Ratings", "column", numeric=False),
                  Param("sample", "Samples", "column", numeric=False),
                  Param("appraiser", "Appraisers", "column", numeric=False),
                  Param("standard", "Known standard", "column", numeric=False, optional=True), CONFIDENCE])
def attribute_agreement(ws, rating, sample, appraiser, standard=None, confidence=95.0):
    cols = [rating, sample, appraiser] + ([standard] if standard else [])
    frame = ws.df[cols].dropna().map(format_value)
    samples = sorted(frame[sample].unique())
    res = Result("Attribute Agreement Analysis")
    within, versus = {}, {}
    for name, group in frame.groupby(appraiser, sort=True):
        per_sample = group.groupby(sample)[rating]
        if per_sample.size().min() > 1:
            agree = int((per_sample.nunique() == 1).sum())
            within[name] = _agreement_row(agree, per_sample.ngroups, confidence)
        if standard:
            truth = group.groupby(sample)[standard].first()
            ok = int(((per_sample.nunique() == 1) & (per_sample.first() == truth)).sum())
            versus[name] = _agreement_row(ok, per_sample.ngroups, confidence)
    if within:
        table = pd.DataFrame.from_dict(within, orient="index")
        table.index.name = appraiser
        res.table(table, title="Within Appraisers", digits=2)
    if versus:
        table = pd.DataFrame.from_dict(versus, orient="index")
        table.index.name = appraiser
        res.table(table, title="Each Appraiser vs Standard", digits=2)
    per_sample = frame.groupby(sample)[rating]
    between = _agreement_row(int((per_sample.nunique() == 1).sum()), len(samples), confidence)
    res.table(pd.DataFrame([between]), title="Between Appraisers", index=False, digits=2)
    if standard:
        truth = frame.groupby(sample)[standard].first()
        ok = int(((per_sample.nunique() == 1) & (per_sample.first() == truth)).sum())
        res.table(pd.DataFrame([_agreement_row(ok, len(samples), confidence)]), title="All Appraisers vs Standard",
                  index=False, digits=2)
    counts = pd.crosstab(frame[sample], frame[rating])
    kappa = fleiss_kappa(counts.to_numpy())
    res.table(pd.DataFrame([{"Kappa": kappa}]), title="Fleiss' Kappa (all ratings)", index=False)
    return res
