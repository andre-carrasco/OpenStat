"""Assistant: a decision tree that leads from a question about the data to an analysis, and the
graphs that suit a selection of columns."""
from __future__ import annotations

from dataclasses import dataclass, field

from openstat.core.registry import Analysis
from openstat.core.worksheet import Worksheet


@dataclass
class Node:
    text: str                                                    # a question, or the advice of a leaf
    options: list[tuple[str, "Node"]] = field(default_factory=list)
    analysis: str | None = None                                  # id of the analysis a leaf opens


def ask(text: str, *options: tuple[str, Node]) -> Node:
    return Node(text, list(options))


def leaf(analysis: str, text: str) -> Node:
    return Node(text, analysis=analysis)


TREE = ask(
    "What do you want to do?",
    ("Describe one variable", ask(
        "What kind of data is it?",
        ("Measurements", leaf("basic.graphical_summary",
                              "A graphical summary shows the shape, center and spread of the data, with a normality test.")),
        ("Categories", leaf("tables.tally", "A tally counts how often each category occurs.")))),
    ("Compare groups", ask(
        "What do you want to compare?",
        ("Means", ask(
            "How many groups are there?",
            ("One group, against a target", leaf("basic.t1", "A 1-sample t test compares the mean with a target value.")),
            ("Two groups", ask(
                "Is each observation of one group paired with one of the other?",
                ("Yes, the same items were measured twice",
                 leaf("basic.paired_t", "A paired t test looks at the differences within each pair.")),
                ("No, the groups are independent",
                 leaf("basic.t2", "A 2-sample t test compares the means of two independent groups.")))),
            ("More than two groups",
             leaf("anova.one_way", "One-way ANOVA tests whether any group mean differs; comparisons show which.")))),
        ("Standard deviations", ask(
            "How many groups are there?",
            ("One group, against a target", leaf("basic.var1", "A 1 variance test compares the spread with a target value.")),
            ("Two groups", leaf("basic.var2", "A 2 variances test compares the spread of two groups.")),
            ("More than two groups",
             leaf("anova.equal_variances", "The test for equal variances compares the spread of several groups.")))),
        ("Proportions or counts", ask(
            "What do the data count?",
            ("Items that pass or fail, in one group", leaf("basic.p1", "A 1 proportion test compares a share with a target.")),
            ("Items that pass or fail, in two groups", leaf("basic.p2", "A 2 proportions test compares two shares.")),
            ("Categories in several groups",
             leaf("tables.crosstab", "A chi-square test of a cross tabulation checks whether the groups differ.")),
            ("Occurrences per unit of time or space",
             leaf("basic.poisson1", "A Poisson rate test compares a rate of occurrence with a target.")))),
        ("Medians, when the data are far from normal", ask(
            "How many groups are there?",
            ("One group, against a target", leaf("nonpar.wilcoxon", "The Wilcoxon test compares the median with a target value.")),
            ("Two groups", leaf("nonpar.mann_whitney", "The Mann-Whitney test compares two independent groups.")),
            ("More than two groups", leaf("nonpar.kruskal", "The Kruskal-Wallis test compares several groups.")))))),
    ("Study how variables are related", ask(
        "What kind of response do you have?",
        ("A measurement", ask(
            "How many predictors are there?",
            ("One", leaf("reg.fitted_line", "A fitted line plot shows the response against one predictor with the fitted model.")),
            ("Several", leaf("reg.fit", "Regression estimates the effect of each predictor with the others held fixed.")))),
        ("Two categories, such as pass and fail",
         leaf("reg.logistic", "Binary logistic regression models the chance of an event.")),
        ("A count", leaf("reg.poisson", "Poisson regression models counts of occurrences.")),
        ("No response: see which variables move together",
         leaf("basic.correlation", "Correlation measures how strongly pairs of variables move together.")))),
    ("Monitor a process over time", ask(
        "What do you record?",
        ("Measurements, several at a time", leaf("cc.xbar_r", "An Xbar-R chart tracks the mean and the range of subgroups.")),
        ("Measurements, one at a time", leaf("cc.imr", "An I-MR chart tracks individual values and their moving range.")),
        ("The number of defective items", leaf("cc.p", "A P chart tracks the share of defective items.")),
        ("The number of defects", leaf("cc.u", "A U chart tracks the number of defects per unit.")),
        ("The time or the number of items between rare events",
         leaf("cc.g", "A G chart tracks the number of opportunities between rare events.")))),
    ("Assess whether a process meets its specifications", ask(
        "What do you record?",
        ("Measurements that follow a normal distribution",
         leaf("quality.capability_normal", "Capability analysis compares the spread of the process with its limits.")),
        ("Measurements that are not normal",
         leaf("quality.capability_nonnormal", "Nonnormal capability analysis uses a distribution that fits the data.")),
        ("The number of defective items", leaf("quality.capability_binomial", "Binomial capability estimates the share of defective items.")),
        ("The number of defects", leaf("quality.capability_poisson", "Poisson capability estimates the defects per unit.")))),
    ("Assess a measurement system", ask(
        "How are the parts measured?",
        ("Every operator measures every part several times",
         leaf("gage.crossed", "A crossed gage R&R study separates the variation of the gage from that of the parts.")),
        ("Each part can be measured by one operator only",
         leaf("gage.nested", "A nested gage R&R study is for destructive tests.")),
        ("One reference part is measured many times",
         leaf("gage.type1", "A type 1 gage study assesses bias and repeatability against a reference.")),
        ("The parts are rated, for example pass or fail",
         leaf("gage.attribute", "Attribute agreement analysis shows how consistent the appraisers are.")))),
    ("Run an experiment", ask(
        "What stage are you at?",
        ("Finding the few factors that matter among many",
         leaf("doe.create_dsd", "A definitive screening design studies many factors in few runs.")),
        ("Studying a few factors and their interactions",
         leaf("doe.create_factorial", "A two-level factorial design estimates effects and interactions.")),
        ("Finding the best settings, with curvature",
         leaf("doe.create_ccd", "A central composite design fits a quadratic response surface.")),
        ("Analyzing a two-level design that has been run",
         leaf("doe.analyze_factorial", "The analysis of a factorial design shows which effects are significant.")),
        ("Choosing the settings that optimize the responses",
         leaf("doe.optimizer", "The response optimizer searches for the settings with the best predicted responses.")))),
    ("Analyze times to failure", ask(
        "What do you need?",
        ("Fit a distribution to the times",
         leaf("rel.parametric", "Parametric distribution analysis estimates percentiles and survival, with censoring.")),
        ("Estimate survival without assuming a distribution",
         leaf("rel.kaplan_meier", "The Kaplan-Meier estimate gives survival probabilities directly from the data.")),
        ("See how predictors change the time to failure",
         leaf("rel.regression", "Regression with life data relates the time to failure to predictors.")))),
    ("Decide how much data to collect", ask(
        "What will the data be used for?",
        ("Comparing a mean with a target", leaf("power.t1", "Power and sample size for a 1-sample t test.")),
        ("Comparing two means", leaf("power.t2", "Power and sample size for a 2-sample t test.")),
        ("Comparing two proportions", leaf("power.p2", "Power and sample size for a test of two proportions.")),
        ("Estimating a mean or a proportion",
         leaf("power.estimation", "Sample size for an estimate with a chosen margin of error.")))),
)


def leaves(node: Node = TREE):
    """Every leaf of the tree below `node`."""
    if node.analysis:
        yield node
    for _, child in node.options:
        yield from leaves(child)


def strings(node: Node = TREE) -> set[str]:
    """Every text shown by the assistant, for the translation table."""
    found = {node.text}
    for label, child in node.options:
        found.add(label)
        found |= strings(child)
    return found


def preset(analysis: Analysis, ws: Worksheet, columns: list[str]) -> dict | None:
    """Values for the column parameters of `analysis` that use exactly the chosen columns, in the
    order they were chosen, or None when the analysis does not fit them.

    A parameter for numeric columns takes numeric columns; any other column parameter takes the
    columns that are not numeric, as those are the ones used for grouping.
    """
    numeric = [c for c in columns if ws.is_numeric(c)]
    other = [c for c in columns if not ws.is_numeric(c)]
    values = {}
    for p in analysis.params:
        if p.kind not in ("column", "columns"):
            continue
        pool = numeric if p.numeric else other
        if p.kind == "columns":
            taken, pool[:] = list(pool), []
        else:
            taken = pool.pop(0) if pool else None
        if not taken and not p.optional:
            return None
        values[p.name] = taken
    return None if numeric or other else values


def suggest_graphs(ws: Worksheet, columns: list[str], analyses: list[Analysis]) -> list[tuple[Analysis, dict]]:
    """The graphs that can be drawn from exactly these columns, with their dialogs filled in."""
    out = []
    for a in analyses:
        if a.menu[0] != "Graph" or not columns:
            continue
        values = preset(a, ws, list(columns))
        if values is not None:
            out.append((a, values))
    return out
