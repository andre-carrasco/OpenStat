"""Power and sample size. In each dialog leave exactly one of sample size, difference and power at 0 to solve it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._tests import ALPHA, ALTERNATIVE, CONFIDENCE, GREATER, LESS, NOT_EQUAL

MENU = ("Stat", "Power and Sample Size")
BLUE, RED = "#5b9bd5", "#c00000"

SIZE = Param("n", "Sample size (0 = calculate)", "int", default=0, minimum=0, maximum=10**7)
DIFFERENCE = Param("difference", "Difference (0 = calculate)", "number", default=1.0)
POWER = Param("power", "Power (0 = calculate)", "number", default=0.9, minimum=0.0, maximum=0.9999)
SD = Param("sd", "Standard deviation", "number", default=1.0, minimum=1e-12)
MAX_N = 10 ** 7


def solve(power_of, n, difference, power, minimum_n=2) -> tuple[int, float, float]:
    """Fill in whichever of (n, difference, power) is zero, given power_of(n, difference)."""
    missing = [n == 0, difference == 0, power == 0]
    if sum(missing) != 1:
        raise ValueError("Leave exactly one of sample size, difference and power at 0")
    if power == 0:
        return n, difference, float(power_of(n, difference))
    if n == 0:
        if power_of(minimum_n, difference) >= power:
            return minimum_n, difference, float(power_of(minimum_n, difference))
        high = minimum_n
        while power_of(high, difference) < power:
            high *= 2
            if high > MAX_N:
                raise ValueError("The required sample size is too large; check the difference and power")
        low = high // 2
        while high - low > 1:
            mid = (low + high) // 2
            if power_of(mid, difference) >= power:
                high = mid
            else:
                low = mid
        return high, difference, float(power_of(high, difference))
    if n < minimum_n:
        raise ValueError(f"The sample size must be at least {minimum_n}")
    scale = 1e-6
    high = scale
    while power_of(n, high) < power:
        high *= 2
        if high > 1e12:
            raise ValueError("The target power cannot be reached with this sample size")
    found = optimize.brentq(lambda d: power_of(n, d) - power, high / 2 if high > scale else 0.0, high)
    return n, float(found), float(power_of(n, found))


def z_power(ncp: float, alpha: float, alternative: str) -> float:
    if alternative == NOT_EQUAL:
        crit = sps.norm.isf(alpha / 2)
        return float(sps.norm.sf(crit - ncp) + sps.norm.cdf(-crit - ncp))
    crit = sps.norm.isf(alpha)
    return float(sps.norm.sf(crit - ncp) if alternative == GREATER else sps.norm.cdf(-crit - ncp))


def t_power(ncp: float, df: float, alpha: float, alternative: str) -> float:
    if df < 1:
        return 0.0
    if alternative == NOT_EQUAL:
        crit = sps.t.isf(alpha / 2, df)
        return float(sps.nct.sf(crit, df, ncp) + sps.nct.cdf(-crit, df, ncp))
    crit = sps.t.isf(alpha, df)
    return float(sps.nct.sf(crit, df, ncp) if alternative == GREATER else sps.nct.cdf(-crit, df, ncp))


def f_power(ncp: float, df1: float, df2: float, alpha: float) -> float:
    crit = sps.f.isf(alpha, df1, df2)
    # the noncentral F is not defined at zero noncentrality in SciPy
    return float(sps.ncf.sf(crit, df1, df2, ncp)) if ncp > 0 else float(sps.f.sf(crit, df1, df2))


def _signed(difference: float, alternative: str) -> float:
    """Differences are entered as magnitudes for two-sided tests and signed for one-sided tests."""
    return abs(difference) if alternative == NOT_EQUAL else difference


def _curve(power_of, n: int, difference: float, xlabel: str, title: str):
    fig = new_figure()
    ax = fig.add_subplot()
    span = abs(difference) * 2 or 1.0
    grid = np.linspace(-span if difference < 0 else 0, 0 if difference < 0 else span, 120)
    ax.plot(grid, [power_of(n, d) for d in grid], color=BLUE)
    ax.plot(difference, power_of(n, difference), "o", color=RED)
    ax.set_ylim(0, 1.02), ax.set_xlabel(tr(xlabel)), ax.set_ylabel(tr("Power"))
    ax.set_title(f"{tr('Power Curve')}: {tr(title)} (n = {n})")
    return fig


def _report(title: str, power_of, n, difference, power, notes: str, size_label="Sample size",
            xlabel="Difference", minimum_n=2) -> Result:
    n, difference, power = solve(power_of, n, difference, power, minimum_n)
    res = Result(title)
    res.text(notes)
    res.table(pd.DataFrame([{"Difference": difference, size_label: n, "Power": power}]), index=False, digits=5)
    return res.figure(_curve(power_of, n, difference, xlabel, title))


def _notes(alpha, alternative, extra="") -> str:
    return f"α = {alpha:g}    {tr('Alternative hypothesis')}: {tr(alternative)}" + (f"    {extra}" if extra else "")


@analysis("power.z1", menu=MENU, title="Power: 1-Sample Z", params=[SIZE, DIFFERENCE, POWER, SD, ALPHA, ALTERNATIVE])
def power_one_sample_z(ws, n=0, difference=0.0, power=0.9, sd=1.0, alpha=0.05, alternative=NOT_EQUAL):
    def power_of(size, diff):
        return z_power(_signed(diff, alternative) * np.sqrt(size) / sd, alpha, alternative)
    return _report("Power: 1-Sample Z", power_of, n, difference, power, _notes(alpha, alternative, f"σ = {sd:g}"),
                   minimum_n=1)


@analysis("power.t1", menu=MENU, title="Power: 1-Sample t", params=[SIZE, DIFFERENCE, POWER, SD, ALPHA, ALTERNATIVE])
def power_one_sample_t(ws, n=0, difference=0.0, power=0.9, sd=1.0, alpha=0.05, alternative=NOT_EQUAL):
    def power_of(size, diff):
        return t_power(_signed(diff, alternative) * np.sqrt(size) / sd, size - 1, alpha, alternative)
    return _report("Power: 1-Sample t", power_of, n, difference, power, _notes(alpha, alternative, f"σ = {sd:g}"))


@analysis("power.t2", menu=MENU, title="Power: 2-Sample t", params=[SIZE, DIFFERENCE, POWER, SD, ALPHA, ALTERNATIVE])
def power_two_sample_t(ws, n=0, difference=0.0, power=0.9, sd=1.0, alpha=0.05, alternative=NOT_EQUAL):
    def power_of(size, diff):
        return t_power(_signed(diff, alternative) / (sd * np.sqrt(2 / size)), 2 * size - 2, alpha, alternative)
    return _report("Power: 2-Sample t", power_of, n, difference, power,
                   _notes(alpha, alternative, f"σ = {sd:g}    {tr('The sample size is for each group.')}"),
                   size_label="Sample size per group")


@analysis("power.paired", menu=MENU, title="Power: Paired t",
          params=[SIZE, DIFFERENCE, POWER, Param("sd", "Standard deviation of paired differences", "number",
                                                 default=1.0, minimum=1e-12), ALPHA, ALTERNATIVE])
def power_paired_t(ws, n=0, difference=0.0, power=0.9, sd=1.0, alpha=0.05, alternative=NOT_EQUAL):
    def power_of(size, diff):
        return t_power(_signed(diff, alternative) * np.sqrt(size) / sd, size - 1, alpha, alternative)
    return _report("Power: Paired t", power_of, n, difference, power, _notes(alpha, alternative, f"σd = {sd:g}"),
                   size_label="Number of pairs")


def one_proportion_power(n, p0, p1, alpha, alternative) -> float:
    """Normal-approximation power of the test of H0: p = p0 when the true proportion is p1."""
    if not 0 < p1 < 1:
        return 0.0
    se0, se1 = np.sqrt(p0 * (1 - p0) / n), np.sqrt(p1 * (1 - p1) / n)
    if alternative == NOT_EQUAL:
        crit = sps.norm.isf(alpha / 2)
        return float(sps.norm.sf((p0 + crit * se0 - p1) / se1) + sps.norm.cdf((p0 - crit * se0 - p1) / se1))
    crit = sps.norm.isf(alpha)
    if alternative == GREATER:
        return float(sps.norm.sf((p0 + crit * se0 - p1) / se1))
    return float(sps.norm.cdf((p0 - crit * se0 - p1) / se1))


def two_proportion_power(n, p1, p2, alpha, alternative) -> float:
    """Normal-approximation power of the pooled two-proportion test with n per group."""
    if not (0 < p1 < 1 and 0 < p2 < 1):
        return 0.0
    pooled = (p1 + p2) / 2
    se0 = np.sqrt(2 * pooled * (1 - pooled) / n)
    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    diff = p1 - p2
    if alternative == NOT_EQUAL:
        crit = sps.norm.isf(alpha / 2)
        return float(sps.norm.sf((crit * se0 - diff) / se1) + sps.norm.cdf((-crit * se0 - diff) / se1))
    crit = sps.norm.isf(alpha)
    return float(sps.norm.sf((crit * se0 - diff) / se1) if alternative == GREATER
                 else sps.norm.cdf((-crit * se0 - diff) / se1))


COMPARISON = Param("comparison", "Comparison proportion (0 = calculate)", "number", default=0.7, minimum=0.0,
                   maximum=0.9999)


def _proportion_report(title, power_at, n, comparison, power, baseline, alpha, alternative, size_label) -> Result:
    if sum([n == 0, comparison == 0, power == 0]) != 1:
        raise ValueError("Leave exactly one of sample size, comparison proportion and power at 0")
    if power == 0:
        power = power_at(n, comparison)
    elif n == 0:
        n, _, power = solve(lambda size, _d: power_at(size, comparison), 0, 1.0, power)
    else:
        upward = alternative != LESS
        lo, hi = (baseline + 1e-9, 1 - 1e-9) if upward else (1e-9, baseline - 1e-9)
        if max(power_at(n, lo), power_at(n, hi)) < power:
            raise ValueError("The target power cannot be reached with this sample size")
        comparison = float(optimize.brentq(lambda p: power_at(n, p) - power, lo, hi))
        power = power_at(n, comparison)
    res = Result(title)
    res.text(_notes(alpha, alternative, f"{tr('Baseline proportion')} = {baseline:g}    "
                                        f"{tr('Normal approximation')}"))
    res.table(pd.DataFrame([{"Comparison p": comparison, size_label: n, "Power": power}]), index=False, digits=5)
    fig = new_figure()
    ax = fig.add_subplot()
    grid = np.linspace(0.001, 0.999, 200)
    ax.plot(grid, [power_at(n, p) for p in grid], color=BLUE)
    ax.plot(comparison, power, "o", color=RED)
    ax.set_ylim(0, 1.02), ax.set_xlabel(tr("Comparison proportion")), ax.set_ylabel(tr("Power"))
    ax.set_title(f"{tr('Power Curve')}: {tr(title)} (n = {n})")
    return res.figure(fig)


@analysis("power.p1", menu=MENU, title="Power: 1 Proportion",
          params=[SIZE, COMPARISON, POWER,
                  Param("p0", "Hypothesized proportion", "number", default=0.5, minimum=0.0001, maximum=0.9999),
                  ALPHA, ALTERNATIVE])
def power_one_proportion(ws, n=0, comparison=0.0, power=0.9, p0=0.5, alpha=0.05, alternative=NOT_EQUAL):
    return _proportion_report("Power: 1 Proportion", lambda size, p: one_proportion_power(size, p0, p, alpha,
                                                                                         alternative),
                              n, comparison, power, p0, alpha, alternative, "Sample size")


@analysis("power.p2", menu=MENU, title="Power: 2 Proportions",
          params=[SIZE, COMPARISON, POWER,
                  Param("p2", "Baseline proportion", "number", default=0.5, minimum=0.0001, maximum=0.9999),
                  ALPHA, ALTERNATIVE])
def power_two_proportions(ws, n=0, comparison=0.0, power=0.9, p2=0.5, alpha=0.05, alternative=NOT_EQUAL):
    return _proportion_report("Power: 2 Proportions", lambda size, p: two_proportion_power(size, p, p2, alpha,
                                                                                          alternative),
                              n, comparison, power, p2, alpha, alternative, "Sample size per group")


def anova_power(n: int, levels: int, difference: float, sd: float, alpha: float) -> float:
    """Power of one-way ANOVA when two level means differ by `difference` and the rest sit at the grand mean."""
    df1, df2 = levels - 1, levels * (n - 1)
    if df2 < 1:
        return 0.0
    return f_power(n * difference ** 2 / (2 * sd ** 2), df1, df2, alpha)


@analysis("power.anova", menu=MENU, title="Power: One-Way ANOVA",
          params=[Param("levels", "Number of levels", "int", default=3, minimum=2, maximum=1000),
                  SIZE, Param("difference", "Maximum difference between means (0 = calculate)", "number", default=1.0),
                  POWER, SD, ALPHA])
def power_anova(ws, levels=3, n=0, difference=0.0, power=0.9, sd=1.0, alpha=0.05):
    def power_of(size, diff):
        return anova_power(size, levels, diff, sd, alpha)
    return _report("Power: One-Way ANOVA", power_of, n, abs(difference), power,
                   f"α = {alpha:g}    σ = {sd:g}    {tr('Number of levels')}: {levels}    "
                   f"{tr('The sample size is for each level.')}", size_label="Sample size per level",
                   xlabel="Maximum difference")


def factorial_power(replicates: int, runs: int, terms: int, effect: float, sd: float, alpha: float) -> float:
    """Power to detect an effect in a two-level factorial with `runs` corner points per replicate."""
    total = runs * replicates
    df = total - terms - 1
    if df < 1:
        return 0.0
    return f_power(total * effect ** 2 / (4 * sd ** 2), 1, df, alpha)


@analysis("power.factorial", menu=MENU, title="Power: 2-Level Factorial Design",
          params=[Param("runs", "Number of corner points", "int", default=8, minimum=4, maximum=4096),
                  Param("terms", "Number of terms in the model", "int", default=7, minimum=1, maximum=4095),
                  Param("n", "Replicates (0 = calculate)", "int", default=0, minimum=0, maximum=10000),
                  Param("difference", "Effect (0 = calculate)", "number", default=2.0), POWER, SD, ALPHA])
def power_factorial(ws, runs=8, terms=7, n=0, difference=0.0, power=0.9, sd=1.0, alpha=0.05):
    if terms >= runs:
        raise ValueError("The model has too many terms for this number of corner points")

    def power_of(size, diff):
        return factorial_power(size, runs, terms, diff, sd, alpha)
    return _report("Power: 2-Level Factorial Design", power_of, n, abs(difference), power,
                   f"α = {alpha:g}    σ = {sd:g}    {tr('Number of corner points')}: {runs}    "
                   f"{tr('Number of terms in the model')}: {terms}", size_label="Replicates", xlabel="Effect",
                   minimum_n=1)


def variance_power(n: int, ratio: float, alpha: float, alternative: str, two_sample: bool) -> float:
    """Power of the chi-square (one sample) or F (two samples) test for a standard deviation ratio."""
    if n < 2 or ratio <= 0:
        return 0.0
    dist = sps.f(n - 1, n - 1) if two_sample else sps.chi2(n - 1)
    k = ratio ** 2
    if alternative == GREATER:
        return float(dist.sf(dist.isf(alpha) / k))
    if alternative == LESS:
        return float(dist.cdf(dist.ppf(alpha) / k))
    return float(dist.sf(dist.isf(alpha / 2) / k) + dist.cdf(dist.ppf(alpha / 2) / k))


def _variance_report(title, two_sample, n, ratio, power, alpha, alternative) -> Result:
    if sum([n == 0, ratio == 0, power == 0]) != 1:
        raise ValueError("Leave exactly one of sample size, ratio and power at 0")

    def power_at(size, r):
        return variance_power(size, r, alpha, alternative, two_sample)

    if power == 0:
        power = power_at(n, ratio)
    elif n == 0:
        if ratio == 1:
            raise ValueError("A ratio of 1 is the null hypothesis; no sample size gives this power")
        n, _, power = solve(lambda size, _d: power_at(size, ratio), 0, 1.0, power)
    else:
        lo, hi = (1 + 1e-9, 1e6) if alternative != LESS else (1e-6, 1 - 1e-9)
        if max(power_at(n, lo), power_at(n, hi)) < power:
            raise ValueError("The target power cannot be reached with this sample size")
        ratio = float(optimize.brentq(lambda r: power_at(n, r) - power, lo, hi))
        power = power_at(n, ratio)
    res = Result(title)
    res.text(_notes(alpha, alternative))
    label = "Sample size per group" if two_sample else "Sample size"
    res.table(pd.DataFrame([{"Ratio of StDevs": ratio, label: n, "Power": power}]), index=False, digits=5)
    fig = new_figure()
    ax = fig.add_subplot()
    grid = np.linspace(0.2, max(3.0, 1.5 * ratio), 200)
    ax.plot(grid, [power_at(n, r) for r in grid], color=BLUE)
    ax.plot(ratio, power, "o", color=RED)
    ax.set_ylim(0, 1.02), ax.set_xlabel(tr("Ratio of StDevs")), ax.set_ylabel(tr("Power"))
    ax.set_title(f"{tr('Power Curve')}: {tr(title)} (n = {n})")
    return res.figure(fig)


RATIO = Param("ratio", "Ratio of standard deviations (0 = calculate)", "number", default=1.5, minimum=0.0)


@analysis("power.var1", menu=MENU, title="Power: 1 Variance", params=[SIZE, RATIO, POWER, ALPHA, ALTERNATIVE])
def power_one_variance(ws, n=0, ratio=0.0, power=0.9, alpha=0.05, alternative=NOT_EQUAL):
    return _variance_report("Power: 1 Variance", False, n, ratio, power, alpha, alternative)


@analysis("power.var2", menu=MENU, title="Power: 2 Variances", params=[SIZE, RATIO, POWER, ALPHA, ALTERNATIVE])
def power_two_variances(ws, n=0, ratio=0.0, power=0.9, alpha=0.05, alternative=NOT_EQUAL):
    return _variance_report("Power: 2 Variances", True, n, ratio, power, alpha, alternative)


def equivalence_power(n: int, difference: float, lower: float, upper: float, sd: float, alpha: float,
                      two_sample: bool) -> float:
    """Approximate power of the two one-sided tests procedure (noncentral t)."""
    df = 2 * n - 2 if two_sample else n - 1
    if df < 1:
        return 0.0
    se = sd * np.sqrt(2 / n) if two_sample else sd / np.sqrt(n)
    crit = sps.t.isf(alpha, df)
    value = sps.nct.cdf(-crit, df, (difference - upper) / se) - sps.nct.cdf(crit, df, (difference - lower) / se)
    return float(max(value, 0.0))


@analysis("power.equivalence", menu=MENU, title="Power: Equivalence Test",
          params=[Param("design", "Design", "choice", default="1-Sample or paired",
                        choices=["1-Sample or paired", "2-Sample"]),
                  SIZE, Param("difference", "True difference", "number", default=0.0), POWER,
                  Param("lower", "Lower equivalence limit", "number", default=-1.0),
                  Param("upper", "Upper equivalence limit", "number", default=1.0), SD, ALPHA])
def power_equivalence(ws, design="1-Sample or paired", n=0, difference=0.0, power=0.9, lower=-1.0, upper=1.0,
                      sd=1.0, alpha=0.05):
    if lower >= upper:
        raise ValueError("The lower limit must be less than the upper limit")
    if not lower < difference < upper:
        raise ValueError("The true difference must lie inside the equivalence limits")
    if (n == 0) == (power == 0):
        raise ValueError("Leave exactly one of sample size and power at 0")
    two = design == "2-Sample"

    def power_at(size):
        return equivalence_power(size, difference, lower, upper, sd, alpha, two)

    if n == 0:
        n, _, power = solve(lambda size, _d: power_at(size), 0, 1.0, power)
    else:
        power = power_at(n)
    res = Result("Power: Equivalence Test")
    res.text(f"α = {alpha:g}    σ = {sd:g}    {tr('Equivalence interval')}: ({lower:g}, {upper:g})    {tr(design)}")
    res.table(pd.DataFrame([{"Difference": difference, "Sample size per group" if two else "Sample size": n,
                             "Power": power}]), index=False, digits=5)
    fig = new_figure()
    ax = fig.add_subplot()
    grid = np.linspace(lower, upper, 160)
    ax.plot(grid, [equivalence_power(n, d, lower, upper, sd, alpha, two) for d in grid], color=BLUE)
    ax.plot(difference, power, "o", color=RED)
    ax.set_ylim(0, 1.02), ax.set_xlabel(tr("Difference")), ax.set_ylabel(tr("Power"))
    ax.set_title(f"{tr('Power Curve')}: {tr('Power: Equivalence Test')} (n = {n})")
    return res.figure(fig)


# ----- estimation -----

def mean_estimation_size(margin: float, sd: float, confidence: float) -> int:
    """Smallest n whose t interval for the mean has half-width <= margin."""
    def half_width(n):
        return sps.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n)

    n = max(2, int(np.ceil((sps.norm.ppf(0.5 + confidence / 2) * sd / margin) ** 2)))
    if n > MAX_N:
        raise ValueError("The required sample size is too large")
    while half_width(n) > margin:
        n += 1
    while n > 2 and half_width(n - 1) <= margin:
        n -= 1
    return n


def proportion_estimation_size(margin: float, p: float, confidence: float) -> int:
    z = sps.norm.ppf(0.5 + confidence / 2)
    return int(np.ceil(z ** 2 * p * (1 - p) / margin ** 2))


PARAMETERS = ["Mean (Normal)", "Proportion (Binomial)"]


@analysis("power.estimation", menu=MENU, title="Sample Size for Estimation",
          params=[Param("parameter", "Parameter", "choice", default=PARAMETERS[0], choices=PARAMETERS),
                  Param("margin", "Margin of error", "number", default=0.5, minimum=1e-9),
                  Param("sd", "Standard deviation (for the mean)", "number", default=1.0, minimum=1e-12),
                  Param("p", "Planning value of the proportion", "number", default=0.5, minimum=0.0001,
                        maximum=0.9999), CONFIDENCE])
def estimation_sample_size(ws, parameter=PARAMETERS[0], margin=0.5, sd=1.0, p=0.5, confidence=95.0):
    level = confidence / 100
    if parameter == PARAMETERS[0]:
        n = mean_estimation_size(margin, sd, level)
        note = f"σ = {sd:g}"
    else:
        if margin >= 1:
            raise ValueError("The margin of error for a proportion must be less than 1")
        n = proportion_estimation_size(margin, p, level)
        note = f"p = {p:g}    {tr('Normal approximation')}"
    res = Result("Sample Size for Estimation")
    res.text(f"{tr(parameter)}    {confidence:g}% {tr('CI')}    {note}")
    return res.table(pd.DataFrame([{"Margin of error": margin, "Sample size": n}]), index=False)


def tolerance_sample_size(coverage: float, confidence: float) -> int:
    """Smallest n for which (minimum, maximum) is a tolerance interval with the given coverage and confidence."""
    def achieved(n):
        return 1 - n * coverage ** (n - 1) + (n - 1) * coverage ** n
    high = 2
    while achieved(high) < confidence:
        high *= 2
        if high > MAX_N:
            raise ValueError("The required sample size is too large")
    low = max(high // 2, 2)
    while low < high:
        mid = (low + high) // 2
        if achieved(mid) >= confidence:
            high = mid
        else:
            low = mid + 1
    return low


@analysis("power.tolerance", menu=MENU, title="Sample Size for Tolerance Intervals",
          params=[Param("coverage", "Minimum percentage of population in interval", "number", default=95.0,
                        minimum=1, maximum=99.999), CONFIDENCE])
def tolerance_interval_size(ws, coverage=95.0, confidence=95.0):
    from openstat.stats.quality import tolerance_factor

    n = tolerance_sample_size(coverage / 100, confidence / 100)
    res = Result("Sample Size for Tolerance Intervals")
    res.text(f"{tr('Minimum percentage of population in interval')}: {coverage:g}    "
             f"{tr('Confidence level (%)')}: {confidence:g}")
    res.table(pd.DataFrame([{"Sample size": n,
                             "Achieved confidence (%)": 100 * (1 - n * (coverage / 100) ** (n - 1)
                                                               + (n - 1) * (coverage / 100) ** n)}]),
              title="Nonparametric method (smallest and largest observation)", index=False)
    sizes = [10, 20, 30, 50, 100, 200, 500]
    return res.table(pd.DataFrame({"k": [tolerance_factor(m, coverage / 100, confidence / 100) for m in sizes]},
                                  index=pd.Index(sizes, name="Sample size")),
                     title="Normal method: interval = mean ± k × StDev")
