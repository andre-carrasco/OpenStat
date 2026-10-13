"""More control charts, capability, tolerance, sampling and measurement studies, checked against
published constants and direct calculation."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import _spc, anova, control, gage, quality


def tables(res):
    return [payload[0] for kind, payload in res.blocks if kind == "table"]


@pytest.fixture
def groups():
    rng = np.random.default_rng(3)
    shifts = rng.normal(0, 1.5, 25)
    return [100 + shift + rng.normal(0, 2, 5) for shift in shifts]


@pytest.fixture
def ws(groups):
    rng = np.random.default_rng(4)
    x = np.concatenate(groups)
    return Worksheet("Q", pd.DataFrame({
        "x": x, "y": 50 + 0.5 * (x - 100) + rng.normal(0, 1, len(x)), "w": rng.normal(10, 1, len(x)),
        "part": np.array(["a", "b", "c"], dtype=object)[np.arange(len(x)) // 45]}))


def test_single_charts(ws, groups):
    limits = tables(control.xbar_chart(ws, "x"))[0]
    sigma = _spc.sigma_pooled(groups)
    assert limits.loc["Xbar", "CL"] == pytest.approx(np.concatenate(groups).mean())
    assert limits.loc["Xbar", "UCL"] == pytest.approx(np.concatenate(groups).mean() + 3 * sigma / np.sqrt(5))
    r_bar = np.mean([np.ptp(g) for g in groups])
    limits = tables(control.r_chart(ws, "x"))[0]
    assert limits.loc["R", "CL"] == pytest.approx(r_bar) and limits.loc["R", "UCL"] == pytest.approx(2.114 * r_bar, rel=1e-3)
    s_bar = np.mean([g.std(ddof=1) for g in groups])
    limits = tables(control.s_chart(ws, "x"))[0]
    assert limits.loc["S", "CL"] == pytest.approx(s_bar) and limits.loc["S", "UCL"] == pytest.approx(2.089 * s_bar, rel=1e-3)


def test_between_within(ws, groups):
    parts = control.between_within(groups)
    means = np.array([g.mean() for g in groups])
    of_means = np.abs(np.diff(means)).mean() / 1.128
    assert parts["within"] == pytest.approx(_spc.sigma_pooled(groups))
    assert parts["between"] == pytest.approx(np.sqrt(of_means ** 2 - parts["within"] ** 2 / 5), rel=1e-3)
    assert parts["total"] ** 2 == pytest.approx(parts["within"] ** 2 + parts["between"] ** 2)
    quiet = [np.array([1.0, 2.0, 3.0]) + 0.01 * i for i in range(6)]
    assert control.between_within(quiet)["between"] == 0.0             # no more than the within variation explains
    out = tables(control.imr_rs(ws, "x"))
    assert out[0].loc["Between/Within", "StDev"] == pytest.approx(parts["total"])
    assert out[1].loc["I", "UCL"] == pytest.approx(means.mean() + 3 * of_means, rel=1e-3)
    with pytest.raises(ValueError):
        control.between_within([np.array([1.0]), np.array([2.0])])


def test_z_mr_standardizes_each_part():
    x = np.array([10.0, 11.0, 9.0, 10.5, 100.0, 104.0, 98.0, 101.0])
    parts = np.array(["p"] * 4 + ["q"] * 4)
    z, table = control.standardize_by_part(x, parts)
    sigma_p = np.abs(np.diff(x[:4])).mean() / 1.128
    assert table.loc["p", "StDev"] == pytest.approx(sigma_p, rel=1e-3) and table.loc["q", "Mean"] == 100.75
    assert z[:4] == pytest.approx((x[:4] - x[:4].mean()) / table.loc["p", "StDev"])
    pooled, table = control.standardize_by_part(x, parts, pooled=True)
    assert table["StDev"].nunique() == 1 and abs(pooled[4]) < abs(z[4]) * 3
    ws = Worksheet("Z", pd.DataFrame({"x": x, "part": parts.astype(object)}))
    res = control.zmr(ws, "x", "part")
    assert tables(res)[1].loc["MR", "UCL"] == pytest.approx(3.686, abs=2e-3)
    with pytest.raises(ValueError):
        control.standardize_by_part([1.0, 2.0, 5.0], np.array(["a", "a", "b"]))


def test_zone_scores():
    # weights 0, 2, 4, 8 by zone; the score restarts on crossing the center line and after a signal
    assert control.zone_scores([0.5, 1.5, 1.2, 2.5]).tolist() == [0, 2, 4, 8]
    assert control.zone_scores([1.5, 1.5, -1.5, -2.5]).tolist() == [2, 4, 2, 6]
    assert control.zone_scores([3.5, 0.2, 2.2]).tolist() == [8, 0, 4]
    assert control.zone_scores([1.5, 0.0, 1.5]).tolist() == [2, 2, 4]           # a point on the line changes nothing
    rng = np.random.default_rng(1)
    ws = Worksheet("Z", pd.DataFrame({"x": np.concatenate([rng.normal(0, 1, 30), rng.normal(2.5, 1, 10)])}))
    res = control.zone_chart(ws, "x")
    assert res.blocks[-1][0] == "text" and any(int(i) > 30 for i in res.blocks[-1][1].rsplit(": ", 1)[1].split(", "))


def test_generalized_variance_and_subgroup_t_squared(ws):
    groups = control.multivariate_subgroups(ws, ["x", "y"], 10)
    gv = control.generalized_variance(groups)
    # Montgomery: b1 = 0.8889 and b2 = 0.4170 for two variables in subgroups of ten
    sigma = gv["center"] / 0.8889
    assert gv["center"] == pytest.approx(np.linalg.det(np.mean([np.cov(g, rowvar=False) for g in groups], axis=0)))
    assert (gv["ucl"] - gv["center"]) / sigma == pytest.approx(3 * np.sqrt(0.4170), rel=1e-3)
    assert gv["lcl"] == 0.0 and len(gv["values"]) == len(groups)
    one = control.generalized_variance([g[:, :1] for g in groups])            # one variable: the variance chart
    assert (one["ucl"] - one["center"]) / one["center"] == pytest.approx(3 * np.sqrt(2 / 9))

    t2 = control.subgroup_t_squared(groups)
    means = np.array([g.mean(axis=0) for g in groups])
    d = means[3] - means.mean(axis=0)
    assert t2["values"][3] == pytest.approx(10 * d @ np.linalg.inv(t2["pooled"]) @ d)
    m, n, p = len(groups), 10, 2
    df = m * n - m - p + 1
    assert t2["ucl"] == pytest.approx(p * (m - 1) * (n - 1) / df * sps.f.ppf(1 - 0.0027, p, df))
    assert len(tables(control.t2_generalized_variance(ws, ["x", "y"], 10))[0]) == 2
    with pytest.raises(ValueError):
        control.multivariate_subgroups(ws, ["x", "y"], 2)


def test_mewma():
    # with a weight of 1 the chart is Hotelling's and the run length is geometric
    assert control.mewma_arl(10.0, 3, 1.0) == pytest.approx(1 / sps.chi2.sf(10.0, 3), rel=1e-6)
    # Lowry et al. (1992): h = 8.66 gives an in-control ARL of 200 for two variables and a weight of 0.1
    assert control.mewma_arl(8.66, 2, 0.1) == pytest.approx(200, rel=0.05)
    assert control.mewma_limit(2, 0.1, 200) == pytest.approx(8.66, abs=0.15)
    assert control.mewma_limit(4, 0.1, 200) > control.mewma_limit(2, 0.1, 200)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 2))
    values = control.mewma(X, 1.0)
    centered = X - X.mean(axis=0)
    assert values == pytest.approx(np.einsum("ij,jk,ik->i", centered, np.linalg.inv(np.cov(X, rowvar=False)), centered))
    shifted = np.vstack([X, rng.normal(size=(20, 2)) + 2.0])
    ws = Worksheet("M", pd.DataFrame(shifted, columns=["a", "b"]))
    res = control.mewma_chart(ws, ["a", "b"])
    assert any(kind == "heading" for kind, _ in res.blocks)                     # the shift is signalled


def test_capability_between_within(ws, groups):
    res = quality.capability_between_within(ws, "x", 5, None, "90", "110")
    out = tables(res)
    parts = control.between_within(groups)
    assert out[0].loc["Between/Within", "StDev"] == pytest.approx(parts["total"])
    assert out[2].loc["Cp / Pp", "Potential (within)"] == pytest.approx(20 / (6 * parts["total"]))
    plain = tables(quality.capability(ws, "x", 5, None, "90", "110"))
    assert out[2].loc["Cp / Pp", "Potential (within)"] < plain[1].loc["Cp / Pp", "Potential (within)"]
    assert out[2].loc["Cp / Pp", "Overall"] == pytest.approx(plain[1].loc["Cp / Pp", "Overall"])
    many = tables(quality.capability_multiple(ws, ["x", "w"], 5, "5", "115"))[0]
    assert list(many.index) == ["x", "w"] and many.loc["x", "Pp"] == pytest.approx(110 / (6 * ws.df["x"].std()))


def test_tolerance_intervals_one_sided_and_lognormal(ws):
    # one-sided normal tolerance factors for 95% coverage with 95% confidence (standard tables)
    assert quality.one_sided_tolerance_factor(10, 0.95, 0.95) == pytest.approx(2.911, abs=1e-3)
    assert quality.one_sided_tolerance_factor(20, 0.95, 0.95) == pytest.approx(2.396, abs=1e-3)
    x = ws.df["x"].to_numpy()
    lower = tables(quality.tolerance_intervals(ws, ["x"], 95.0, 95.0, "Lower bound"))
    k = quality.one_sided_tolerance_factor(len(x), 0.95, 0.95)
    assert lower[0].loc["x", "Lower"] == pytest.approx(x.mean() - k * x.std(ddof=1)) and np.isinf(lower[0].loc["x", "Upper"])
    assert lower[1].loc["x", "Lower"] == x.min() and lower[1].loc["x", "Achieved confidence (%)"] == pytest.approx(
        100 * (1 - 0.95 ** len(x)))
    logs = Worksheet("L", pd.DataFrame({"x": np.log(x)}))
    normal = tables(quality.tolerance_intervals(logs, ["x"]))[0]
    lognormal = tables(quality.tolerance_intervals(ws, ["x"], 95.0, 95.0, "Two-sided", "Lognormal"))[0]
    assert lognormal.loc["x", "Lower"] == pytest.approx(np.exp(normal.loc["x", "Lower"]))
    assert lognormal.loc["x", "Upper"] == pytest.approx(np.exp(normal.loc["x", "Upper"]))
    with pytest.raises(ValueError):
        quality.tolerance_intervals(Worksheet("N", pd.DataFrame({"x": [-1.0, 2.0, 3.0]})), ["x"], 95.0, 95.0,
                                    "Two-sided", "Lognormal")


def test_acceptance_sampling_by_variables(ws):
    # Montgomery's example: AQL 1%, RQL 6%, alpha 0.05, beta 0.10 gives k near 1.9 and n near 40
    n, k = quality.variables_plan(0.01, 0.06)
    assert k == pytest.approx(1.893, abs=2e-3) and n == 41
    known_n, known_k = quality.variables_plan(0.01, 0.06, sigma_known=True)
    assert known_n == 15 and known_k == k
    for sigma_known, size in ((False, n), (True, known_n)):
        accept = quality.variables_acceptance([0.01, 0.06], size, k, sigma_known)
        assert accept[0] == pytest.approx(0.95, abs=0.02) and accept[1] == pytest.approx(0.10, abs=0.02)
    res = quality.acceptance_by_variables(ws, 1.0, 6.0, 0.05, 0.10, 0.0, "x", "90", "")
    lot = tables(res)[2].iloc[0]
    assert lot["Z.LSL"] == pytest.approx((ws.df["x"].mean() - 90) / ws.df["x"].std()) and "Z.USL" not in lot.index
    assert res.blocks[-1][1].endswith("accept the lot") == (lot["Z.LSL"] >= k)
    with pytest.raises(ValueError):
        quality.variables_plan(0.06, 0.01)


def test_multi_vari_symmetry_and_variability(ws):
    ws.df["shift"] = np.array(["d", "n"], dtype=object)[np.arange(len(ws.df)) % 2]
    res = quality.multi_vari(ws, "x", "shift", "part")
    table = tables(res)[0]
    assert len(table) == 6 and table["N"].sum() == len(ws.df)
    assert table.loc["a, d", "Mean"] == pytest.approx(ws.df["x"][(ws.df["part"] == "a") & (ws.df["shift"] == "d")].mean())
    assert len(tables(quality.multi_vari(ws, "x", "shift"))[0]) == 2
    with pytest.raises(ValueError):
        quality.multi_vari(ws, "x", "shift", "shift")

    below, above = quality.symmetry_points([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert below.tolist() == above.tolist() == [3.0, 2.0, 1.0]
    below, above = quality.symmetry_points([1.0, 2.0, 3.0, 10.0])
    assert below.tolist() == [1.5, 0.5] and above.tolist() == [7.5, 0.5]
    skewed = Worksheet("S", pd.DataFrame({"e": np.random.default_rng(2).exponential(size=200)}))
    assert tables(quality.symmetry_plot(skewed, ["e"]))[0].loc["e", "Points above the line (%)"] > 70

    table = tables(gage.variability_chart(ws, "x", ["part", "shift"]))[0]
    assert list(table.index[:2]) == ["a, d", "a, n"] and table.loc["a, d", "N"] > 0
    assert table.loc["c, n", "Maximum"] == ws.df["x"][(ws.df["part"] == "c") & (ws.df["shift"] == "n")].max()


def gage_data(extra_effect=0.0, seed=7):
    rng = np.random.default_rng(seed)
    idx = np.array(np.meshgrid(np.arange(6), np.arange(3), np.arange(2), np.arange(2), indexing="ij")).reshape(4, -1)
    part, oper, fixture = rng.normal(0, 3, 6), rng.normal(0, 0.5, 3), np.array([0.0, extra_effect])
    y = 20 + part[idx[0]] + oper[idx[1]] + fixture[idx[2]] + rng.normal(0, 0.3, idx.shape[1])
    return Worksheet("G", pd.DataFrame({
        "y": y, "part": np.array([f"p{i}" for i in idx[0]], dtype=object),
        "oper": np.array([f"o{i}" for i in idx[1]], dtype=object),
        "fixture": np.array([f"f{i}" for i in idx[2]], dtype=object)}))


def test_expanded_gage_study():
    ws = gage_data(extra_effect=2.0)
    out = tables(gage.gage_expanded(ws, "part", "oper", "y", ["fixture"], "30"))
    ref = anova.balanced_anova_table(ws.df, "y", ["part", "oper", "fixture"], ["part", "oper", "fixture"])
    components, evaluation = out[1], out[2]
    assert components.loc["Part-To-Part", "VarComp"] == pytest.approx(ref.loc["part", "Variance component"])
    assert components.loc["  Repeatability", "VarComp"] == pytest.approx(ref.loc["Error", "MS"])
    assert components.loc["    fixture", "VarComp"] == pytest.approx(ref.loc["fixture", "Variance component"])
    assert components.loc["Total Gage R&R", "VarComp"] == pytest.approx(
        components.loc["  Repeatability", "VarComp"] + components.loc["  Reproducibility", "VarComp"])
    assert components.loc["Total Variation", "% Contribution"] == pytest.approx(100.0)
    assert evaluation.loc["Total Gage R&R", "% Tolerance"] == pytest.approx(
        100 * 6 * np.sqrt(components.loc["Total Gage R&R", "VarComp"]) / 30)
    quiet = tables(gage.gage_expanded(gage_data(0.0), "part", "oper", "y", ["fixture"]))[1]
    assert quiet.loc["    fixture", "VarComp"] < components.loc["    fixture", "VarComp"]
    with pytest.raises(ValueError):
        gage.gage_expanded(ws, "part", "oper", "y", ["oper"])


def test_gage_run_chart_and_study_worksheet():
    ws = gage_data()
    res = gage.gage_run_chart(ws, "part", "oper", "y")
    assert len(res.figures()) == 1 and len(res.figures()[0].axes) == 10        # six parts on a 2 x 5 grid
    frame = gage.create_gage_worksheet(None, parts=4, operators=3, replicates=2, randomize=True, seed=1).worksheets[0][1]
    assert len(frame) == 24 and frame["RunOrder"].tolist() == list(range(1, 25)) and frame["Measurement"].isna().all()
    for block in (frame.iloc[:12], frame.iloc[12:]):                           # each replicate has every combination
        assert len(block[["Parts", "Operators"]].drop_duplicates()) == 12
    plain = gage.create_gage_worksheet(None, 2, 2, 1, randomize=False).worksheets[0][1]
    assert plain[["Parts", "Operators"]].to_numpy().tolist() == [[1, 1], [2, 1], [1, 2], [2, 2]]


def test_attribute_gage_study():
    # a gage that accepts half the time at 0.105 with a standard deviation of 0.004, against a lower limit of 0.100
    reference = np.round(np.arange(0.090, 0.1201, 0.002), 3)
    acceptances = np.round(20 * sps.norm.cdf((reference - 0.105) / 0.004))
    out = gage.attribute_gage(reference, acceptances, 20, 0.100)
    assert out["bias"] == pytest.approx(-0.005, abs=5e-4)
    # moving each share half a count towards 0.5 flattens the line, which is what the divisor 1.08 allows for
    assert out["repeatability"] == pytest.approx(2 * 2.5758 * 0.004, rel=0.15)
    assert out["t"] == pytest.approx(31.3 * abs(out["bias"]) / out["repeatability"]) and out["p"] < 0.05
    assert 2 <= out["used"].sum() < len(reference)                             # the extremes are left out of the fit
    ws = Worksheet("A", pd.DataFrame({"ref": reference, "acc": acceptances}))
    assert tables(gage.attribute_gage_study(ws, "ref", "acc", 20, 0.100))[1].iloc[0]["Bias"] == pytest.approx(out["bias"])
    for bad in ([25.0] * len(reference), [0.0] * len(reference)):
        with pytest.raises(ValueError):
            gage.attribute_gage(reference, bad, 20, 0.1)
