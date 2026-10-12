"""Design of experiments: design generation and analysis recovering known models."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats as sps

from openstat.core.worksheet import Worksheet
from openstat.stats import doe


def tables(res):
    return {payload[1]: payload[0] for kind, payload in res.blocks if kind == "table"}


def test_two_level_designs_are_orthogonal():
    full, aliases = doe.two_level_design(3)
    assert full.shape == (8, 3) and aliases == []
    half, aliases = doe.two_level_design(5, "1/2")
    assert half.shape == (16, 5) and aliases
    assert np.allclose(half.T @ half, 16 * np.eye(5))
    with pytest.raises(ValueError):
        doe.two_level_design(3, "1/4")


def test_finish_design_replicates_center_and_seed():
    matrix, _ = doe.two_level_design(2)
    a = doe.finish_design(matrix, "AB", replicates=2, center_points=3, randomize=True, seed=5)
    b = doe.finish_design(matrix, "AB", replicates=2, center_points=3, randomize=True, seed=5)
    assert a.equals(b) and len(a) == 11
    assert sorted(a["StdOrder"]) == list(range(1, 12)) and a["RunOrder"].tolist() == list(range(1, 12))
    assert (a["CenterPt"] == 0).sum() == 3
    assert (a.loc[a["CenterPt"] == 0, ["A", "B"]] == 0).all().all()
    plain = doe.finish_design(matrix, "AB", randomize=False)
    assert plain["StdOrder"].tolist() == [1, 2, 3, 4]


def test_other_design_generators():
    assert doe.create_plackett_burman(None, factors=7, randomize=False).worksheets[0][1].shape[0] == 8
    general = doe.create_general_factorial(None, "3, 2, 2", randomize=False).worksheets[0][1]
    assert len(general) == 12 and sorted(general["A"].unique()) == [1, 2, 3]
    ccd = doe.create_ccd(None, factors=2, center_points=5, randomize=False).worksheets[0][1]
    assert len(ccd) == 13 and ccd["A"].abs().max() == pytest.approx(np.sqrt(2))
    faced = doe.create_ccd(None, factors=3, kind=doe.CCD_TYPES[1], center_points=2, randomize=False).worksheets[0][1]
    assert faced[["A", "B", "C"]].abs().max().max() == 1.0 and len(faced) == 16
    bb = doe.create_box_behnken(None, factors=3, center_points=3, randomize=False).worksheets[0][1]
    assert len(bb) == 15
    assert doe.simplex_lattice(3, 2).shape == (6, 3) and np.allclose(doe.simplex_lattice(3, 2).sum(axis=1), 1)
    assert doe.simplex_centroid(3).shape == (7, 3)
    taguchi = doe.create_taguchi(None, "L9(3^4)", 4).worksheets[0][1]
    assert taguchi.shape == (9, 4) and sorted(taguchi["A"].unique()) == [1, 2, 3]
    with pytest.raises(ValueError):
        doe.parse_levels("3, 1")


def factorial_ws(replicates, noise, seed=0):
    rng = np.random.default_rng(seed)
    matrix, _ = doe.two_level_design(3)
    frame = doe.finish_design(matrix, "ABC", replicates=replicates, randomize=False)
    frame["y"] = 10 + 3 * frame["A"] - 2 * frame["B"] + 1.5 * frame["A"] * frame["B"] + rng.normal(0, noise, len(frame))
    frame["Temp"] = np.where(frame["A"] > 0, 180.0, 160.0)          # uncoded numeric factor
    frame["Cat"] = np.where(frame["B"] > 0, "new", "old").astype(object)  # text factor
    return Worksheet("F", frame)


def test_analyze_factorial_recovers_effects():
    ws = factorial_ws(replicates=2, noise=0.2)
    res = doe.analyze_factorial(ws, "y", ["Temp", "Cat", "C"], doe.ORDERS[3])
    coefs = tables(res)["Coded Coefficients"]
    assert coefs.loc["Temp", "Effect"] == pytest.approx(6.0, abs=0.4)
    assert coefs.loc["Cat", "Effect"] == pytest.approx(4.0, abs=0.4)      # 'new' sorts first, so the sign flips
    assert coefs.loc["Temp*Cat", "Effect"] == pytest.approx(-3.0, abs=0.4)
    assert coefs.loc["C", "P-Value"] > 0.01 and coefs.loc["Temp", "P-Value"] < 0.001
    anova = tables(res)["Analysis of Variance"]
    assert anova.loc["Main Effects", "DF"] == 3 and anova.loc["Error", "DF"] == 8
    ref = sm.OLS(ws.df["y"], sm.add_constant(ws.df[["A", "B", "C"]].assign(
        AB=ws.df["A"] * ws.df["B"], AC=ws.df["A"] * ws.df["C"], BC=ws.df["B"] * ws.df["C"],
        ABC=ws.df["A"] * ws.df["B"] * ws.df["C"]))).fit()
    assert coefs.loc["Temp", "Coef"] == pytest.approx(ref.params["A"])


def test_analyze_factorial_saturated_uses_lenth():
    ws = factorial_ws(replicates=1, noise=0.05)
    res = doe.analyze_factorial(ws, "y", ["A", "B", "C"], doe.ORDERS[3])
    assert "Model Summary" not in tables(res)
    effects = tables(res)["Coded Coefficients"]["Effect"].dropna().to_numpy()
    pse, margin = doe.lenth(effects)
    assert (np.abs(effects) > margin).sum() == 3
    magnitude = np.abs(effects)
    s0 = 1.5 * np.median(magnitude)
    assert pse == pytest.approx(1.5 * np.median(magnitude[magnitude < 2.5 * s0]))


def test_aliased_terms_are_dropped_and_bad_factors_rejected():
    matrix, _ = doe.two_level_design(4, "1/2")
    frame = doe.finish_design(matrix, "ABCD", randomize=False)
    frame["y"] = 5 + frame["A"] + np.random.default_rng(1).normal(0, 0.1, len(frame))
    X, groups, dropped = doe.factorial_matrix(frame[list("ABCD")], 2)
    assert X.shape[1] == 8 and len(dropped) == 3      # 2-way interactions are aliased in pairs
    with pytest.raises(ValueError):
        doe.code_two_level(pd.Series([1.0, 2.0, 3.0, 4.0]), "x")
    with pytest.raises(ValueError):
        doe.code_two_level(pd.Series(["a", "b", "c"], dtype=object), "x")


def rsm_frame(noise=0.0):
    design = doe.create_ccd(None, factors=2, center_points=5, randomize=False).worksheets[0][1]
    rng = np.random.default_rng(3)
    design["y"] = 50 - (design["A"] - 0.3) ** 2 - 2 * (design["B"] + 0.2) ** 2 + rng.normal(0, noise, len(design))
    return design


def test_response_surface_stationary_point():
    frame = rsm_frame()
    fit, _, _ = doe.fit_surface(frame, "y", ["A", "B"])
    out = doe.stationary_point(fit, ["A", "B"])
    assert out["kind"] == "Maximum"
    assert out["point"] == pytest.approx([0.3, -0.2], abs=1e-8)
    assert out["value"] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        doe.fit_surface(frame.iloc[:5], "y", ["A", "B"])


def test_response_surface_report_has_lack_of_fit():
    res = doe.analyze_response_surface(Worksheet("R", rsm_frame(noise=0.05)), "y", ["A", "B"])
    anova = tables(res)["Analysis of Variance"]
    assert anova.loc["  Pure Error", "DF"] == 4 and anova.loc["  Lack-of-Fit", "DF"] == 3
    assert anova.loc["  Lack-of-Fit", "Adj SS"] + anova.loc["  Pure Error", "Adj SS"] == \
        pytest.approx(anova.loc["Error", "Adj SS"])
    assert anova.loc["Square", "P-Value"] < 0.001


def test_desirability_and_optimizer():
    assert doe.desirability(5, 0, 10, "Maximize") == pytest.approx(0.5)
    assert doe.desirability(5, 0, 10, "Minimize") == pytest.approx(0.5)
    assert doe.desirability(12, 0, 10, "Maximize") == 1.0
    assert doe.desirability(4, 0, 10, "Target", 8) == pytest.approx(0.5)
    assert doe.desirability(9, 0, 10, "Target", 8) == pytest.approx(0.5)
    frame = rsm_frame()
    best = doe.optimize_responses(frame, ["y"], ["A", "B"], "Maximize")
    assert best["x"] == pytest.approx([0.3, -0.2], abs=1e-3)
    worst = doe.optimize_responses(frame, ["y"], ["A", "B"], "Minimize")
    assert worst["predicted"]["y"] <= frame["y"].min() + 1e-6
    hit = doe.optimize_responses(frame, ["y"], ["A", "B"], "Target", target=48.0)
    assert hit["predicted"]["y"] == pytest.approx(48.0, abs=0.05)


def test_mixture_model_recovers_coefficients():
    points = np.vstack([doe.simplex_lattice(3, 2), doe.simplex_centroid(3)[-1:], doe.simplex_lattice(3, 3)])
    frame = pd.DataFrame(points * 100, columns=list("ABC"))   # amounts that sum to 100
    p = points
    frame["y"] = 10 * p[:, 0] + 20 * p[:, 1] + 30 * p[:, 2] + 8 * p[:, 0] * p[:, 1]
    ws = Worksheet("M", frame)
    coefs = tables(doe.analyze_mixture(ws, "y", ["A", "B", "C"], "Quadratic"))["Coefficients"]
    assert coefs["Coef"].to_numpy() == pytest.approx([10, 20, 30, 8, 0, 0], abs=1e-8)
    assert np.isnan(coefs.loc["A", "P-Value"])
    ws.df.loc[0, "A"] = 50.0
    with pytest.raises(ValueError):
        doe.analyze_mixture(ws, "y", ["A", "B", "C"])


def test_signal_to_noise_and_response_table():
    values = np.array([[10.0, 12.0], [20.0, 20.0]])
    assert doe.signal_to_noise(values, doe.SN_RATIOS[0])[0] == pytest.approx(
        -10 * np.log10((1 / 100 + 1 / 144) / 2))
    assert doe.signal_to_noise(values, doe.SN_RATIOS[1])[1] == pytest.approx(-10 * np.log10(400))
    assert doe.signal_to_noise(values, doe.SN_RATIOS[2])[0] == pytest.approx(10 * np.log10(121 / 2))
    factors = pd.DataFrame({"A": [1.0, 1, 2, 2], "B": [1.0, 2, 1, 2]})
    table = doe.response_table(pd.Series([1.0, 3.0, 10.0, 12.0]), factors)
    assert table.loc["Delta"].tolist() == [9.0, 2.0]
    assert table.iloc[-1].tolist() == [1.0, 2.0]


def test_group_tests_match_nested_model_f_test():
    frame = rsm_frame(noise=0.3)
    fit, X, groups = doe.fit_surface(frame, "y", ["A", "B"])
    anova = doe.group_tests(fit, list(X.columns), groups)
    reduced = sm.OLS(frame["y"], X.drop(columns=groups["Square"])).fit()
    f = ((reduced.ssr - fit.ssr) / 2) / fit.mse_resid
    assert anova.loc["Square", "F-Value"] == pytest.approx(f)
    assert anova.loc["Square", "P-Value"] == pytest.approx(sps.f.sf(f, 2, fit.df_resid))
