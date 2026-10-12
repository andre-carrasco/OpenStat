"""Design of experiments: creating designs and analysing factorial, response surface, mixture and Taguchi data."""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd
import pyDOE3
import statsmodels.api as sm
from scipy import optimize
from scipy import stats as sps

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._design import CONSTANT, Design
from openstat.stats._util import median_ranks
from openstat.stats.regression import coefficient_table, model_summary, residual_plots

MENU = ("Stat", "DOE")
FACTORIAL, RSM, MIXTURE, TAGUCHI = MENU + ("Factorial",), MENU + ("Response Surface",), MENU + ("Mixture",), \
    MENU + ("Taguchi",)
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # I is skipped, as is conventional for factor names

REPLICATES = Param("replicates", "Number of replicates", "int", default=1, minimum=1, maximum=100)
CENTER = Param("center_points", "Number of center points", "int", default=0, minimum=0, maximum=100)
RANDOMIZE = Param("randomize", "Randomize runs", "bool", default=True)
SEED = Param("seed", "Base for random numbers (0 = random)", "int", default=0, minimum=0)
RESPONSE = Param("response", "Response", "column")
FACTOR_COLUMNS = Param("factors", "Factors", "columns", numeric=False)
CONTINUOUS_FACTORS = Param("factors", "Factors", "columns")


# ----- creating designs -----

def finish_design(matrix, names, replicates=1, center_points=0, randomize=True, seed=0) -> pd.DataFrame:
    """Replicate, add center points and randomize; returns the design worksheet."""
    matrix = np.asarray(matrix, dtype=float)
    runs = np.tile(matrix, (replicates, 1))
    center = np.zeros(len(runs))
    if center_points:
        runs = np.vstack([runs, np.zeros((center_points, matrix.shape[1]))])
        center = np.concatenate([center, np.ones(center_points)])
    n = len(runs)
    order = np.arange(n)
    if randomize:
        order = np.random.default_rng(seed or None).permutation(n)
    frame = pd.DataFrame(runs, columns=list(names))
    frame.insert(0, "CenterPt", 1 - center)   # 1 = factorial point, 0 = center point
    frame.insert(0, "StdOrder", np.arange(1, n + 1, dtype=float))
    frame = frame.iloc[order].reset_index(drop=True)
    frame.insert(1, "RunOrder", np.arange(1, n + 1, dtype=float))
    return frame


def _design_result(title: str, frame: pd.DataFrame, summary: dict, notes=()) -> Result:
    res = Result(title)
    res.table(pd.DataFrame([summary]), title="Design Summary", index=False, digits=0)
    for note in notes:
        res.text(note)
    res.table(frame.head(64), title="Design Table (first runs)", index=False, digits=3)
    res.worksheet(tr(title), frame)
    return res


FRACTIONS = ["Full", "1/2", "1/4", "1/8", "1/16"]


def two_level_design(factors: int, fraction: str = "Full") -> tuple[np.ndarray, list[str]]:
    """Coded two-level design and its alias structure (empty for a full factorial)."""
    erased = FRACTIONS.index(fraction)
    if erased == 0:
        return pyDOE3.ff2n(factors), []
    if factors - erased < 2 or 2 ** (factors - erased) <= factors:
        raise ValueError("The fraction is too small for this number of factors")
    generator, aliases, _ = pyDOE3.fracfact_opt(factors, erased)
    return pyDOE3.fracfact(generator), [a.upper() for a in aliases]


@analysis("doe.create_factorial", menu=FACTORIAL, title="Create Two-Level Factorial Design",
          params=[Param("factors", "Number of factors", "int", default=3, minimum=2, maximum=12),
                  Param("fraction", "Fraction", "choice", default="Full", choices=FRACTIONS),
                  REPLICATES, CENTER, RANDOMIZE, SEED])
def create_factorial(ws, factors=3, fraction="Full", replicates=1, center_points=0, randomize=True, seed=0):
    matrix, aliases = two_level_design(factors, fraction)
    frame = finish_design(matrix, LETTERS[:factors], replicates, center_points, randomize, seed)
    notes = []
    if aliases:
        notes.append(f"{tr('Alias structure')}: " + ";  ".join(aliases[:40]))
    return _design_result("Create Two-Level Factorial Design", frame, {
        "Factors": factors, "Base runs": len(matrix), "Replicates": replicates, "Center points": center_points,
        "Total runs": len(frame)}, notes)


@analysis("doe.create_pb", menu=FACTORIAL, title="Create Plackett-Burman Design",
          params=[Param("factors", "Number of factors", "int", default=7, minimum=2, maximum=47),
                  REPLICATES, CENTER, RANDOMIZE, SEED])
def create_plackett_burman(ws, factors=7, replicates=1, center_points=0, randomize=True, seed=0):
    matrix = pyDOE3.pbdesign(factors)[:, :factors]
    frame = finish_design(matrix, LETTERS[:factors] if factors <= len(LETTERS) else
                          [f"X{i + 1}" for i in range(factors)], replicates, center_points, randomize, seed)
    return _design_result("Create Plackett-Burman Design", frame, {
        "Factors": factors, "Base runs": len(matrix), "Replicates": replicates, "Center points": center_points,
        "Total runs": len(frame)})


def parse_levels(text: str) -> list[int]:
    try:
        levels = [int(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError("Enter the number of levels of each factor, separated by commas") from exc
    if len(levels) < 1 or any(v < 2 for v in levels):
        raise ValueError("Each factor needs at least two levels")
    if np.prod(levels) > 100000:
        raise ValueError("The design is too large")
    return levels


@analysis("doe.create_general", menu=FACTORIAL, title="Create General Full Factorial Design",
          params=[Param("levels", "Levels of each factor (e.g. 3, 2, 4)", "text", default="3, 2"),
                  REPLICATES, RANDOMIZE, SEED])
def create_general_factorial(ws, levels="3, 2", replicates=1, randomize=True, seed=0):
    counts = parse_levels(levels)
    matrix = pyDOE3.fullfact(counts) + 1
    frame = finish_design(matrix, LETTERS[:len(counts)], replicates, 0, randomize, seed).drop(columns="CenterPt")
    return _design_result("Create General Full Factorial Design", frame, {
        "Factors": len(counts), "Base runs": len(matrix), "Replicates": replicates, "Total runs": len(frame)})


CCD_TYPES = ["Rotatable (circumscribed)", "Face-centered", "Inscribed"]


@analysis("doe.create_ccd", menu=RSM, title="Create Central Composite Design",
          params=[Param("factors", "Number of factors", "int", default=2, minimum=2, maximum=8),
                  Param("kind", "Type", "choice", default=CCD_TYPES[0], choices=CCD_TYPES),
                  Param("center_points", "Number of center points", "int", default=5, minimum=1, maximum=50),
                  RANDOMIZE, SEED])
def create_ccd(ws, factors=2, kind=CCD_TYPES[0], center_points=5, randomize=True, seed=0):
    face = ["circumscribed", "faced", "inscribed"][CCD_TYPES.index(kind)]
    matrix = pyDOE3.ccdesign(factors, center=(0, center_points), alpha="rotatable", face=face)
    is_center = np.all(matrix == 0, axis=1)
    frame = finish_design(matrix[~is_center], LETTERS[:factors], 1, int(is_center.sum()), randomize, seed)
    alpha = float(np.abs(matrix).max())
    return _design_result("Create Central Composite Design", frame, {
        "Factors": factors, "Cube points": 2 ** factors, "Axial points": 2 * factors,
        "Center points": int(is_center.sum()), "Total runs": len(frame)}, [f"Alpha = {alpha:.5g}"])


@analysis("doe.create_bb", menu=RSM, title="Create Box-Behnken Design",
          params=[Param("factors", "Number of factors", "int", default=3, minimum=3, maximum=10),
                  Param("center_points", "Number of center points", "int", default=3, minimum=1, maximum=50),
                  RANDOMIZE, SEED])
def create_box_behnken(ws, factors=3, center_points=3, randomize=True, seed=0):
    matrix = pyDOE3.bbdesign(factors, center=center_points)
    is_center = np.all(matrix == 0, axis=1)
    frame = finish_design(matrix[~is_center], LETTERS[:factors], 1, int(is_center.sum()), randomize, seed)
    return _design_result("Create Box-Behnken Design", frame, {
        "Factors": factors, "Center points": int(is_center.sum()), "Total runs": len(frame)})


def simplex_lattice(components: int, degree: int) -> np.ndarray:
    points = [np.array(c) / degree for c in product(range(degree + 1), repeat=components) if sum(c) == degree]
    return np.array(sorted(points, key=lambda p: tuple(-p)))


def simplex_centroid(components: int) -> np.ndarray:
    points = []
    for size in range(1, components + 1):
        for subset in combinations(range(components), size):
            point = np.zeros(components)
            point[list(subset)] = 1 / size
            points.append(point)
    return np.array(points)


MIXTURE_TYPES = ["Simplex lattice", "Simplex centroid"]


@analysis("doe.create_mixture", menu=MIXTURE, title="Create Mixture Design",
          params=[Param("components", "Number of components", "int", default=3, minimum=2, maximum=10),
                  Param("kind", "Type", "choice", default=MIXTURE_TYPES[0], choices=MIXTURE_TYPES),
                  Param("degree", "Degree of lattice", "int", default=2, minimum=1, maximum=6),
                  REPLICATES, RANDOMIZE, SEED])
def create_mixture(ws, components=3, kind=MIXTURE_TYPES[0], degree=2, replicates=1, randomize=True, seed=0):
    matrix = simplex_lattice(components, degree) if kind == MIXTURE_TYPES[0] else simplex_centroid(components)
    frame = finish_design(matrix, LETTERS[:components], replicates, 0, randomize, seed).drop(columns="CenterPt")
    return _design_result("Create Mixture Design", frame, {
        "Components": components, "Design points": len(matrix), "Replicates": replicates,
        "Total runs": len(frame)}, [tr("Components are proportions that sum to 1.")])


ARRAYS = pyDOE3.list_orthogonal_arrays()


@analysis("doe.create_taguchi", menu=TAGUCHI, title="Create Taguchi Design",
          params=[Param("array", "Orthogonal array", "choice", default="L8(2^7)", choices=ARRAYS),
                  Param("factors", "Number of factors", "int", default=3, minimum=1, maximum=63)])
def create_taguchi(ws, array="L8(2^7)", factors=3):
    matrix = pyDOE3.get_orthogonal_array(array) + 1
    if factors > matrix.shape[1]:
        raise ValueError(f"{array}: this array has only {matrix.shape[1]} columns")
    names = LETTERS[:factors] if factors <= len(LETTERS) else [f"X{i + 1}" for i in range(factors)]
    frame = pd.DataFrame(matrix[:, :factors].astype(float), columns=list(names))
    return _design_result("Create Taguchi Design", frame, {"Factors": factors, "Total runs": len(frame)}, [array])


# ----- shared model helpers -----

def group_tests(fit, columns: list[str], groups: dict[str, list[str]]) -> pd.DataFrame:
    """ANOVA rows testing each group of model columns jointly, plus error and total."""
    rows = {}
    mse = fit.mse_resid
    for name, cols in groups.items():
        if not cols:
            continue
        restriction = np.zeros((len(cols), len(columns)))
        for i, col in enumerate(cols):
            restriction[i, columns.index(col)] = 1.0
        test = fit.f_test(restriction)
        f = float(np.squeeze(test.fvalue))
        rows[name] = {"DF": len(cols), "Adj SS": f * len(cols) * mse, "Adj MS": f * mse, "F-Value": f,
                      "P-Value": float(test.pvalue)}
    table = pd.DataFrame.from_dict(rows, orient="index")
    head = pd.DataFrame([{"DF": fit.df_model, "Adj SS": fit.ess, "Adj MS": fit.mse_model, "F-Value": fit.fvalue,
                          "P-Value": fit.f_pvalue}], index=["Model"])
    tail = pd.DataFrame([{"DF": fit.df_resid, "Adj SS": fit.ssr, "Adj MS": mse},
                         {"DF": fit.df_model + fit.df_resid, "Adj SS": fit.centered_tss}], index=["Error", "Total"])
    out = pd.concat([head, table, tail])
    out.index.name = "Source"
    return out


def lack_of_fit(fit, factor_frame: pd.DataFrame, y: np.ndarray) -> dict | None:
    """Split the error into lack of fit and pure error when there are replicated points."""
    keys = factor_frame.round(8).astype(str).agg("|".join, axis=1)
    groups = pd.Series(y).groupby(keys.to_numpy())
    pure_ss = float(groups.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum())
    pure_df = len(y) - groups.ngroups
    lof_df = int(fit.df_resid) - pure_df
    if pure_df < 1 or lof_df < 1:
        return None
    lof_ss = max(fit.ssr - pure_ss, 0.0)
    f = (lof_ss / lof_df) / (pure_ss / pure_df) if pure_ss > 0 else np.inf
    return {"lof_df": lof_df, "lof_ss": lof_ss, "pure_df": pure_df, "pure_ss": pure_ss, "f": f,
            "p": float(sps.f.sf(f, lof_df, pure_df))}


# ----- two-level factorial analysis -----

def code_two_level(series: pd.Series, name: str) -> np.ndarray:
    """Code a two-level factor to -1/+1; numeric factors may also have center points at 0."""
    if pd.api.types.is_numeric_dtype(series):
        x = series.to_numpy(dtype=float)
        low, high = x.min(), x.max()
        if low == high:
            raise ValueError(f"{name}: a factor needs at least two levels")
        coded = (x - (low + high) / 2) / ((high - low) / 2)
        if not np.all(np.isclose(np.abs(coded), 1) | np.isclose(coded, 0)):
            raise ValueError(f"{name}: a two-level factor may only have a low level, a high level and center points")
        return np.round(coded, 10)
    labels = series.map(format_value)
    levels = sorted(labels.unique())
    if len(levels) != 2:
        raise ValueError(f"{name}: a text factor must have exactly two levels")
    return np.where(labels == levels[0], -1.0, 1.0)


def factorial_matrix(coded: pd.DataFrame, order: int) -> tuple[pd.DataFrame, dict, list[str]]:
    """Model matrix with all interactions up to `order`; aliased or inestimable terms are dropped."""
    names = list(coded.columns)
    n = len(coded)
    X = pd.DataFrame({CONSTANT: np.ones(n)}, index=coded.index)
    groups: dict[str, list[str]] = {}
    dropped: list[str] = []
    for size in range(1, min(order, len(names)) + 1):
        label = "Main Effects" if size == 1 else f"{size}-Way Interactions"
        for combo in combinations(names, size):
            term = "*".join(combo)
            column = coded[list(combo)].prod(axis=1)
            trial = pd.concat([X, column.rename(term)], axis=1)
            if trial.shape[1] <= n and np.linalg.matrix_rank(trial.to_numpy()) == trial.shape[1]:
                X = trial
                groups.setdefault(label, []).append(term)
            else:
                dropped.append(term)
    center = np.all(coded.to_numpy() == 0, axis=1)
    if center.any():
        trial = pd.concat([X, pd.Series(center.astype(float), index=coded.index, name="Ct Pt")], axis=1)
        if trial.shape[1] <= n and np.linalg.matrix_rank(trial.to_numpy()) == trial.shape[1]:
            X = trial
            groups["Curvature"] = ["Ct Pt"]
    return X, groups, dropped


def lenth(effects: np.ndarray, alpha=0.05) -> tuple[float, float]:
    """Lenth's pseudo standard error and margin of error for unreplicated designs."""
    magnitude = np.abs(np.asarray(effects, dtype=float))
    s0 = 1.5 * np.median(magnitude)
    pse = 1.5 * np.median(magnitude[magnitude < 2.5 * s0]) if s0 > 0 else 0.0
    return float(pse), float(sps.t.ppf(1 - alpha / 2, len(magnitude) / 3) * pse)


def _effects_figure(labels, values, limit, xlabel: str, title: str, effects):
    fig = new_figure(9.0, max(3.6, 0.32 * len(labels) + 1.6))
    left, right = fig.subplots(1, 2)
    order = np.argsort(values)
    colors = [RED if values[i] > limit else BLUE for i in order]
    left.barh([labels[i] for i in order], values[order], color=colors)
    left.axvline(limit, color=RED, linestyle="--", linewidth=1)
    left.set_xlabel(xlabel), left.set_title(tr("Pareto Chart of the Effects"), fontsize=9)
    rank = np.argsort(effects)
    scores = sps.norm.ppf(median_ranks(len(effects)))
    right.plot(effects[rank], scores, "o", color=BLUE)
    for i, r in enumerate(rank):
        if values[r] > limit:
            right.annotate(labels[r], (effects[r], scores[i]), fontsize=7, xytext=(4, 0), textcoords="offset points")
    right.set_xlabel(tr("Effect")), right.set_ylabel(tr("Normal score"))
    right.set_title(tr("Normal Plot of the Effects"), fontsize=9)
    fig.suptitle(title)
    return fig


ORDERS = ["Main effects", "Up to 2-way interactions", "Up to 3-way interactions", "All interactions"]


@analysis("doe.analyze_factorial", menu=FACTORIAL, title="Analyze Factorial Design",
          params=[RESPONSE, FACTOR_COLUMNS,
                  Param("order", "Terms in the model", "choice", default=ORDERS[1], choices=ORDERS),
                  Param("alpha", "Significance level", "number", default=0.05, minimum=0.0001, maximum=0.5)])
def analyze_factorial(ws, response, factors, order=ORDERS[1], alpha=0.05):
    factors = [f for f in factors if f != response]
    if not factors:
        raise ValueError("Select at least one factor")
    frame = ws.df[[response] + factors].dropna()
    y = frame[response].to_numpy(dtype=float)
    coded = pd.DataFrame({f: code_two_level(frame[f], f) for f in factors}, index=frame.index)
    max_order = len(factors) if order == ORDERS[3] else ORDERS.index(order) + 1
    X, groups, dropped = factorial_matrix(coded, max_order)
    fit = sm.OLS(y, X).fit()
    terms = [c for c in X.columns if c not in (CONSTANT, "Ct Pt")]
    effects = 2 * fit.params[terms].to_numpy()
    res = Result("Analyze Factorial Design")
    res.text(f"{tr('Response')}: {response}    {tr('Runs')}: {len(y)}    {tr('Terms')}: {len(terms)}")
    if dropped:
        res.text(f"{tr('Terms that cannot be estimated (aliased) were removed')}: {', '.join(dropped)}")
    saturated = fit.df_resid < 1
    coefs = pd.DataFrame({"Effect": 2 * fit.params, "Coef": fit.params}, index=X.columns)
    coefs.loc[[c for c in (CONSTANT, "Ct Pt") if c in coefs.index], "Effect"] = np.nan
    if saturated:
        pse, margin = lenth(effects, alpha)
        coefs.index.name = "Term"
        res.table(coefs, title="Coded Coefficients")
        res.text(f"{tr('No degrees of freedom for error; significance is judged by the method of Lenth.')}  "
                 f"PSE = {pse:.5g}, ME = {margin:.5g}")
        res.figure(_effects_figure(terms, np.abs(effects), margin, tr("Effect"), response, effects))
        return res
    coefs["SE Coef"], coefs["T-Value"], coefs["P-Value"] = fit.bse, fit.tvalues, fit.pvalues
    coefs.index.name = "Term"
    res.table(coefs, title="Coded Coefficients")
    res.table(pd.DataFrame([model_summary(fit)]), title="Model Summary", index=False)
    res.table(group_tests(fit, list(X.columns), groups), title="Analysis of Variance")
    critical = sps.t.ppf(1 - alpha / 2, fit.df_resid)
    res.figure(_effects_figure(terms, np.abs(fit.tvalues[terms].to_numpy()), critical,
                               tr("Standardized effect"), response, effects))
    res.figure(residual_plots(fit.fittedvalues, fit.resid, response))
    return res


# ----- response surface -----

RSM_MODELS = ["Full quadratic", "Linear", "Linear + interactions", "Linear + squares"]


def quadratic_matrix(factors: pd.DataFrame, model: str = RSM_MODELS[0]) -> tuple[pd.DataFrame, dict]:
    X = pd.DataFrame({CONSTANT: np.ones(len(factors))}, index=factors.index)
    groups = {"Linear": [], "Square": [], "2-Way Interactions": []}
    for name in factors.columns:
        X[name] = factors[name].astype(float)
        groups["Linear"].append(name)
    if model in (RSM_MODELS[0], RSM_MODELS[3]):
        for name in factors.columns:
            X[f"{name}*{name}"] = factors[name].astype(float) ** 2
            groups["Square"].append(f"{name}*{name}")
    if model in (RSM_MODELS[0], RSM_MODELS[2]):
        for a, b in combinations(factors.columns, 2):
            X[f"{a}*{b}"] = factors[a].astype(float) * factors[b].astype(float)
            groups["2-Way Interactions"].append(f"{a}*{b}")
    return X, groups


def fit_surface(frame: pd.DataFrame, response: str, factors: list[str], model: str = RSM_MODELS[0]):
    X, groups = quadratic_matrix(frame[factors], model)
    if len(frame) <= X.shape[1] or np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("The design does not support this model; choose a simpler model or add runs")
    return sm.OLS(frame[response].to_numpy(dtype=float), X).fit(), X, groups


def stationary_point(fit, factors: list[str]) -> dict | None:
    k = len(factors)
    b = np.array([fit.params.get(f, 0.0) for f in factors])
    B = np.zeros((k, k))
    for i, fi in enumerate(factors):
        B[i, i] = fit.params.get(f"{fi}*{fi}", 0.0)
        for j in range(i + 1, k):
            B[i, j] = B[j, i] = fit.params.get(f"{fi}*{factors[j]}", 0.0) / 2
    if abs(np.linalg.det(B)) < 1e-12:
        return None
    point = -0.5 * np.linalg.solve(B, b)
    eigen = np.linalg.eigvalsh(B)
    kind = "Maximum" if (eigen < 0).all() else "Minimum" if (eigen > 0).all() else "Saddle point"
    value = fit.params[CONSTANT] + b @ point + point @ B @ point
    return {"point": point, "kind": kind, "value": float(value)}


def surface_figure(fit, frame, factors, model, response, x_name, y_name):
    grid_x = np.linspace(frame[x_name].min(), frame[x_name].max(), 60)
    grid_y = np.linspace(frame[y_name].min(), frame[y_name].max(), 60)
    gx, gy = np.meshgrid(grid_x, grid_y)
    points = pd.DataFrame({f: np.full(gx.size, frame[f].astype(float).mean()) for f in factors})
    points[x_name], points[y_name] = gx.ravel(), gy.ravel()
    z = np.asarray(fit.predict(quadratic_matrix(points[factors], model)[0])).reshape(gx.shape)
    fig = new_figure(10.0, 4.4)
    left = fig.add_subplot(1, 2, 1)
    filled = left.contourf(gx, gy, z, levels=12, cmap="viridis")
    fig.colorbar(filled, ax=left, label=response)
    left.plot(frame[x_name], frame[y_name], "o", color="white", markeredgecolor="black", markersize=4)
    left.set_xlabel(x_name), left.set_ylabel(y_name), left.set_title(tr("Contour Plot"), fontsize=9)
    right = fig.add_subplot(1, 2, 2, projection="3d")
    right.plot_surface(gx, gy, z, cmap="viridis", linewidth=0)
    right.set_xlabel(x_name), right.set_ylabel(y_name), right.set_zlabel(response)
    right.set_title(tr("Surface Plot"), fontsize=9)
    return fig


@analysis("doe.analyze_rsm", menu=RSM, title="Analyze Response Surface Design",
          params=[RESPONSE, CONTINUOUS_FACTORS,
                  Param("model", "Terms in the model", "choice", default=RSM_MODELS[0], choices=RSM_MODELS)])
def analyze_response_surface(ws, response, factors, model=RSM_MODELS[0]):
    factors = [f for f in factors if f != response]
    if not factors:
        raise ValueError("Select at least one factor")
    frame = ws.df[[response] + factors].dropna().astype(float)
    fit, X, groups = fit_surface(frame, response, factors, model)
    res = Result("Analyze Response Surface Design")
    design = Design(X, [(c, [c]) for c in X.columns if c != CONSTANT], np.arange(len(frame)), len(frame))
    res.table(coefficient_table(fit, design), title="Coefficients")
    res.table(pd.DataFrame([model_summary(fit)]), title="Model Summary", index=False)
    anova = group_tests(fit, list(X.columns), groups)
    lof = lack_of_fit(fit, frame[factors], frame[response].to_numpy())
    if lof:
        extra = pd.DataFrame([{"DF": lof["lof_df"], "Adj SS": lof["lof_ss"], "Adj MS": lof["lof_ss"] / lof["lof_df"],
                               "F-Value": lof["f"], "P-Value": lof["p"]},
                              {"DF": lof["pure_df"], "Adj SS": lof["pure_ss"],
                               "Adj MS": lof["pure_ss"] / lof["pure_df"]}], index=["  Lack-of-Fit", "  Pure Error"])
        anova = pd.concat([anova.iloc[:-1], extra, anova.iloc[-1:]])
        anova.index.name = "Source"
    res.table(anova, title="Analysis of Variance")
    if model in (RSM_MODELS[0], RSM_MODELS[3]):
        stationary = stationary_point(fit, factors)
        if stationary:
            row = dict(zip(factors, stationary["point"]))
            row[tr("Predicted response")] = stationary["value"]
            res.table(pd.DataFrame([row]), title=f"{tr('Stationary point')}: {tr(stationary['kind'])}", index=False)
    if len(factors) >= 2:
        res.figure(surface_figure(fit, frame, factors, model, response, factors[0], factors[1]))
        if len(factors) > 2:
            res.text(tr("Plots show the first two factors; the others are held at their mean."))
    res.figure(residual_plots(fit.fittedvalues, fit.resid, response))
    return res


GOALS = ["Maximize", "Minimize", "Target"]


def desirability(value, low, high, goal: str, target=None):
    """Derringer-Suich desirability with linear weights."""
    if high <= low:
        return np.ones_like(np.asarray(value, dtype=float))
    value = np.asarray(value, dtype=float)
    if goal == "Maximize":
        return np.clip((value - low) / (high - low), 0, 1)
    if goal == "Minimize":
        return np.clip((high - value) / (high - low), 0, 1)
    target = np.clip(target, low, high)
    up = (value - low) / (target - low) if target > low else np.ones_like(value)
    down = (high - value) / (high - target) if high > target else np.ones_like(value)
    return np.clip(np.where(value <= target, up, down), 0, 1)


def optimize_responses(frame, responses, factors, goal, target=None, model=RSM_MODELS[0], seed=0) -> dict:
    fits = {r: fit_surface(frame[[r] + factors].dropna(), r, factors, model)[0] for r in responses}
    bounds = [(frame[f].min(), frame[f].max()) for f in factors]
    ranges = {r: (frame[r].min(), frame[r].max()) for r in responses}

    def predict(x):
        point = pd.DataFrame([x], columns=factors)
        matrix = quadratic_matrix(point, model)[0]
        return {r: float(np.asarray(fit.predict(matrix))[0]) for r, fit in fits.items()}

    def composite(x):
        predicted = predict(x)
        values = [float(desirability(predicted[r], *ranges[r], goal, target)) for r in responses]
        return float(np.prod(values) ** (1 / len(values)))

    rng = np.random.default_rng(seed)
    starts = [np.array([np.mean(b) for b in bounds])] + [np.array([rng.uniform(*b) for b in bounds])
                                                         for _ in range(40)]
    starts += [np.array(c) for c in product(*bounds)][:64]
    best_x, best_d = starts[0], -1.0
    for start in starts:
        # desirability is flat at 0 and 1, so the fitted response guides the search inside the region
        def objective(x):
            predicted = predict(x)
            score = 0.0
            for r in responses:
                low, high = ranges[r]
                span = (high - low) or 1.0
                if goal == "Maximize":
                    score -= predicted[r] / span
                elif goal == "Minimize":
                    score += predicted[r] / span
                else:
                    score += abs(predicted[r] - target) / span
            return score
        out = optimize.minimize(objective, start, bounds=bounds, method="L-BFGS-B")
        d = composite(out.x)
        if d > best_d + 1e-12:
            best_x, best_d = out.x, d
    return {"x": best_x, "desirability": best_d, "predicted": predict(best_x)}


@analysis("doe.optimizer", menu=RSM, title="Response Optimizer",
          params=[Param("responses", "Responses", "columns"), CONTINUOUS_FACTORS,
                  Param("goal", "Goal", "choice", default="Maximize", choices=GOALS),
                  Param("target", "Target", "number", default=0.0),
                  Param("model", "Terms in the model", "choice", default=RSM_MODELS[0], choices=RSM_MODELS)])
def response_optimizer(ws, responses, factors, goal="Maximize", target=0.0, model=RSM_MODELS[0]):
    factors = [f for f in factors if f not in responses]
    if not factors:
        raise ValueError("Select at least one factor")
    frame = ws.df[list(responses) + factors].dropna().astype(float)
    out = optimize_responses(frame, list(responses), factors, goal, target, model)
    res = Result("Response Optimizer")
    res.text(f"{tr('Goal')}: {tr(goal)}" + (f" ({target:g})" if goal == "Target" else ""))
    res.table(pd.DataFrame([dict(zip(factors, out["x"]))]), title="Optimal settings", index=False)
    res.table(pd.DataFrame({"Predicted response": out["predicted"]}), title="Prediction")
    res.text(f"{tr('Composite desirability')} = {out['desirability']:.4f}")
    return res


# ----- mixtures -----

MIXTURE_MODELS = ["Linear", "Quadratic"]


def scheffe_matrix(components: pd.DataFrame, model: str) -> pd.DataFrame:
    X = components.astype(float).copy()
    if model == "Quadratic":
        for a, b in combinations(components.columns, 2):
            X[f"{a}*{b}"] = components[a].astype(float) * components[b].astype(float)
    return X


@analysis("doe.analyze_mixture", menu=MIXTURE, title="Analyze Mixture Design",
          params=[RESPONSE, Param("components", "Components", "columns"),
                  Param("model", "Terms in the model", "choice", default="Quadratic", choices=MIXTURE_MODELS)])
def analyze_mixture(ws, response, components, model="Quadratic"):
    components = [c for c in components if c != response]
    if len(components) < 2:
        raise ValueError("Select at least two components")
    frame = ws.df[[response] + components].dropna().astype(float)
    totals = frame[components].sum(axis=1)
    if not np.allclose(totals, totals.iloc[0], rtol=1e-6, atol=1e-8):
        raise ValueError("The components must sum to the same total in every run")
    X = scheffe_matrix(frame[components] / totals.iloc[0], model)
    y = frame[response].to_numpy()
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("The design does not support this model; choose a simpler model or add runs")
    fit = sm.OLS(y, X).fit()
    sst = float(((y - y.mean()) ** 2).sum())
    df_model, df_error = X.shape[1] - 1, len(y) - X.shape[1]
    ssr = sst - fit.ssr
    f = (ssr / df_model) / (fit.ssr / df_error)
    res = Result("Analyze Mixture Design")
    res.text(tr("Scheffe model in proportions (no constant term)."))
    coefs = pd.DataFrame({"Coef": fit.params, "SE Coef": fit.bse})
    interactions = [c for c in X.columns if "*" in c]
    coefs.loc[interactions, "T-Value"] = fit.tvalues[interactions]
    coefs.loc[interactions, "P-Value"] = fit.pvalues[interactions]
    coefs.index.name = "Term"
    res.table(coefs, title="Coefficients")
    res.table(pd.DataFrame([{"S": np.sqrt(fit.ssr / df_error), "R-sq": 100 * ssr / sst,
                             "R-sq(adj)": 100 * (1 - (fit.ssr / df_error) / (sst / (len(y) - 1)))}]),
              title="Model Summary", index=False)
    anova = pd.DataFrame({"DF": [df_model, df_error, len(y) - 1], "SS": [ssr, fit.ssr, sst],
                          "MS": [ssr / df_model, fit.ssr / df_error, np.nan], "F-Value": [f, np.nan, np.nan],
                          "P-Value": [float(sps.f.sf(f, df_model, df_error)), np.nan, np.nan]},
                         index=pd.Index(["Regression", "Error", "Total"], name="Source"))
    res.table(anova, title="Analysis of Variance")
    return res.figure(residual_plots(fit.fittedvalues, fit.resid, response))


# ----- Taguchi -----

SN_RATIOS = ["Larger is better", "Smaller is better", "Nominal is best"]


def signal_to_noise(values: np.ndarray, kind: str) -> np.ndarray:
    """S/N ratio of each run, from the replicate responses in the rows of `values`."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == SN_RATIOS[0]:
            return -10 * np.log10(np.nanmean(1 / values ** 2, axis=1))
        if kind == SN_RATIOS[1]:
            return -10 * np.log10(np.nanmean(values ** 2, axis=1))
        return 10 * np.log10(np.nanmean(values, axis=1) ** 2 / np.nanvar(values, axis=1, ddof=1))


def response_table(values: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Mean of `values` at each level of each factor, with delta and rank."""
    columns = {}
    for name in factors.columns:
        columns[name] = values.groupby(factors[name].map(format_value)).mean()
    table = pd.DataFrame(columns)
    table.index = [f"{tr('Level')} {i}" for i in table.index]
    delta = table.max() - table.min()
    table.loc["Delta"] = delta
    table.loc[tr("Rank")] = delta.rank(ascending=False, method="min")
    table.index.name = ""
    return table


@analysis("doe.analyze_taguchi", menu=TAGUCHI, title="Analyze Taguchi Design",
          params=[Param("responses", "Response data (one column per replicate)", "columns"), FACTOR_COLUMNS,
                  Param("kind", "Signal-to-noise ratio", "choice", default=SN_RATIOS[0], choices=SN_RATIOS)])
def analyze_taguchi(ws, responses, factors, kind=SN_RATIOS[0]):
    factors = [f for f in factors if f not in responses]
    if not factors:
        raise ValueError("Select at least one factor")
    if kind == SN_RATIOS[2] and len(responses) < 2:
        raise ValueError("Nominal is best needs at least two response columns")
    frame = ws.df[list(responses) + factors].dropna()
    values = frame[list(responses)].to_numpy(dtype=float)
    sn = pd.Series(signal_to_noise(values, kind), index=frame.index)
    means = pd.Series(values.mean(axis=1), index=frame.index)
    res = Result("Analyze Taguchi Design")
    res.text(f"{tr('Signal-to-noise ratio')}: {tr(kind)}")
    res.table(response_table(sn, frame[factors]), title="Response Table for Signal-to-Noise Ratios")
    res.table(response_table(means, frame[factors]), title="Response Table for Means")
    fig = new_figure(max(5.0, 2.6 * len(factors)), 6.0)
    axes = fig.subplots(2, len(factors), sharey="row", squeeze=False)
    for row, (series, label) in enumerate(((sn, tr("Mean of S/N ratios")), (means, tr("Mean of means")))):
        for ax, factor in zip(axes[row], factors):
            level_means = series.groupby(frame[factor].map(format_value)).mean()
            ax.plot(list(level_means.index), level_means.to_numpy(), "-o", color=BLUE)
            ax.axhline(series.mean(), color=GRAY, linestyle="--")
            ax.set_xlabel(factor)
        axes[row][0].set_ylabel(label)
    fig.suptitle(tr("Main Effects Plot"))
    res.figure(fig)
    return res

