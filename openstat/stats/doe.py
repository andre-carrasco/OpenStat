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
from openstat.core.worksheet import format_value, parse_cell
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._design import CONSTANT, Design
from openstat.stats._util import median_ranks
from openstat.stats.regression import (PALETTE, coefficient_table, glm_report, model_summary, parse_points,
                                       residual_plots)

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


CONFERENCE_ORDERS = [4, 6, 8, 10, 12, 14, 18, 20, 24, 30]


def conference_matrix(order: int) -> np.ndarray:
    """Conference matrix (zero diagonal, +-1 elsewhere, orthogonal columns) by Paley's construction
    over the field with order - 1 elements, which must be a prime or 9."""
    q = order - 1
    if q == 9:   # GF(9) as a + b i over the integers mod 3, with i^2 = -1
        elements = [(a, b) for a in range(3) for b in range(3)]
        zero = (0, 0)

        def subtract(x, y):
            return (x[0] - y[0]) % 3, (x[1] - y[1]) % 3

        def square(x):
            return (x[0] * x[0] - x[1] * x[1]) % 3, (2 * x[0] * x[1]) % 3
    elif q > 2 and all(q % d for d in range(2, int(q ** 0.5) + 1)):
        elements, zero = list(range(q)), 0

        def subtract(x, y):
            return (x - y) % q

        def square(x):
            return x * x % q
    else:
        raise ValueError(f"No conference matrix of order {order} is available")
    squares = {square(x) for x in elements if x != zero}
    matrix = np.zeros((order, order))
    matrix[0, 1:] = 1
    matrix[1:, 0] = 1 if q % 4 == 1 else -1
    matrix[1:, 1:] = [[0 if a == b else 1 if subtract(a, b) in squares else -1 for b in elements] for a in elements]
    return matrix


def definitive_screening(factors: int) -> np.ndarray:
    """The fold-over pairs of a definitive screening design (Jones and Nachtsheim), without the center run.

    When no conference matrix has exactly `factors` columns the next larger one is used and the
    extra columns are dropped, which adds runs.
    """
    orders = [o for o in CONFERENCE_ORDERS if o >= factors]
    if factors < 3 or not orders:
        raise ValueError(f"A definitive screening design takes 3 to {CONFERENCE_ORDERS[-1]} factors")
    half = conference_matrix(orders[0])[:, :factors]
    return np.vstack([half, -half])


def _factor_names(count: int) -> list[str]:
    return list(LETTERS[:count]) if count <= len(LETTERS) else [f"X{i + 1}" for i in range(count)]


@analysis("doe.create_dsd", menu=FACTORIAL, title="Create Definitive Screening Design",
          params=[Param("factors", "Number of factors", "int", default=6, minimum=3, maximum=CONFERENCE_ORDERS[-1]),
                  Param("center_points", "Number of center points", "int", default=1, minimum=1, maximum=20),
                  RANDOMIZE, SEED])
def create_definitive_screening(ws, factors=6, center_points=1, randomize=True, seed=0):
    matrix = definitive_screening(factors)
    frame = finish_design(matrix, _factor_names(factors), 1, center_points, randomize, seed)
    return _design_result("Create Definitive Screening Design", frame, {
        "Factors": factors, "Fold-over pairs": len(matrix) // 2, "Center points": center_points,
        "Total runs": len(frame)},
        [tr("Each factor has three levels; main effects are not aliased with two-factor interactions or squares.")])


@analysis("doe.create_split_plot", menu=FACTORIAL, title="Create Split-Plot Design",
          params=[Param("hard", "Number of hard-to-change factors", "int", default=1, minimum=1, maximum=4),
                  Param("easy", "Number of easy-to-change factors", "int", default=2, minimum=1, maximum=7),
                  Param("replicates", "Replicates of the whole plots", "int", default=2, minimum=1, maximum=50),
                  RANDOMIZE, SEED])
def create_split_plot(ws, hard=1, easy=2, replicates=2, randomize=True, seed=0):
    rng = np.random.default_rng(seed or None)
    whole, sub = np.tile(pyDOE3.ff2n(hard), (replicates, 1)), pyDOE3.ff2n(easy)
    names = _factor_names(hard + easy)
    rows = []
    # the hard-to-change factors are set once for each whole plot; the others vary inside it
    for plot in (rng.permutation(len(whole)) if randomize else range(len(whole))):
        for run in (rng.permutation(len(sub)) if randomize else range(len(sub))):
            rows.append([plot * len(sub) + run + 1, plot + 1, *whole[plot], *sub[run]])
    frame = pd.DataFrame(rows, columns=["StdOrder", "WP"] + names, dtype=float)
    frame.insert(1, "RunOrder", np.arange(1, len(frame) + 1, dtype=float))
    return _design_result("Create Split-Plot Design", frame, {
        "Hard-to-change factors": hard, "Easy-to-change factors": easy, "Whole plots": len(whole),
        "Runs per whole plot": len(sub), "Total runs": len(frame)},
        [f"{tr('Hard-to-change factors')}: {', '.join(names[:hard])}",
         tr("Analyze with Mixed Effects Model, taking WP as the random factor.")])


def d_optimal_rows(X: np.ndarray, runs: int, seed=0, starts=10) -> tuple[np.ndarray, float]:
    """Rows of the candidate model matrix X that maximize det(X'X), by Fedorov exchanges from
    several random starts; returns the sorted row positions and log det(X'X)."""
    count, terms = X.shape
    if not terms <= runs <= count:
        raise ValueError("The number of runs must be between the number of model terms and the number of candidates")
    rng = np.random.default_rng(seed)
    best, best_logdet = None, -np.inf
    for _ in range(starts):
        chosen = rng.choice(count, runs, replace=False)
        for _ in range(50):   # a random start may be singular
            if np.linalg.matrix_rank(X[chosen]) == terms:
                break
            chosen = rng.choice(count, runs, replace=False)
        else:
            continue
        for _ in range(100):
            improved = False
            for position in range(runs):
                inverse = np.linalg.inv(X[chosen].T @ X[chosen])
                variance = np.einsum("ij,jk,ik->i", X, inverse, X)
                out = chosen[position]
                cross = X @ inverse @ X[out]
                # swapping `out` for candidate j multiplies the determinant by 1 + gain[j]
                gain = variance - (variance[out] * variance - cross ** 2) - variance[out]
                gain[chosen] = -np.inf
                into = int(np.argmax(gain))
                if gain[into] > 1e-9:
                    chosen[position] = into
                    improved = True
            if not improved:
                break
        logdet = np.linalg.slogdet(X[chosen].T @ X[chosen])[1]
        if logdet > best_logdet:
            best, best_logdet = np.sort(chosen), float(logdet)
    if best is None:
        raise ValueError("The candidate points do not support this model")
    return best, best_logdet


MAX_CANDIDATES = 2000


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


def parse_numbers(text: str, count: int, what: str) -> np.ndarray:
    try:
        values = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise ValueError(f"{what}: enter numbers separated by commas") from None
    if len(values) != count:
        raise ValueError(f"{what}: {count} values are needed")
    return np.array(values)


def extreme_vertices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Vertices of the mixtures that sum to 1 with each component between its bounds.

    At a vertex all components but one sit on a bound and the last one takes what is left.
    """
    if (lower < 0).any() or (upper > 1).any() or (lower >= upper).any():
        raise ValueError("Each lower bound must be less than its upper bound, both between 0 and 1")
    if lower.sum() >= 1 or upper.sum() <= 1:
        raise ValueError("The bounds leave no room for mixtures that sum to 1")
    q, found = len(lower), set()
    for free in range(q):
        fixed = [i for i in range(q) if i != free]
        for corner in product((0, 1), repeat=q - 1):
            point = np.zeros(q)
            point[fixed] = np.where(corner, upper[fixed], lower[fixed])
            point[free] = 1 - point[fixed].sum()
            if lower[free] - 1e-9 <= point[free] <= upper[free] + 1e-9:
                found.add(tuple(np.round(point, 9) + 0.0))
    return np.array(sorted(found, reverse=True))


@analysis("doe.create_extreme_vertices", menu=MIXTURE, title="Create Extreme Vertices Design",
          params=[Param("lower", "Lower bound of each component (e.g. 0.1, 0.2, 0)", "text", default="0.1, 0.2, 0.1"),
                  Param("upper", "Upper bound of each component (e.g. 0.6, 0.7, 0.5)", "text", default="0.6, 0.7, 0.5"),
                  Param("centroid", "Add the center point", "bool", default=True), REPLICATES, RANDOMIZE, SEED])
def create_extreme_vertices(ws, lower="0.1, 0.2, 0.1", upper="0.6, 0.7, 0.5", centroid=True, replicates=1,
                            randomize=True, seed=0):
    count = len([v for v in str(lower).replace(";", ",").split(",") if v.strip()])
    if not 2 <= count <= 10:
        raise ValueError("A mixture design takes 2 to 10 components")
    matrix = extreme_vertices(parse_numbers(lower, count, tr("Lower bound")), parse_numbers(upper, count, tr("Upper bound")))
    vertices = len(matrix)
    if centroid:
        matrix = np.vstack([matrix, matrix.mean(axis=0)])
    frame = finish_design(matrix, LETTERS[:count], replicates, 0, randomize, seed).drop(columns="CenterPt")
    return _design_result("Create Extreme Vertices Design", frame, {
        "Components": count, "Vertices": vertices, "Design points": len(matrix), "Replicates": replicates,
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

def code_two_level(series: pd.Series, name: str, ws=None) -> np.ndarray:
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
    levels = ws.levels(name, labels) if ws else sorted(labels.unique())
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
    coded = pd.DataFrame({f: code_two_level(frame[f], f, ws) for f in factors}, index=frame.index)
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


@analysis("doe.analyze_binary", menu=FACTORIAL, title="Analyze Binary Response for Factorial Design",
          params=[Param("events", "Number of events", "column"), Param("trials", "Number of trials", "column"),
                  FACTOR_COLUMNS, Param("order", "Terms in the model", "choice", default=ORDERS[1], choices=ORDERS)])
def analyze_binary_factorial(ws, events, trials, factors, order=ORDERS[1]):
    factors = [f for f in factors if f not in (events, trials)]
    if not factors:
        raise ValueError("Select at least one factor")
    frame = ws.df[[events, trials] + factors].dropna()
    hits, n = frame[events].to_numpy(dtype=float), frame[trials].to_numpy(dtype=float)
    if (hits < 0).any() or (hits > n).any() or (n <= 0).any():
        raise ValueError("Events must be between 0 and the number of trials")
    coded = pd.DataFrame({f: code_two_level(frame[f], f, ws) for f in factors}, index=frame.index)
    max_order = len(factors) if order == ORDERS[3] else ORDERS.index(order) + 1
    X, _, dropped = factorial_matrix(coded, max_order)
    if len(frame) <= X.shape[1]:
        raise ValueError("The design does not support this model; choose a simpler model or add runs")
    fit = sm.GLM(np.column_stack([hits, n - hits]), X, family=sm.families.Binomial()).fit()
    res = Result("Analyze Binary Response for Factorial Design")
    res.text(f"{tr('Events')}: {int(hits.sum())}    {tr('Trials')}: {int(n.sum())}    {tr('Runs')}: {len(frame)}    "
             f"{tr('Coded factors: low level = -1, high level = +1')}")
    if dropped:
        res.text(f"{tr('Terms that cannot be estimated (aliased) were removed')}: {', '.join(dropped)}")
    glm_report(res, fit, "Odds Ratio")
    return res


# ----- response surface -----

RSM_MODELS =["Full quadratic", "Linear", "Linear + interactions", "Linear + squares"]


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


def surface_grid(fit, frame, factors, model, x_name, y_name, hold=None, size=60):
    """Fitted response over a grid of two factors, the others at their mean or at the `hold` values."""
    grid_x = np.linspace(frame[x_name].min(), frame[x_name].max(), size)
    grid_y = np.linspace(frame[y_name].min(), frame[y_name].max(), size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    points = pd.DataFrame({f: np.full(gx.size, (hold or {}).get(f, frame[f].astype(float).mean())) for f in factors})
    points[x_name], points[y_name] = gx.ravel(), gy.ravel()
    return gx, gy, np.asarray(fit.predict(quadratic_matrix(points[factors], model)[0])).reshape(gx.shape)


def surface_figure(fit, frame, factors, model, response, x_name, y_name, hold=None):
    gx, gy, z = surface_grid(fit, frame, factors, model, x_name, y_name, hold)
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


def parse_goals(text: str, responses: list[str]) -> dict:
    """'Y=max; Z=min; W=target 5' -> {response: (goal, target)}."""
    goals = {}
    for part in str(text).split(";"):
        if not part.strip():
            continue
        name, _, spec = (s.strip() for s in part.partition("="))
        words = spec.lower().split()
        if name not in responses or not words:
            raise ValueError(f"Invalid goal: {part.strip()}")
        if words[0].startswith("max"):
            goals[name] = (GOALS[0], None)
        elif words[0].startswith("min"):
            goals[name] = (GOALS[1], None)
        else:
            try:
                goals[name] = (GOALS[2], float(words[-1].replace(",", ".")))
            except ValueError:
                raise ValueError(f"Invalid goal: {part.strip()}") from None
    return goals


def optimize_responses(frame, responses, factors, goal, target=None, model=RSM_MODELS[0], seed=0,
                       goals=None) -> dict:
    """`goal` and `target` apply to every response that has no entry of its own in `goals`."""
    fits = {r: fit_surface(frame[[r] + factors].dropna(), r, factors, model)[0] for r in responses}
    bounds = [(frame[f].min(), frame[f].max()) for f in factors]
    ranges = {r: (frame[r].min(), frame[r].max()) for r in responses}
    goals = {r: (goals or {}).get(r, (goal, target)) for r in responses}

    def predict(x):
        point = pd.DataFrame([x], columns=factors)
        matrix = quadratic_matrix(point, model)[0]
        return {r: float(np.asarray(fit.predict(matrix))[0]) for r, fit in fits.items()}

    def composite(x):
        predicted = predict(x)
        values = [float(desirability(predicted[r], *ranges[r], *goals[r])) for r in responses]
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
                if goals[r][0] == "Maximize":
                    score -= predicted[r] / span
                elif goals[r][0] == "Minimize":
                    score += predicted[r] / span
                else:
                    score += abs(predicted[r] - goals[r][1]) / span
            return score
        out = optimize.minimize(objective, start, bounds=bounds, method="L-BFGS-B")
        d = composite(out.x)
        if d > best_d + 1e-12:
            best_x, best_d = out.x, d
    return {"x": best_x, "desirability": best_d, "predicted": predict(best_x), "goals": goals}


@analysis("doe.optimizer", menu=RSM, title="Response Optimizer",
          params=[Param("responses", "Responses", "columns"), CONTINUOUS_FACTORS,
                  Param("goal", "Goal", "choice", default="Maximize", choices=GOALS),
                  Param("target", "Target", "number", default=0.0),
                  Param("model", "Terms in the model", "choice", default=RSM_MODELS[0], choices=RSM_MODELS),
                  Param("goals", "Goals that differ by response (e.g. Y=max; Z=min; W=target 5)", "text")])
def response_optimizer(ws, responses, factors, goal="Maximize", target=0.0, model=RSM_MODELS[0], goals=""):
    factors = [f for f in factors if f not in responses]
    if not factors:
        raise ValueError("Select at least one factor")
    frame = ws.df[list(responses) + factors].dropna().astype(float)
    out = optimize_responses(frame, list(responses), factors, goal, target, model,
                             goals=parse_goals(goals, list(responses)))
    res = Result("Response Optimizer")
    res.table(pd.DataFrame([dict(zip(factors, out["x"]))]), title="Optimal settings", index=False)
    prediction = pd.DataFrame({"Goal": [tr(g) + (f" ({t:g})" if g == "Target" else "") for g, t in out["goals"].values()],
                               "Predicted response": list(out["predicted"].values())},
                              index=pd.Index(list(out["predicted"]), name="Response"))
    res.table(prediction, title="Prediction")
    res.text(f"{tr('Composite desirability')} = {out['desirability']:.4f}")
    return res


def parse_hold(text: str, factors: list[str]) -> dict:
    """'C=5, D=2' -> the values at which factors off the axes of a plot are held."""
    hold = {}
    for part in str(text).replace(";", ",").split(","):
        if not part.strip():
            continue
        name, _, value = (s.strip() for s in part.partition("="))
        try:
            hold[name] = float(value)
        except ValueError:
            raise ValueError(f"Invalid hold value: {part.strip()}") from None
        if name not in factors:
            raise ValueError(f"Invalid hold value: {part.strip()}")
    return hold


def _plot_factors(responses, factors, x, y) -> list[str]:
    factors = [f for f in factors if f not in responses]
    if x == y or x not in factors or y not in factors:
        raise ValueError("The two axes must be different factors of the model")
    return factors


X_AXIS = Param("x", "Factor on the X axis", "column")
Y_AXIS = Param("y", "Factor on the Y axis", "column")
HOLD = Param("hold", "Values of the other factors (default: mean; e.g. C=5)", "text")
MODEL = Param("model", "Terms in the model", "choice", default=RSM_MODELS[0], choices=RSM_MODELS)


@analysis("doe.surface_plots", menu=RSM, title="Contour and Surface Plots",
          params=[RESPONSE, CONTINUOUS_FACTORS, X_AXIS, Y_AXIS, MODEL, HOLD])
def surface_plots(ws, response, factors, x, y, model=RSM_MODELS[0], hold=""):
    factors = _plot_factors([response], factors, x, y)
    held = parse_hold(hold, factors)
    frame = ws.df[[response] + factors].dropna().astype(float)
    fit = fit_surface(frame, response, factors, model)[0]
    res = Result("Contour and Surface Plots")
    others = {f: held.get(f, float(frame[f].mean())) for f in factors if f not in (x, y)}
    if others:
        res.text(f"{tr('Hold values')}: " + ", ".join(f"{f} = {v:.5g}" for f, v in others.items()))
    return res.figure(surface_figure(fit, frame, factors, model, response, x, y, held))


@analysis("doe.overlaid_contour", menu=RSM, title="Overlaid Contour Plot",
          params=[Param("responses", "Responses", "columns"), CONTINUOUS_FACTORS, X_AXIS, Y_AXIS,
                  Param("lower", "Lower bound of each response (e.g. 10, 5)", "text"),
                  Param("upper", "Upper bound of each response (e.g. 20, 8)", "text"), MODEL, HOLD])
def overlaid_contour(ws, responses, factors, x, y, lower, upper, model=RSM_MODELS[0], hold=""):
    responses = list(responses)
    factors = _plot_factors(responses, factors, x, y)
    held = parse_hold(hold, factors)
    low = parse_numbers(lower, len(responses), tr("Lower bound"))
    high = parse_numbers(upper, len(responses), tr("Upper bound"))
    if (low >= high).any():
        raise ValueError("The lower limit must be less than the upper limit")
    frame = ws.df[responses + factors].dropna().astype(float)
    fig = new_figure()
    ax = fig.add_subplot()
    feasible = None
    for i, response in enumerate(responses):
        fit = fit_surface(frame, response, factors, model)[0]
        gx, gy, z = surface_grid(fit, frame, factors, model, x, y, held, size=120)
        inside = (z >= low[i]) & (z <= high[i])
        feasible = inside if feasible is None else feasible & inside
        color = PALETTE[i % len(PALETTE)]
        ax.contour(gx, gy, z, levels=[low[i]], colors=[color], linestyles="solid")
        ax.contour(gx, gy, z, levels=[high[i]], colors=[color], linestyles="dashed")
        ax.plot([], [], color=color, label=f"{response}: {low[i]:g} – {high[i]:g}")
    ax.contourf(gx, gy, feasible.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#d9d9d9", "white"])
    ax.set_xlabel(x), ax.set_ylabel(y), ax.legend(fontsize=8)
    ax.set_title(tr("Overlaid Contour Plot"))
    res = Result("Overlaid Contour Plot")
    res.text(tr("White: every response is within its bounds. Solid line: lower bound; dashed line: upper bound."))
    res.text(f"{tr('Share of the plotted region where every response is within its bounds')}: "
             f"{100 * feasible.mean():.1f}%")
    return res.figure(fig)


@analysis("doe.select_optimal", menu=RSM, title="Select Optimal Design",
          params=[Param("factors", "Factors (the rows are the candidate points)", "columns"),
                  Param("runs", "Number of runs to select", "int", default=12, minimum=2, maximum=MAX_CANDIDATES),
                  MODEL, Param("seed", "Base for random numbers", "int", default=1, minimum=1)])
def select_optimal(ws, factors, runs=12, model=RSM_MODELS[0], seed=1):
    rows = np.flatnonzero(ws.df[list(factors)].notna().all(axis=1).to_numpy())
    if len(rows) > MAX_CANDIDATES:
        raise ValueError(f"At most {MAX_CANDIDATES} candidate points are accepted")
    points = ws.df[list(factors)].iloc[rows].astype(float)
    spread = points.max() - points.min()
    if (spread == 0).any():
        raise ValueError("Each factor needs at least two levels")
    coded = (points - (points.max() + points.min()) / 2) / (spread / 2)   # each factor on -1 .. +1
    X = quadratic_matrix(coded, model)[0].to_numpy()
    chosen, logdet = d_optimal_rows(X, runs, seed)
    full = np.linalg.slogdet(X.T @ X)[1]
    terms = X.shape[1]
    res = Result("Select Optimal Design")
    res.table(pd.DataFrame([{
        "Candidate points": len(rows), "Selected runs": runs, "Model terms": terms,
        "D-efficiency (%)": 100 * np.exp(logdet / terms) / runs,
        "Relative to all candidates (%)": 100 * np.exp((logdet - full) / terms) * len(rows) / runs}]),
        title="D-optimal design", index=False, digits=2)
    res.text(f"{tr('Selected rows')}: " + ", ".join(str(r + 1) for r in rows[chosen]))
    res.worksheet(f"{ws.name} ({tr('D-optimal')})", ws.df.iloc[rows[chosen]])
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


@analysis("doe.predict_taguchi", menu=TAGUCHI, title="Predict Taguchi Results",
          params=[Param("responses", "Response data (one column per replicate)", "columns"), FACTOR_COLUMNS,
                  Param("kind", "Signal-to-noise ratio", "choice", default=SN_RATIOS[0], choices=SN_RATIOS),
                  Param("levels", "Factor levels to predict (e.g. A=1, B=2)", "text")])
def predict_taguchi(ws, responses, factors, kind=SN_RATIOS[0], levels=""):
    factors = [f for f in factors if f not in responses]
    if not factors:
        raise ValueError("Select at least one factor")
    if kind == SN_RATIOS[2] and len(responses) < 2:
        raise ValueError("Nominal is best needs at least two response columns")
    settings = parse_points(levels, factors)
    frame = ws.df[list(responses) + factors].dropna()
    values = frame[list(responses)].to_numpy(dtype=float)
    series = {"S/N ratio": pd.Series(signal_to_noise(values, kind), index=frame.index),
              "Mean": pd.Series(values.mean(axis=1), index=frame.index)}
    labels = {f: frame[f].map(format_value) for f in factors}
    rows = {}
    for point in settings:
        wanted = {f: format_value(parse_cell(point[f])) for f in factors}
        for f in factors:
            if wanted[f] not in set(labels[f]):
                raise ValueError(f"{f}: unknown level {wanted[f]}")
        # additive model: the overall mean plus the effect of each chosen level
        rows[", ".join(f"{f} = {wanted[f]}" for f in factors)] = {
            name: float(s.mean() + sum(s[labels[f] == wanted[f]].mean() - s.mean() for f in factors))
            for name, s in series.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Settings"
    res = Result("Predict Taguchi Results")
    res.text(f"{tr('Signal-to-noise ratio')}: {tr(kind)}    {tr('Additive model of the main effects')}")
    return res.table(table, title="Prediction")


# ----- changing a design -----

MODIFICATIONS = ["Randomize the run order", "Add a replicate", "Fold on all factors", "Sort by standard order"]


@analysis("doe.modify", menu=MENU, title="Modify Design",
          params=[CONTINUOUS_FACTORS,
                  Param("operation", "Operation", "choice", default=MODIFICATIONS[0], choices=MODIFICATIONS), SEED])
def modify_design(ws, factors, operation=MODIFICATIONS[0], seed=0):
    frame = ws.df.dropna(subset=list(factors)).reset_index(drop=True)
    if not len(frame):
        raise ValueError("The worksheet has no data.")
    if "StdOrder" not in frame:
        frame.insert(0, "StdOrder", np.arange(1, len(frame) + 1, dtype=float))
    if operation in MODIFICATIONS[1:3]:
        extra = frame.copy()
        if operation == MODIFICATIONS[2]:   # each factor is reflected about the middle of its range
            for f in factors:
                extra[f] = frame[f].min() + frame[f].max() - frame[f]
        extra["StdOrder"] = frame["StdOrder"] + frame["StdOrder"].max()
        frame = pd.concat([frame, extra], ignore_index=True)
    elif operation == MODIFICATIONS[3]:
        frame = frame.sort_values("StdOrder", kind="stable").reset_index(drop=True)
    if operation != MODIFICATIONS[3]:
        frame = frame.iloc[np.random.default_rng(seed or None).permutation(len(frame))].reset_index(drop=True)
    frame["RunOrder"] = np.arange(1, len(frame) + 1, dtype=float)
    res = Result("Modify Design")
    res.text(f"{tr(operation)}    {tr('Total runs')}: {len(frame)}")
    res.worksheet(f"{ws.name} ({tr('modified')})", frame)
    return res

