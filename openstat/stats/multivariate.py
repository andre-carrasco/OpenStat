"""Multivariate analysis: principal components, factors, items, clusters, discriminant and correspondence."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._design import complete_rows

MENU = ("Stat", "Multivariate")
BLUE, RED, GRAY = "#5b9bd5", "#c00000", "#7f7f7f"
PALETTE = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#7030a0", "#c00000", "#00b0f0", "#7f7f7f"]

VARIABLES = Param("columns", "Variables", "columns")
STORE = Param("store", "Store results in the worksheet", "bool", default=False)
MATRIX_TYPES = ["Correlation", "Covariance"]


def _data(ws, columns, minimum=2) -> tuple[pd.DataFrame, np.ndarray]:
    if len(columns) < minimum:
        raise ValueError(f"Select at least {minimum} columns")
    rows = complete_rows(ws, columns)
    frame = ws.df[list(columns)].iloc[rows].astype(float)
    if len(frame) <= len(columns):
        raise ValueError("More observations than variables are needed")
    return frame, rows


def _aligned(values, rows, nrows) -> list:
    out = np.full(nrows, np.nan)
    out[rows] = values
    return out.tolist()


def principal_components(frame: pd.DataFrame, matrix: str = "Correlation") -> dict:
    X = frame.to_numpy(dtype=float)
    centered = X - X.mean(axis=0)
    if matrix == "Correlation":
        sd = X.std(axis=0, ddof=1)
        if (sd == 0).any():
            raise ValueError("A variable has no variation")
        centered = centered / sd
    eigenvalues, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = np.clip(eigenvalues[order], 0, None), vectors[:, order]
    # fix the sign so that the largest loading of each component is positive
    signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(vectors.shape[1])])
    vectors = vectors * np.where(signs == 0, 1, signs)
    return {"eigenvalues": eigenvalues, "vectors": vectors, "scores": centered @ vectors,
            "proportion": eigenvalues / eigenvalues.sum()}


@analysis("mv.pca", menu=MENU, title="Principal Components Analysis",
          params=[VARIABLES, Param("matrix", "Type of matrix", "choice", default="Correlation", choices=MATRIX_TYPES),
                  Param("components", "Number of components to compute (0 = all)", "int", default=0, minimum=0),
                  STORE])
def pca(ws, columns, matrix="Correlation", components=0, store=False):
    frame, rows = _data(ws, columns)
    out = principal_components(frame, matrix)
    k = len(columns) if components < 1 else min(components, len(columns))
    names = [f"PC{i + 1}" for i in range(len(columns))]
    res = Result("Principal Components Analysis")
    res.table(pd.DataFrame({"Eigenvalue": out["eigenvalues"], "Proportion": out["proportion"],
                            "Cumulative": np.cumsum(out["proportion"])}, index=pd.Index(names, name="Component")),
              title=f"{tr('Eigenanalysis of the matrix')}: {tr(matrix)}")
    res.table(pd.DataFrame(out["vectors"][:, :k], index=pd.Index(columns, name="Variable"), columns=names[:k]),
              title="Eigenvectors")
    fig = new_figure(11.0, 3.8)
    a, b, c = fig.subplots(1, 3)
    a.plot(np.arange(1, len(columns) + 1), out["eigenvalues"], "-o", color=BLUE)
    a.set_xlabel(tr("Component number")), a.set_ylabel(tr("Eigenvalue")), a.set_title(tr("Scree Plot"), fontsize=9)
    if len(columns) > 1:
        b.plot(out["scores"][:, 0], out["scores"][:, 1], "o", color=BLUE, markersize=4)
        b.axhline(0, color=GRAY, linewidth=0.8), b.axvline(0, color=GRAY, linewidth=0.8)
        b.set_xlabel("PC1"), b.set_ylabel("PC2"), b.set_title(tr("Score Plot"), fontsize=9)
        for name, (x, y) in zip(columns, out["vectors"][:, :2]):
            c.plot([0, x], [0, y], color=BLUE)
            c.annotate(name, (x, y), fontsize=8)
        c.axhline(0, color=GRAY, linewidth=0.8), c.axvline(0, color=GRAY, linewidth=0.8)
        c.set_xlabel("PC1"), c.set_ylabel("PC2"), c.set_title(tr("Loading Plot"), fontsize=9)
    res.figure(fig)
    if store:
        for i in range(k):
            res.store(f"PC{i + 1}", _aligned(out["scores"][:, i], rows, ws.nrows))
    return res


def varimax(loadings: np.ndarray, iterations=200, tolerance=1e-8) -> np.ndarray:
    """Varimax rotation with Kaiser normalization."""
    p, k = loadings.shape
    if k < 2:
        return loadings
    norms = np.sqrt((loadings ** 2).sum(axis=1))
    norms[norms == 0] = 1.0
    L = loadings / norms[:, None]
    rotation = np.eye(k)
    previous = 0.0
    for _ in range(iterations):
        rotated = L @ rotation
        u, s, vt = np.linalg.svd(L.T @ (rotated ** 3 - rotated @ np.diag((rotated ** 2).sum(axis=0)) / p))
        rotation = u @ vt
        if s.sum() < previous * (1 + tolerance):
            break
        previous = s.sum()
    return (L @ rotation) * norms[:, None]


EXTRACTION = ["Principal components", "Maximum likelihood"]
ROTATION = ["None", "Varimax"]


def factor_loadings(frame: pd.DataFrame, factors: int, method=EXTRACTION[0], rotation="Varimax") -> np.ndarray:
    if not 1 <= factors < frame.shape[1] + (1 if method == EXTRACTION[0] else 0):
        raise ValueError("The number of factors must be less than the number of variables")
    if method == EXTRACTION[0]:
        out = principal_components(frame, "Correlation")
        loadings = out["vectors"][:, :factors] * np.sqrt(out["eigenvalues"][:factors])
    else:
        from sklearn.decomposition import FactorAnalysis

        z = (frame - frame.mean()) / frame.std(ddof=1)
        loadings = FactorAnalysis(n_components=factors, max_iter=2000).fit(z.to_numpy()).components_.T
    loadings = varimax(loadings) if rotation == "Varimax" else loadings
    # order factors by explained variance and make the dominant loading positive
    order = np.argsort(-(loadings ** 2).sum(axis=0))
    loadings = loadings[:, order]
    signs = np.sign(loadings[np.abs(loadings).argmax(axis=0), np.arange(loadings.shape[1])])
    return loadings * np.where(signs == 0, 1, signs)


@analysis("mv.factor", menu=MENU, title="Factor Analysis",
          params=[VARIABLES, Param("factors", "Number of factors to extract", "int", default=2, minimum=1),
                  Param("method", "Method of extraction", "choice", default=EXTRACTION[0], choices=EXTRACTION),
                  Param("rotation", "Type of rotation", "choice", default="Varimax", choices=ROTATION)])
def factor_analysis(ws, columns, factors=2, method=EXTRACTION[0], rotation="Varimax"):
    frame, _ = _data(ws, columns)
    loadings = factor_loadings(frame, factors, method, rotation)
    names = [f"{tr('Factor')}{i + 1}" for i in range(loadings.shape[1])]
    table = pd.DataFrame(loadings, index=pd.Index(columns, name="Variable"), columns=names)
    table[tr("Communality")] = (loadings ** 2).sum(axis=1)
    variance = (loadings ** 2).sum(axis=0)
    res = Result("Factor Analysis")
    res.text(f"{tr(method)}    {tr('Type of rotation')}: {tr(rotation)}")
    res.table(table, title="Factor Loadings and Communalities")
    return res.table(pd.DataFrame({"Variance": variance, "% Var": 100 * variance / len(columns)},
                                  index=pd.Index(names, name="Factor")), title="Variance Explained")


def cronbach_alpha(frame: pd.DataFrame) -> float:
    k = frame.shape[1]
    total_var = frame.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - frame.var(ddof=1).sum() / total_var)) if total_var > 0 else float("nan")


@analysis("mv.item", menu=MENU, title="Item Analysis", params=[VARIABLES])
def item_analysis(ws, columns):
    frame, _ = _data(ws, columns)
    res = Result("Item Analysis")
    res.text(f"{tr('Cronbach alpha')} = {cronbach_alpha(frame):.4f}    N = {len(frame)}")
    rows = {}
    for name in columns:
        rest = frame.drop(columns=name)
        rows[name] = {"Mean": frame[name].mean(), "StDev": frame[name].std(ddof=1),
                      "Item-adjusted total correlation": float(np.corrcoef(frame[name], rest.sum(axis=1))[0, 1]),
                      "Alpha if item omitted": cronbach_alpha(rest) if rest.shape[1] > 1 else np.nan}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Variable"
    res.table(table, title="Omitted Item Statistics")
    return res.table(frame.corr(), title="Correlation Matrix")


LINKAGE = ["Single", "Complete", "Average", "Centroid", "Median", "Ward", "McQuitty"]
SCIPY_LINKAGE = dict(zip(LINKAGE, ["single", "complete", "average", "centroid", "median", "ward", "weighted"]))
DISTANCES = ["Euclidean", "Squared Euclidean", "Manhattan", "Pearson"]
SCIPY_DISTANCE = dict(zip(DISTANCES, ["euclidean", "sqeuclidean", "cityblock", "seuclidean"]))


def _dendrogram(tree, labels, title: str, clusters: int):
    fig = new_figure(max(6.0, min(14.0, 0.25 * len(labels) + 3)), 4.2)
    ax = fig.add_subplot()
    threshold = tree[-(clusters - 1), 2] if 1 < clusters <= len(tree) else 0
    hierarchy.dendrogram(tree, labels=labels, ax=ax, color_threshold=threshold, above_threshold_color=GRAY,
                         leaf_font_size=7 if len(labels) > 30 else 9)
    ax.set_ylabel(tr("Distance")), ax.set_title(title)
    return fig


@analysis("mv.cluster_obs", menu=MENU, title="Cluster Observations",
          params=[VARIABLES, Param("linkage", "Linkage method", "choice", default="Ward", choices=LINKAGE),
                  Param("distance", "Distance measure", "choice", default="Euclidean", choices=DISTANCES),
                  Param("standardize", "Standardize variables", "bool", default=True),
                  Param("clusters", "Number of clusters", "int", default=3, minimum=1), STORE])
def cluster_observations(ws, columns, linkage="Ward", distance="Euclidean", standardize=True, clusters=3,
                         store=False):
    frame, rows = _data(ws, columns, minimum=1)
    X = frame.to_numpy()
    if standardize:
        sd = X.std(axis=0, ddof=1)
        X = (X - X.mean(axis=0)) / np.where(sd == 0, 1, sd)
    if linkage in ("Ward", "Centroid", "Median") and distance != "Euclidean":
        raise ValueError(f"{linkage} linkage requires the Euclidean distance")
    tree = hierarchy.linkage(pdist(X, SCIPY_DISTANCE[distance]), SCIPY_LINKAGE[linkage])
    clusters = min(clusters, len(X))
    membership = hierarchy.fcluster(tree, clusters, criterion="maxclust")
    res = Result("Cluster Observations")
    res.text(f"{tr(linkage)}    {tr(distance)}    {tr('Number of clusters')}: {clusters}")
    summary = pd.DataFrame({"Number of observations": pd.Series(membership).value_counts().sort_index()})
    centroids = frame.groupby(membership).mean()
    summary = pd.concat([summary, centroids], axis=1)
    summary.index.name = "Cluster"
    res.table(summary, title="Final Partition (cluster centroids)")
    res.figure(_dendrogram(tree, [str(r + 1) for r in rows], tr("Dendrogram"), clusters))
    if store:
        res.store("Cluster", _aligned(membership, rows, ws.nrows))
    return res


@analysis("mv.cluster_var", menu=MENU, title="Cluster Variables",
          params=[VARIABLES, Param("linkage", "Linkage method", "choice", default="Average",
                                   choices=[m for m in LINKAGE if m not in ("Ward", "Centroid", "Median")]),
                  Param("absolute", "Use absolute correlation as similarity", "bool", default=False),
                  Param("clusters", "Number of clusters", "int", default=2, minimum=1)])
def cluster_variables(ws, columns, linkage="Average", absolute=False, clusters=2):
    frame, _ = _data(ws, columns, minimum=3)
    corr = frame.corr().to_numpy()
    distance = 1 - (np.abs(corr) if absolute else corr)
    np.fill_diagonal(distance, 0.0)
    condensed = distance[np.triu_indices(len(columns), 1)]
    tree = hierarchy.linkage(np.clip(condensed, 0, None), SCIPY_LINKAGE[linkage])
    clusters = min(clusters, len(columns))
    membership = hierarchy.fcluster(tree, clusters, criterion="maxclust")
    res = Result("Cluster Variables")
    res.text(f"{tr(linkage)}    {tr('Distance')}: 1 - {'|r|' if absolute else 'r'}")
    res.table(pd.DataFrame({"Cluster": membership}, index=pd.Index(columns, name="Variable")),
              title="Final Partition", digits=0)
    return res.figure(_dendrogram(tree, list(columns), tr("Dendrogram"), clusters))


@analysis("mv.kmeans", menu=MENU, title="Cluster K-Means",
          params=[VARIABLES, Param("clusters", "Number of clusters", "int", default=3, minimum=2),
                  Param("standardize", "Standardize variables", "bool", default=True), STORE])
def k_means(ws, columns, clusters=3, standardize=True, store=False):
    from sklearn.cluster import KMeans

    frame, rows = _data(ws, columns, minimum=1)
    if clusters > len(frame):
        raise ValueError("There are more clusters than observations")
    X = frame.to_numpy()
    if standardize:
        sd = X.std(axis=0, ddof=1)
        X = (X - X.mean(axis=0)) / np.where(sd == 0, 1, sd)
    model = KMeans(n_clusters=clusters, n_init=20, random_state=0).fit(X)
    labels = model.labels_ + 1
    distances = np.linalg.norm(X - model.cluster_centers_[model.labels_], axis=1)
    summary = pd.DataFrame({
        "Number of observations": pd.Series(labels).value_counts().sort_index(),
        "Within cluster sum of squares": pd.Series(distances ** 2).groupby(labels).sum(),
        "Average distance from centroid": pd.Series(distances).groupby(labels).mean(),
        "Maximum distance from centroid": pd.Series(distances).groupby(labels).max()})
    summary.index.name = "Cluster"
    centroids = frame.groupby(labels).mean().T
    centroids.columns = [f"{tr('Cluster')}{c}" for c in centroids.columns]
    centroids[tr("Grand centroid")] = frame.mean()
    centroids.index.name = "Variable"
    res = Result("Cluster K-Means")
    res.table(summary, title="Final Partition").table(centroids, title="Cluster Centroids")
    if store:
        res.store("Cluster", _aligned(labels, rows, ws.nrows))
    return res


DISCRIMINANT = ["Linear", "Quadratic"]


@analysis("mv.discriminant", menu=MENU, title="Discriminant Analysis",
          params=[Param("groups", "Groups", "column", numeric=False), Param("predictors", "Predictors", "columns"),
                  Param("kind", "Discriminant function", "choice", default="Linear", choices=DISCRIMINANT),
                  Param("cross_validate", "Use cross-validation", "bool", default=False), STORE])
def discriminant(ws, groups, predictors, kind="Linear", cross_validate=False, store=False):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    predictors = [p for p in predictors if p != groups]
    if not predictors:
        raise ValueError("Select at least one predictor")
    rows = complete_rows(ws, [groups] + predictors)
    X = ws.df[predictors].iloc[rows].to_numpy(dtype=float)
    y = ws.df[groups].iloc[rows].map(format_value).to_numpy()
    levels = sorted(set(y))
    if len(levels) < 2:
        raise ValueError("At least two groups are needed")
    priors = np.full(len(levels), 1 / len(levels))
    model = (LinearDiscriminantAnalysis(priors=priors, store_covariance=True) if kind == "Linear"
             else QuadraticDiscriminantAnalysis(priors=priors))
    model.fit(X, y)
    predicted = cross_val_predict(model, X, y, cv=LeaveOneOut()) if cross_validate else model.predict(X)
    table = pd.crosstab(pd.Series(predicted, name="Put into group"), pd.Series(y, name="True group"))
    table = table.reindex(index=levels, columns=levels, fill_value=0)
    correct = np.diag(table.to_numpy())
    summary = table.copy()
    summary.loc["Total N"] = table.sum(axis=0)
    summary.loc["N correct"] = correct
    summary.loc["Proportion"] = correct / table.sum(axis=0).to_numpy()
    summary.index.name = "Put into group"
    res = Result("Discriminant Analysis")
    res.text(f"{tr(kind)}    {tr('Predictors')}: {', '.join(predictors)}"
             + (f"    {tr('Use cross-validation')}" if cross_validate else ""))
    res.table(summary, title="Summary of Classification", digits=3)
    res.text(f"N = {len(y)}    {tr('N correct')} = {int(correct.sum())}    "
             f"{tr('Proportion correct')} = {correct.sum() / len(y):.3f}")
    if kind == "Linear":
        # linear discriminant function for each group: constant + coefficients
        inverse = np.linalg.pinv(model.covariance_)
        coef = model.means_ @ inverse
        const = -0.5 * np.einsum("ij,ij->i", coef, model.means_)
        function = pd.DataFrame(np.vstack([const, coef.T]), index=pd.Index(["Constant"] + predictors, name="Term"),
                                columns=list(model.classes_))
        res.table(function, title="Linear Discriminant Function for Groups")
    if store:
        out = np.full(ws.nrows, None, dtype=object)
        out[rows] = predicted
        res.store("Predicted group", out.tolist())
    return res


def correspondence(table: np.ndarray) -> dict:
    """Simple correspondence analysis of a contingency table by singular value decomposition."""
    N = np.asarray(table, dtype=float)
    if (N < 0).any() or N.sum() <= 0:
        raise ValueError("Counts must be non-negative and not all zero")
    P = N / N.sum()
    r, c = P.sum(axis=1), P.sum(axis=0)
    if (r == 0).any() or (c == 0).any():
        raise ValueError("A row or column has a total of zero")
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    u, s, vt = np.linalg.svd(S, full_matrices=False)
    k = min(N.shape) - 1
    u, s, vt = u[:, :k], s[:k], vt[:k]
    return {"inertia": s ** 2, "total": float((s ** 2).sum()), "rows": (u / np.sqrt(r)[:, None]) * s,
            "cols": (vt.T / np.sqrt(c)[:, None]) * s, "row_mass": r, "col_mass": c}


def _correspondence_report(res: Result, out: dict, row_names, col_names, title: str) -> Result:
    k = len(out["inertia"])
    axes = [f"{tr('Axis')} {i + 1}" for i in range(k)]
    res.table(pd.DataFrame({"Inertia": out["inertia"], "Proportion": out["inertia"] / out["total"],
                            "Cumulative": np.cumsum(out["inertia"]) / out["total"]},
                           index=pd.Index(axes, name="Axis")), title="Analysis of Contingency Table")
    shown = min(k, 2)
    rows = pd.DataFrame(out["rows"][:, :shown], index=pd.Index(row_names, name="Name"), columns=axes[:shown])
    rows.insert(0, "Mass", out["row_mass"])
    res.table(rows, title="Row Contributions")
    cols = pd.DataFrame(out["cols"][:, :shown], index=pd.Index(col_names, name="Name"), columns=axes[:shown])
    cols.insert(0, "Mass", out["col_mass"])
    res.table(cols, title="Column Contributions")
    if k >= 2:
        fig = new_figure(6.5, 5.0)
        ax = fig.add_subplot()
        for coords, names, color, marker in ((out["rows"], row_names, BLUE, "o"), (out["cols"], col_names, RED, "s")):
            ax.plot(coords[:, 0], coords[:, 1], marker, color=color)
            for name, (x, y) in zip(names, coords[:, :2]):
                ax.annotate(str(name), (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color=GRAY, linewidth=0.8), ax.axvline(0, color=GRAY, linewidth=0.8)
        ax.set_xlabel(axes[0]), ax.set_ylabel(axes[1]), ax.set_title(title)
        res.figure(fig)
    return res


@analysis("mv.correspondence", menu=MENU, title="Simple Correspondence Analysis",
          params=[Param("columns", "Columns containing the table", "columns"),
                  Param("labels", "Row labels", "column", numeric=False, optional=True)])
def simple_correspondence(ws, columns, labels=None):
    if len(columns) < 2:
        raise ValueError("Select at least 2 columns")
    frame = ws.df[list(columns) + ([labels] if labels else [])].dropna()
    if len(frame) < 2:
        raise ValueError("The table needs at least two rows and two columns")
    names = frame[labels].map(format_value).tolist() if labels else [str(i + 1) for i in range(len(frame))]
    out = correspondence(frame[list(columns)].to_numpy(dtype=float))
    res = Result("Simple Correspondence Analysis")
    res.text(f"{tr('Total inertia')} = {out['total']:.5f}")
    return _correspondence_report(res, out, names, list(columns), tr("Symmetric Plot"))


@analysis("mv.multiple_correspondence", menu=MENU, title="Multiple Correspondence Analysis",
          params=[Param("columns", "Categorical variables", "columns", numeric=False)])
def multiple_correspondence(ws, columns):
    if len(columns) < 2:
        raise ValueError("Select at least 2 columns")
    frame = ws.df[list(columns)].dropna().map(format_value)
    frame = frame.apply(lambda col: pd.Categorical(col, categories=ws.levels(col.name, col)))
    indicator = pd.get_dummies(frame, prefix_sep=" = ").astype(float)
    if indicator.shape[1] - len(columns) < 2:
        raise ValueError("The variables need more categories for this analysis")
    out = correspondence(indicator.to_numpy())
    res = Result("Multiple Correspondence Analysis")
    res.text(f"N = {len(frame)}    {tr('Total inertia')} = {out['total']:.5f}")
    k = len(out["inertia"])
    axes = [f"{tr('Axis')} {i + 1}" for i in range(k)]
    res.table(pd.DataFrame({"Inertia": out["inertia"], "Proportion": out["inertia"] / out["total"],
                            "Cumulative": np.cumsum(out["inertia"]) / out["total"]},
                           index=pd.Index(axes, name="Axis")).head(10), title="Analysis of Indicator Matrix")
    shown = min(k, 2)
    cols = pd.DataFrame(out["cols"][:, :shown], index=pd.Index(indicator.columns, name="Category"),
                        columns=axes[:shown])
    cols.insert(0, "Mass", out["col_mass"])
    res.table(cols, title="Column Contributions")
    if k >= 2:
        fig = new_figure(6.5, 5.0)
        ax = fig.add_subplot()
        ax.plot(out["cols"][:, 0], out["cols"][:, 1], "s", color=RED)
        for name, (x, y) in zip(indicator.columns, out["cols"][:, :2]):
            ax.annotate(name, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color=GRAY, linewidth=0.8), ax.axvline(0, color=GRAY, linewidth=0.8)
        ax.set_xlabel(axes[0]), ax.set_ylabel(axes[1]), ax.set_title(tr("Column Plot"))
        res.figure(fig)
    return res
