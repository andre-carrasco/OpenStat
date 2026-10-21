"""Predictive analytics: classification and regression trees, random forests and gradient boosting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openstat.core.registry import Param, analysis
from openstat.core.result import Result
from openstat.core.worksheet import format_value
from openstat.graphs import new_figure
from openstat.i18n import tr
from openstat.stats._design import complete_rows

MENU = ("Stat", "Predictive Analytics")
BLUE = "#5b9bd5"

CONTINUOUS = Param("continuous", "Continuous predictors", "columns", optional=True)
CATEGORICAL = Param("categorical", "Categorical predictors", "columns", numeric=False, optional=True)
FOLDS = Param("folds", "K-fold cross-validation (0 = none)", "int", default=5, minimum=0, maximum=50)
STORE = Param("store", "Store fitted values", "bool", default=False)
DEPTH = Param("max_depth", "Maximum tree depth (0 = no limit)", "int", default=4, minimum=0, maximum=50)
MIN_LEAF = Param("min_leaf", "Minimum cases in a terminal node", "int", default=3, minimum=1, maximum=100000)
TREES = Param("trees", "Number of trees", "int", default=300, minimum=10, maximum=5000)
SEED = Param("seed", "Base for random numbers", "int", default=1, minimum=0)


def predictor_matrix(ws, response, continuous, categorical) -> tuple[pd.DataFrame, np.ndarray]:
    """Predictors with categorical columns expanded to indicators; returns the matrix and the rows used."""
    continuous = [c for c in (continuous or []) if c != response]
    categorical = [c for c in (categorical or []) if c != response and c not in continuous]
    if not continuous and not categorical:
        raise ValueError("Select at least one predictor")
    rows = complete_rows(ws, [response] + continuous + categorical)
    if len(rows) < 10:
        raise ValueError("At least ten complete rows are needed")
    frame = ws.df.iloc[rows]
    parts = [frame[continuous].astype(float)] if continuous else []
    for name in categorical:
        labels = frame[name].map(format_value)
        labels = pd.Categorical(labels, categories=ws.levels(name, labels))
        parts.append(pd.get_dummies(labels, prefix=name, prefix_sep=" = ").astype(float).set_axis(frame.index))
    return pd.concat(parts, axis=1), rows


def _make_model(kind: str, classify: bool, max_depth=4, min_leaf=3, trees=300, learning_rate=0.1, seed=1,
                max_terms=21, max_degree=1):
    from sklearn import ensemble, linear_model, pipeline, preprocessing, tree

    depth = max_depth or None
    if kind == "mars":
        from openstat.stats._mars import MarsRegressor

        return MarsRegressor(max_terms=max_terms, max_degree=max_degree)
    if kind == "linear":
        if classify:   # standardized predictors keep the iterations of the logistic fit well behaved
            return pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression(max_iter=2000))
        return linear_model.LinearRegression()
    if kind == "cart":
        cls = tree.DecisionTreeClassifier if classify else tree.DecisionTreeRegressor
        return cls(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
    if kind == "forest":
        cls = ensemble.RandomForestClassifier if classify else ensemble.RandomForestRegressor
        return cls(n_estimators=trees, max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
    cls = ensemble.GradientBoostingClassifier if classify else ensemble.GradientBoostingRegressor
    return cls(n_estimators=trees, max_depth=max_depth or 3, min_samples_leaf=min_leaf,
               learning_rate=learning_rate, random_state=seed)


def regression_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    error = actual - predicted
    total = ((actual - actual.mean()) ** 2).sum()
    return {"R-sq": 100 * (1 - (error ** 2).sum() / total) if total > 0 else np.nan,
            "RMSE": float(np.sqrt(np.mean(error ** 2))), "MSE": float(np.mean(error ** 2)),
            "MAD": float(np.mean(np.abs(error)))}


def classification_metrics(actual, predicted, probability=None, classes=None) -> dict:
    from sklearn.metrics import roc_auc_score

    actual, predicted = np.asarray(actual), np.asarray(predicted)
    out = {"N": len(actual), "Correct": int((actual == predicted).sum()),
           "Accuracy (%)": 100 * float((actual == predicted).mean()),
           "Misclassification (%)": 100 * float((actual != predicted).mean())}
    if probability is not None and classes is not None and len(classes) == 2:
        out["Area under ROC curve"] = float(roc_auc_score(actual == classes[1], probability[:, 1]))
    return out


def importance_figure(names, values, title: str):
    relative = 100 * values / values.max() if values.max() > 0 else values
    order = np.argsort(relative)[-25:]
    fig = new_figure(7.0, max(3.0, 0.3 * len(order) + 1.4))
    ax = fig.add_subplot()
    ax.barh([names[i] for i in order], relative[order], color=BLUE)
    ax.set_xlabel(tr("Relative importance (%)")), ax.set_title(title)
    return fig


def response_values(ws, response, rows, classify: bool):
    """The response as class labels or numbers, and the sorted classes (None for a numeric response)."""
    raw = ws.df[response].iloc[rows]
    if not classify:
        if not ws.is_numeric(response):
            raise ValueError("The response must be numeric")
        return raw.to_numpy(dtype=float), None
    y = raw.map(format_value).to_numpy()
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("The response needs at least two classes")
    return y, classes


def holdout_split(n: int, percent: float, seed: int) -> np.ndarray:
    """True for the rows used in training; `percent` of the rows, chosen at random, are held out."""
    train = np.ones(n, dtype=bool)
    if percent > 0:
        held = int(round(n * percent / 100))
        if held < 2 or n - held < 10:
            raise ValueError("The test set leaves too few rows to train or to test the model")
        train[np.random.default_rng(seed).choice(n, held, replace=False)] = False
    return train


def cross_validated(kind: str, classify: bool, X, y, folds: int, classes=None, **options):
    """Out-of-fold predictions (and class probabilities), or None when the data do not allow `folds`."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    k = min(folds, int(pd.Series(y).value_counts().min()) if classify else len(y))
    if k < 2:
        return None, None, k
    splitter = (StratifiedKFold if classify else KFold)(n_splits=k, shuffle=True, random_state=options.get("seed", 1))
    fresh = _make_model(kind, classify, **options)
    if classify:
        proba = cross_val_predict(fresh, X, y, cv=splitter, method="predict_proba")
        return np.asarray(classes)[proba.argmax(axis=1)], proba, k
    return cross_val_predict(fresh, X, y, cv=splitter), None, k


def _fit_report(title: str, kind: str, classify: bool, ws, response, continuous, categorical, folds, store,
                holdout=0.0, **options) -> Result:
    X, rows = predictor_matrix(ws, response, continuous, categorical)
    y, classes = response_values(ws, response, rows, classify)
    train = holdout_split(len(y), holdout, options.get("seed", 1))
    model = _make_model(kind, classify, **options).fit(X[train], y[train])
    fitted = model.predict(X)
    res = Result(title)
    res.text(f"{tr('Response')}: {response}    N = {len(y)}    {tr('Predictors')}: {X.shape[1]}"
             + (f"    {tr('Test set')}: {int((~train).sum())}" if holdout > 0 else ""))

    def metrics(part, predicted, probability):
        if classify:
            return classification_metrics(y[part], predicted, probability, classes)
        return regression_metrics(y[part], predicted)

    probability = model.predict_proba(X) if classify else None
    summary = {tr("Training"): metrics(train, fitted[train], probability[train] if classify else None)}
    cv_predicted, cv_probability, k = cross_validated(kind, classify, X[train], y[train], folds, classes, **options) \
        if folds >= 2 else (None, None, 0)
    if cv_predicted is not None:
        summary[f"{k}-fold"] = metrics(train, cv_predicted, cv_probability)
    if holdout > 0:
        summary[tr("Test")] = metrics(~train, fitted[~train], probability[~train] if classify else None)
    table = pd.DataFrame(summary)
    table.index.name = "Statistic"
    res.table(table, title="Model Summary")
    if classify:
        if holdout > 0:
            actual, shown, name = y[~train], fitted[~train], "Confusion Matrix (test set)"
        elif cv_predicted is not None:
            actual, shown, name = y, cv_predicted, "Confusion Matrix (cross-validation)"
        else:
            actual, shown, name = y, fitted, "Confusion Matrix"
        confusion = pd.crosstab(pd.Series(actual, name="Actual class"), pd.Series(shown, name="Predicted class"))
        confusion = confusion.reindex(index=classes, columns=classes, fill_value=0)
        confusion.index.name = "Actual class"
        res.table(confusion, title=name)
    if kind == "mars":
        basis = pd.DataFrame({"Coef": model.coef_},
                             index=pd.Index([t.describe(list(X.columns)) for t in model.terms_], name="Basis function"))
        res.table(basis, title="Basis Functions")
        res.text(f"{tr('Basis functions')}: {len(model.terms_)}    GCV = {model.gcv_:.6g}    "
                 f"{tr('Maximum interaction order')}: {options.get('max_degree', 1)}")
    importance = np.asarray(model.feature_importances_)
    ranked = pd.DataFrame({"Relative importance (%)": 100 * importance / importance.max()
                           if importance.max() > 0 else importance}, index=pd.Index(X.columns, name="Variable"))
    res.table(ranked.sort_values("Relative importance (%)", ascending=False), title="Variable Importance", digits=1)
    res.figure(importance_figure(list(X.columns), importance, tr("Relative Variable Importance")))
    if kind == "cart":
        from sklearn.tree import plot_tree

        leaves = model.get_n_leaves()
        res.text(f"{tr('Terminal nodes')}: {leaves}    {tr('Tree depth')}: {model.get_depth()}")
        fig = new_figure(min(22.0, max(8.0, 1.6 * leaves)), min(14.0, max(4.5, 1.7 * (model.get_depth() + 1))))
        plot_tree(model, feature_names=list(X.columns), class_names=[str(c) for c in model.classes_] if classify
                  else None, filled=True, rounded=True, impurity=False, fontsize=8, ax=fig.add_subplot())
        res.figure(fig)
    if store:
        out = np.full(ws.nrows, None if classify else np.nan, dtype=object if classify else float)
        out[rows] = fitted
        res.store(f"PFITS_{response}", out.tolist())
    return res


RESPONSE_CLASS = Param("response", "Response (categories)", "column", numeric=False)
RESPONSE_NUM = Param("response", "Response", "column")
HOLDOUT = Param("holdout", "Rows held out as a test set (%, 0 = none)", "number", default=0.0, minimum=0.0,
                maximum=90.0)
TREE_PARAMS = [CONTINUOUS, CATEGORICAL, DEPTH, MIN_LEAF, FOLDS, SEED, STORE, HOLDOUT]
FOREST_PARAMS = [CONTINUOUS, CATEGORICAL, TREES, DEPTH, MIN_LEAF, FOLDS, SEED, STORE, HOLDOUT]
BOOST_PARAMS = [CONTINUOUS, CATEGORICAL, TREES,
                Param("learning_rate", "Learning rate", "number", default=0.1, minimum=0.001, maximum=1.0),
                Param("max_depth", "Maximum tree depth", "int", default=3, minimum=1, maximum=20), MIN_LEAF, FOLDS,
                SEED, STORE, HOLDOUT]


@analysis("pred.cart_class", menu=MENU, title="CART Classification", params=[RESPONSE_CLASS] + TREE_PARAMS)
def cart_classification(ws, response, continuous=None, categorical=None, max_depth=4, min_leaf=3, folds=5, seed=1,
                        store=False, holdout=0.0):
    return _fit_report("CART Classification", "cart", True, ws, response, continuous, categorical, folds, store,
                       holdout, max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.cart_reg", menu=MENU, title="CART Regression", params=[RESPONSE_NUM] + TREE_PARAMS)
def cart_regression(ws, response, continuous=None, categorical=None, max_depth=4, min_leaf=3, folds=5, seed=1,
                    store=False, holdout=0.0):
    return _fit_report("CART Regression", "cart", False, ws, response, continuous, categorical, folds, store,
                       holdout, max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.forest_class", menu=MENU, title="Random Forest Classification",
          params=[RESPONSE_CLASS] + FOREST_PARAMS)
def forest_classification(ws, response, continuous=None, categorical=None, trees=300, max_depth=0, min_leaf=1,
                          folds=5, seed=1, store=False, holdout=0.0):
    return _fit_report("Random Forest Classification", "forest", True, ws, response, continuous, categorical, folds,
                       store, holdout, trees=trees, max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.forest_reg", menu=MENU, title="Random Forest Regression", params=[RESPONSE_NUM] + FOREST_PARAMS)
def forest_regression(ws, response, continuous=None, categorical=None, trees=300, max_depth=0, min_leaf=1, folds=5,
                      seed=1, store=False, holdout=0.0):
    return _fit_report("Random Forest Regression", "forest", False, ws, response, continuous, categorical, folds,
                       store, holdout, trees=trees, max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.boost_class", menu=MENU, title="Gradient Boosting Classification",
          params=[RESPONSE_CLASS] + BOOST_PARAMS)
def boosting_classification(ws, response, continuous=None, categorical=None, trees=300, learning_rate=0.1,
                            max_depth=3, min_leaf=3, folds=5, seed=1, store=False, holdout=0.0):
    return _fit_report("Gradient Boosting Classification", "boost", True, ws, response, continuous, categorical,
                       folds, store, holdout, trees=trees, learning_rate=learning_rate, max_depth=max_depth,
                       min_leaf=min_leaf, seed=seed)


@analysis("pred.boost_reg", menu=MENU, title="Gradient Boosting Regression", params=[RESPONSE_NUM] + BOOST_PARAMS)
def boosting_regression(ws, response, continuous=None, categorical=None, trees=300, learning_rate=0.1, max_depth=3,
                        min_leaf=3, folds=5, seed=1, store=False, holdout=0.0):
    return _fit_report("Gradient Boosting Regression", "boost", False, ws, response, continuous, categorical, folds,
                       store, holdout, trees=trees, learning_rate=learning_rate, max_depth=max_depth,
                       min_leaf=min_leaf, seed=seed)


@analysis("pred.mars", menu=MENU, title="MARS Regression",
          params=[RESPONSE_NUM, CONTINUOUS, CATEGORICAL,
                  Param("max_terms", "Maximum number of basis functions", "int", default=21, minimum=3, maximum=200),
                  Param("max_degree", "Maximum interaction order", "int", default=1, minimum=1, maximum=3), FOLDS,
                  SEED, STORE, HOLDOUT])
def mars_regression(ws, response, continuous=None, categorical=None, max_terms=21, max_degree=1, folds=5, seed=1,
                    store=False, holdout=0.0):
    return _fit_report("MARS Regression", "mars", False, ws, response, continuous, categorical, folds, store,
                       holdout, max_terms=max_terms, max_degree=max_degree, seed=seed)


# ----- comparing models and choosing predictors -----

MODELS = {"Linear regression": "linear", "Logistic regression": "linear", "CART": "cart", "Random forest": "forest",
          "Gradient boosting": "boost", "MARS": "mars"}
RESPONSE_KINDS = ["Continuous", "Categorical"]
RESPONSE_ANY = Param("response", "Response", "column", numeric=False)
RESPONSE_KIND = Param("kind", "Type of response", "choice", default=RESPONSE_KINDS[0], choices=RESPONSE_KINDS)
FEWER_TREES = Param("trees", "Number of trees", "int", default=100, minimum=10, maximum=5000)
CV_FOLDS = Param("folds", "K-fold cross-validation", "int", default=5, minimum=2, maximum=50)


def compare_models(X, y, classify: bool, classes=None, folds=5, trees=100, seed=1) -> pd.DataFrame:
    """Cross-validated performance of each kind of model, best first."""
    names = ["Logistic regression", "CART", "Random forest", "Gradient boosting"] if classify else \
        ["Linear regression", "CART", "Random forest", "Gradient boosting", "MARS"]
    rows = {}
    for name in names:
        options = {"seed": seed, **({"trees": trees} if MODELS[name] in ("forest", "boost") else {})}
        predicted, probability, k = cross_validated(MODELS[name], classify, X, y, folds, classes, **options)
        if predicted is None:
            raise ValueError("There are too few rows in a class for cross-validation")
        rows[name] = classification_metrics(y, predicted, probability, classes) if classify \
            else regression_metrics(y, predicted)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Model"
    return table.sort_values("Accuracy (%)" if classify else "R-sq", ascending=False, kind="stable")


@analysis("pred.compare", menu=MENU, title="Compare Models (Automated)",
          params=[RESPONSE_ANY, RESPONSE_KIND, CONTINUOUS, CATEGORICAL, CV_FOLDS, FEWER_TREES, SEED])
def compare(ws, response, kind=RESPONSE_KINDS[0], continuous=None, categorical=None, folds=5, trees=100, seed=1):
    classify = kind == RESPONSE_KINDS[1]
    X, rows = predictor_matrix(ws, response, continuous, categorical)
    y, classes = response_values(ws, response, rows, classify)
    table = compare_models(X, y, classify, classes, folds, trees, seed)
    score = "Accuracy (%)" if classify else "R-sq"
    fig = new_figure(7.0, 3.6)
    ax = fig.add_subplot()
    ax.barh([tr(name) for name in table.index[::-1]], table[score][::-1], color=BLUE)
    ax.set_xlabel(f"{tr(score)} ({folds}-fold)"), ax.set_title(tr("Compare Models (Automated)"))
    res = Result("Compare Models (Automated)")
    res.text(f"{tr('Response')}: {response}    N = {len(y)}    {tr('Predictors')}: {X.shape[1]}    "
             f"{tr('Cross-validated performance, best model first')}")
    res.table(table, digits=3)
    res.text(f"{tr('Best model')}: {tr(table.index[0])}")
    return res.figure(fig)


def key_predictors(ws, response, classify: bool, continuous, categorical, folds=5, trees=100, seed=1) -> pd.DataFrame:
    """Backward elimination with random forests: the least important predictor leaves at each step
    and the cross-validated performance of what remains is recorded."""
    continuous = [c for c in (continuous or []) if c != response]
    categorical = [c for c in (categorical or []) if c != response and c not in continuous]
    remaining = continuous + categorical
    rows = []
    while remaining:
        X, used = predictor_matrix(ws, response, [c for c in continuous if c in remaining],
                                   [c for c in categorical if c in remaining])
        y, classes = response_values(ws, response, used, classify)
        predicted, probability, _ = cross_validated("forest", classify, X, y, folds, classes, trees=trees, seed=seed)
        if predicted is None:
            raise ValueError("There are too few rows in a class for cross-validation")
        score = classification_metrics(y, predicted)["Accuracy (%)"] if classify else regression_metrics(y, predicted)["R-sq"]
        importance = pd.Series(_make_model("forest", classify, trees=trees, seed=seed).fit(X, y).feature_importances_,
                               index=X.columns)
        # the indicator columns of a categorical predictor count together
        by_predictor = {name: float(importance[[c for c in X.columns if c == name or c.startswith(f"{name} = ")]].sum())
                        for name in remaining}
        weakest = min(remaining, key=lambda name: by_predictor[name])
        rows.append({"Predictors": len(remaining), "Score": score, "Least important": weakest,
                     "In the model": ", ".join(remaining)})
        remaining = [name for name in remaining if name != weakest]
    return pd.DataFrame(rows)


@analysis("pred.key_predictors", menu=MENU, title="Discover Key Predictors",
          params=[RESPONSE_ANY, RESPONSE_KIND, CONTINUOUS, CATEGORICAL, CV_FOLDS, FEWER_TREES, SEED])
def discover_key_predictors(ws, response, kind=RESPONSE_KINDS[0], continuous=None, categorical=None, folds=5,
                            trees=100, seed=1):
    classify = kind == RESPONSE_KINDS[1]
    table = key_predictors(ws, response, classify, continuous, categorical, folds, trees, seed)
    score = "Accuracy (%)" if classify else "R-sq"
    best = table.sort_values(["Score", "Predictors"], ascending=[False, True], kind="stable").iloc[0]
    fig = new_figure(7.0, 3.8)
    ax = fig.add_subplot()
    ax.plot(table["Predictors"], table["Score"], "-o", color=BLUE)
    ax.plot(best["Predictors"], best["Score"], "s", color="#c00000", markersize=8)
    ax.set_xlabel(tr("Number of predictors")), ax.set_ylabel(f"{tr(score)} ({folds}-fold)")
    ax.set_title(tr("Discover Key Predictors"))
    res = Result("Discover Key Predictors")
    res.text(tr("Random forests are fitted while the least important predictor is removed at each step."))
    res.table(table.rename(columns={"Score": score}), index=False, digits=3)
    res.text(f"{tr('Key predictors')} ({int(best['Predictors'])}): {best['In the model']}")
    return res.figure(fig)
