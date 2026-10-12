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
        parts.append(pd.get_dummies(frame[name].map(format_value), prefix=name, prefix_sep=" = ").astype(float))
    return pd.concat(parts, axis=1), rows


def _make_model(kind: str, classify: bool, max_depth=4, min_leaf=3, trees=300, learning_rate=0.1, seed=1):
    from sklearn import ensemble, tree

    depth = max_depth or None
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


def _fit_report(title: str, kind: str, classify: bool, ws, response, continuous, categorical, folds, store,
                **options) -> Result:
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    X, rows = predictor_matrix(ws, response, continuous, categorical)
    raw = ws.df[response].iloc[rows]
    if classify:
        y = raw.map(format_value).to_numpy()
        classes = sorted(set(y))
        if len(classes) < 2:
            raise ValueError("The response needs at least two classes")
        smallest = pd.Series(y).value_counts().min()
    else:
        y = raw.to_numpy(dtype=float)
    model = _make_model(kind, classify, **options).fit(X, y)
    fitted = model.predict(X)
    res = Result(title)
    res.text(f"{tr('Response')}: {response}    N = {len(y)}    {tr('Predictors')}: {X.shape[1]}")
    summary = {}
    if classify:
        summary[tr("Training")] = classification_metrics(y, fitted, model.predict_proba(X), list(model.classes_))
    else:
        summary[tr("Training")] = regression_metrics(y, fitted)
    cv_predicted = None
    if folds >= 2:
        k = min(folds, len(y) if not classify else int(smallest))
        if k >= 2:
            splitter = (StratifiedKFold if classify else KFold)(n_splits=k, shuffle=True,
                                                                random_state=options.get("seed", 1))
            fresh = _make_model(kind, classify, **options)
            if classify:
                proba = cross_val_predict(fresh, X, y, cv=splitter, method="predict_proba")
                cv_predicted = np.asarray(classes)[proba.argmax(axis=1)]
                summary[f"{k}-fold"] = classification_metrics(y, cv_predicted, proba, classes)
            else:
                cv_predicted = cross_val_predict(fresh, X, y, cv=splitter)
                summary[f"{k}-fold"] = regression_metrics(y, cv_predicted)
    table = pd.DataFrame(summary)
    table.index.name = "Statistic"
    res.table(table, title="Model Summary")
    if classify:
        shown = cv_predicted if cv_predicted is not None else fitted
        confusion = pd.crosstab(pd.Series(y, name="Actual class"), pd.Series(shown, name="Predicted class"))
        confusion = confusion.reindex(index=classes, columns=classes, fill_value=0)
        confusion.index.name = "Actual class"
        if cv_predicted is not None:
            res.table(confusion, title="Confusion Matrix (cross-validation)")
        else:
            res.table(confusion, title="Confusion Matrix")
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
TREE_PARAMS = [CONTINUOUS, CATEGORICAL, DEPTH, MIN_LEAF, FOLDS, SEED, STORE]
FOREST_PARAMS = [CONTINUOUS, CATEGORICAL, TREES, DEPTH, MIN_LEAF, FOLDS, SEED, STORE]
BOOST_PARAMS = [CONTINUOUS, CATEGORICAL, TREES,
                Param("learning_rate", "Learning rate", "number", default=0.1, minimum=0.001, maximum=1.0),
                Param("max_depth", "Maximum tree depth", "int", default=3, minimum=1, maximum=20), MIN_LEAF, FOLDS,
                SEED, STORE]


@analysis("pred.cart_class", menu=MENU, title="CART Classification", params=[RESPONSE_CLASS] + TREE_PARAMS)
def cart_classification(ws, response, continuous=None, categorical=None, max_depth=4, min_leaf=3, folds=5, seed=1,
                        store=False):
    return _fit_report("CART Classification", "cart", True, ws, response, continuous, categorical, folds, store,
                       max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.cart_reg", menu=MENU, title="CART Regression", params=[RESPONSE_NUM] + TREE_PARAMS)
def cart_regression(ws, response, continuous=None, categorical=None, max_depth=4, min_leaf=3, folds=5, seed=1,
                    store=False):
    return _fit_report("CART Regression", "cart", False, ws, response, continuous, categorical, folds, store,
                       max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.forest_class", menu=MENU, title="Random Forest Classification",
          params=[RESPONSE_CLASS] + FOREST_PARAMS)
def forest_classification(ws, response, continuous=None, categorical=None, trees=300, max_depth=0, min_leaf=1,
                          folds=5, seed=1, store=False):
    return _fit_report("Random Forest Classification", "forest", True, ws, response, continuous, categorical, folds,
                       store, trees=trees, max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.forest_reg", menu=MENU, title="Random Forest Regression", params=[RESPONSE_NUM] + FOREST_PARAMS)
def forest_regression(ws, response, continuous=None, categorical=None, trees=300, max_depth=0, min_leaf=1, folds=5,
                      seed=1, store=False):
    return _fit_report("Random Forest Regression", "forest", False, ws, response, continuous, categorical, folds,
                       store, trees=trees, max_depth=max_depth, min_leaf=min_leaf, seed=seed)


@analysis("pred.boost_class", menu=MENU, title="Gradient Boosting Classification",
          params=[RESPONSE_CLASS] + BOOST_PARAMS)
def boosting_classification(ws, response, continuous=None, categorical=None, trees=300, learning_rate=0.1,
                            max_depth=3, min_leaf=3, folds=5, seed=1, store=False):
    return _fit_report("Gradient Boosting Classification", "boost", True, ws, response, continuous, categorical,
                       folds, store, trees=trees, learning_rate=learning_rate, max_depth=max_depth,
                       min_leaf=min_leaf, seed=seed)


@analysis("pred.boost_reg", menu=MENU, title="Gradient Boosting Regression", params=[RESPONSE_NUM] + BOOST_PARAMS)
def boosting_regression(ws, response, continuous=None, categorical=None, trees=300, learning_rate=0.1, max_depth=3,
                        min_leaf=3, folds=5, seed=1, store=False):
    return _fit_report("Gradient Boosting Regression", "boost", False, ws, response, continuous, categorical, folds,
                       store, trees=trees, learning_rate=learning_rate, max_depth=max_depth, min_leaf=min_leaf,
                       seed=seed)
