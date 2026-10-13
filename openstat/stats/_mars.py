"""Multivariate adaptive regression splines (Friedman, 1991).

Basis functions are products of hinges max(0, x - t) and max(0, t - x). A forward pass adds the
pair of hinges that most reduces the residual sum of squares; a backward pass then removes terms
one at a time and keeps the model with the smallest generalized cross-validation criterion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

MAX_KNOTS = 20   # knots tried for each variable and parent term


@dataclass(frozen=True)
class Term:
    factors: tuple = ()   # (variable index, knot, sign) for each hinge; empty for the constant

    def values(self, X: np.ndarray) -> np.ndarray:
        out = np.ones(len(X))
        for variable, knot, sign in self.factors:
            out = out * np.maximum(0.0, sign * (X[:, variable] - knot))
        return out

    def describe(self, names) -> str:
        if not self.factors:
            return "1"
        return " * ".join(f"max(0, {names[v]} - {t:.5g})" if s > 0 else f"max(0, {t:.5g} - {names[v]})"
                          for v, t, s in self.factors)


def gcv(rss: float, n: int, terms: int, penalty: float) -> float:
    """Generalized cross-validation: the mean squared residual inflated for the effective number of parameters."""
    effective = terms + penalty * (terms - 1) / 2
    return float("inf") if effective >= n else rss / n / (1 - effective / n) ** 2


def _rss(B: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef = np.linalg.lstsq(B, y, rcond=None)[0]
    return coef, float(((y - B @ coef) ** 2).sum())


def _best_pair(B, residual, Q, X, terms, max_degree):
    """The parent term, variable and knot whose pair of hinges explains most of the residual."""
    best = (0.0, None)
    for m, parent in enumerate(terms):
        if len(parent.factors) >= max_degree:
            continue
        active = B[:, m] > 0
        for v in range(X.shape[1]):
            if any(v == f[0] for f in parent.factors):
                continue
            knots = np.unique(X[active, v])[:-1]   # a knot at the largest value would give a zero column
            if not len(knots):
                continue
            if len(knots) > MAX_KNOTS:
                knots = np.unique(np.quantile(knots, np.linspace(0, 1, MAX_KNOTS), method="nearest"))
            gap = X[:, v][:, None] - knots[None, :]
            pair = [B[:, m][:, None] * np.maximum(0.0, sign * gap) for sign in (1, -1)]
            pair = [c - Q @ (Q.T @ c) for c in pair]          # what the model does not already span
            a, c, b = (pair[0] ** 2).sum(axis=0), (pair[1] ** 2).sum(axis=0), (pair[0] * pair[1]).sum(axis=0)
            g0, g1 = pair[0].T @ residual, pair[1].T @ residual
            det = a * c - b ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                both = (c * g0 ** 2 - 2 * b * g0 * g1 + a * g1 ** 2) / det
                single = np.maximum(np.where(a > 1e-10, g0 ** 2 / a, 0.0), np.where(c > 1e-10, g1 ** 2 / c, 0.0))
            gain = np.where(det > 1e-8 * np.maximum(a * c, 1e-300), both, single)
            k = int(np.nanargmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), (m, v, float(knots[k])))
    return best


def fit_terms(X, y, max_terms=21, max_degree=1) -> tuple[list[Term], np.ndarray, list[dict]]:
    """Forward and backward passes; returns the kept terms, their coefficients and the pruning path."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    penalty = 2.0 if max_degree == 1 else 3.0
    terms, B = [Term()], np.ones((n, 1))
    total = float(((y - y.mean()) ** 2).sum())
    while len(terms) + 2 <= max_terms and total > 0:
        Q = np.linalg.qr(B)[0]
        residual = y - Q @ (Q.T @ y)
        gain, choice = _best_pair(B, residual, Q, X, terms, max_degree)
        if choice is None or gain < 1e-6 * total:
            break
        m, v, knot = choice
        for sign in (1, -1):
            term = Term(terms[m].factors + ((v, knot, sign),))
            column = term.values(X)
            if column.any():   # a hinge that is zero on all the data adds nothing
                terms.append(term)
                B = np.column_stack([B, column])
    keep = list(range(len(terms)))
    coef, rss = _rss(B, y)
    path = [{"terms": len(keep), "rss": rss, "gcv": gcv(rss, n, len(keep), penalty), "keep": list(keep)}]
    while len(keep) > 1:
        trials = [(_rss(B[:, [k for k in keep if k != drop]], y)[1], drop) for drop in keep[1:]]
        rss, drop = min(trials)
        keep.remove(drop)
        path.append({"terms": len(keep), "rss": rss, "gcv": gcv(rss, n, len(keep), penalty), "keep": list(keep)})
    best = min(path, key=lambda step: (step["gcv"], step["terms"]))
    coef, _ = _rss(B[:, best["keep"]], y)
    return [terms[k] for k in best["keep"]], coef, path


class MarsRegressor(RegressorMixin, BaseEstimator):
    """MARS with the scikit-learn estimator interface, so that it can be cross-validated like the other models."""

    def __init__(self, max_terms=21, max_degree=1):
        self.max_terms = max_terms
        self.max_degree = max_degree

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.terms_, self.coef_, self.path_ = fit_terms(X, y, self.max_terms, self.max_degree)
        self.n_features_in_ = X.shape[1]
        basis = self._basis(X)
        self.rss_ = float(((y - basis @ self.coef_) ** 2).sum())
        self.gcv_ = gcv(self.rss_, len(y), len(self.terms_), 2.0 if self.max_degree == 1 else 3.0)
        # importance of a variable: what the fit loses when the terms that use it are taken out
        loss = np.zeros(X.shape[1])
        for v in range(X.shape[1]):
            others = [i for i, term in enumerate(self.terms_) if all(f[0] != v for f in term.factors)]
            if len(others) < len(self.terms_):
                loss[v] = _rss(basis[:, others], y)[1] - self.rss_
        self.feature_importances_ = np.maximum(loss, 0.0)
        return self

    def _basis(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([term.values(X) for term in self.terms_])

    def predict(self, X) -> np.ndarray:
        return self._basis(X) @ self.coef_
