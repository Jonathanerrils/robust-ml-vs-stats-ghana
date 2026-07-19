"""Interpretability layer: permutation importance, SHAP, PDPs, stability.

Feature *stability* — do the same variables matter in every fold? — is
treated as a first-class result, since an importance ranking that
reshuffles across time windows is not a usable risk signal.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def perm_importance(model, X: pd.DataFrame, y: pd.Series,
                    n_repeats: int = 10, random_state: int = 0) -> pd.Series:
    r = permutation_importance(model, X, y, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=random_state,
                               n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def shap_importance(model, X: pd.DataFrame, max_samples: int = 800) -> pd.Series:
    """Mean |SHAP| per feature for tree models (TreeExplainer)."""
    import shap

    Xs = X.iloc[-max_samples:]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):  # binary classifiers may return [class0, class1]
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(
        ascending=False)


def stability(importances_by_fold: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise Spearman rank correlation of importance vectors across folds."""
    folds = list(importances_by_fold)
    m = pd.DataFrame(np.eye(len(folds)), index=folds, columns=folds)
    for i, a in enumerate(folds):
        for b in folds[i + 1:]:
            common = importances_by_fold[a].index.intersection(
                importances_by_fold[b].index)
            rho = stats.spearmanr(importances_by_fold[a][common],
                                  importances_by_fold[b][common]).statistic
            m.loc[a, b] = m.loc[b, a] = rho
    return m


def plot_top_importance(imp: pd.Series, title: str, path: str, k: int = 15):
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.head(k)[::-1].plot.barh(ax=ax, color="#2F6B4F")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pdp(model, X: pd.DataFrame, features: list[str], path: str):
    fig, ax = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.6))
    PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
    fig.suptitle("Partial dependence — event probability vs feature")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
