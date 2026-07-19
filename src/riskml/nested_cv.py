"""Nested time-series cross-validation.

For each outer expanding-window fold, hyperparameters are selected by an
inner expanding-window CV split of the outer training set only -- never
touching the outer test set -- then the winning configuration is refit
on the full outer training set before scoring on the outer test set.

Statistical models (base rate, GARCH-t, VAR, Markov-switching) are not
tuned here: VAR already self-selects its lag order by AIC, Markov-
switching is fixed at 2 states as a deliberate, disclosed scope choice,
and the other two expose no hyperparameters in this design.

Minimum-viable-positives fallback: an inner fold whose test split would
contain fewer than MIN_INNER_POSITIVES positive examples produces an
unreliable tuning signal (a handful of positives can flip an AUC
comparison almost arbitrarily). Rather than tune on a signal known in
advance to be noisy, any outer fold whose inner split fails this check
for ANY of its inner folds skips nested tuning entirely and uses a
fixed, documented default configuration instead -- logged explicitly,
not silently substituted.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import models as models_mod
from .validation import Fold, expanding_folds

MIN_INNER_POSITIVES = 10
N_INNER_FOLDS = 3

# Modest grids (2-4 values per hyperparameter), not exhaustive sweeps --
# sized to keep the nested loop's total fit count proportionate, per the
# scope agreed before building this.
PARAM_GRIDS: dict[str, list[dict]] = {
    "logistic": [{"clf__C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "lasso": [{"clf__C": c} for c in (0.01, 0.1, 1.0)],
    "elastic_net": [{"clf__C": c} for c in (0.01, 0.1, 1.0)],
    "random_forest": [{"min_samples_leaf": m} for m in (10, 25, 50)],
    "xgboost": [{"max_depth": d, "min_child_weight": w}
                for d in (2, 3) for w in (3, 10)],
    "svm": [{"clf__C": c, "clf__gamma": g}
            for c in (0.5, 2.0) for g in ("scale", 0.01)],
    "mlp": [{"clf__hidden_layer_sizes": h, "clf__alpha": a}
            for h in ((8,), (16,)) for a in (0.5, 2.0)],
}
DEFAULT_PARAMS: dict[str, dict] = {name: grid[len(grid) // 2]
                                   for name, grid in PARAM_GRIDS.items()}


def inner_folds_viable(idx: pd.DatetimeIndex, y: pd.Series) -> tuple[bool, list[Fold]]:
    """Build N_INNER_FOLDS expanding-window folds within idx; return
    (viable, folds). viable=False if any inner test split has fewer
    than MIN_INNER_POSITIVES positive examples."""
    inner = expanding_folds(idx, n_folds=N_INNER_FOLDS, min_train_frac=0.4)
    for f in inner:
        if y.loc[f.test_idx].sum() < MIN_INNER_POSITIVES:
            return False, inner
    return True, inner


def tune_model(name: str, factory, X: pd.DataFrame, y: pd.Series,
              train_idx: pd.DatetimeIndex, returns: pd.DataFrame) -> tuple[dict, bool]:
    """Return (best_params, was_tuned). was_tuned=False means the inner
    split was not viable and the documented default was used instead."""
    grid = PARAM_GRIDS.get(name)
    if grid is None:  # model has nothing to tune
        return {}, True

    viable, inner = inner_folds_viable(train_idx, y)
    if not viable:
        return DEFAULT_PARAMS[name], False

    mean_auc = []
    for params in grid:
        aucs = []
        for f in inner:
            m = factory()
            est = m.est if hasattr(m, "est") else m
            est.set_params(**params)
            m.fit(X.loc[f.train_idx], y.loc[f.train_idx], returns=returns)
            p = m.predict_proba(X.loc[f.test_idx], returns=returns)
            yt = y.loc[f.test_idx]
            if yt.nunique() > 1:
                aucs.append(roc_auc_score(yt, p))
        mean_auc.append(np.mean(aucs) if aucs else -np.inf)
    best = grid[int(np.argmax(mean_auc))]
    return best, True


def run_nested_cv(X: pd.DataFrame, y: pd.Series, returns: pd.DataFrame,
                  target: str, outer_folds: list[Fold]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full nested-CV sweep for one target across all outer folds and
    the full model zoo. Returns (summary_rows, oof_predictions), where
    oof_predictions has one row per (date, model) with the predicted
    probability and actual outcome -- needed for paired significance
    testing on AUC differences, not just point-estimate comparison."""
    rows = []
    oof_rows = []
    factories = {
        "logistic": models_mod.make_logistic,
        "lasso": models_mod.make_lasso,
        "elastic_net": models_mod.make_elastic_net,
        "random_forest": models_mod.make_random_forest,
        "xgboost": models_mod.make_xgboost,
        "svm": models_mod.make_svm,
        "mlp": models_mod.make_mlp,
    }

    for fold in outer_folds:
        ytr, yte = y.loc[fold.train_idx], y.loc[fold.test_idx]
        if ytr.sum() == 0 or yte.nunique() < 2:
            continue

        for m in [models_mod.BaseRateModel(),
                  models_mod.GarchVarModel(target_col="cocoa"),
                  models_mod.VarExceedanceModel(target_col="cocoa"),
                  models_mod.MarkovSwitchingVarModel(target_col="cocoa")]:
            m.fit(X.loc[fold.train_idx], ytr, returns=returns)
            p = m.predict_proba(X.loc[fold.test_idx], returns=returns)
            rows.append({"fold": fold.name, "model": m.name, "target": target,
                        "tuned": None, "best_params": None,
                        "auc": roc_auc_score(yte, p) if yte.nunique() > 1 else np.nan,
                        "n_train": len(ytr), "n_test": len(yte)})
            for dt, pi, yi in zip(fold.test_idx, p, yte):
                oof_rows.append({"date": dt, "model": m.name, "target": target,
                                "p": pi, "y": yi, "fold": fold.name})

        for name, factory in factories.items():
            best_params, was_tuned = tune_model(
                name, lambda n=name, f=factory: models_mod.SklearnModel(n, f()),
                X, y, fold.train_idx, returns)
            m = models_mod.SklearnModel(name, factory())
            if best_params:
                m.est.set_params(**best_params)
            m.fit(X.loc[fold.train_idx], ytr, returns=returns)
            p = m.predict_proba(X.loc[fold.test_idx], returns=returns)
            rows.append({"fold": fold.name, "model": name, "target": target,
                        "tuned": was_tuned, "best_params": str(best_params),
                        "auc": roc_auc_score(yte, p) if yte.nunique() > 1 else np.nan,
                        "n_train": len(ytr), "n_test": len(yte)})
            for dt, pi, yi in zip(fold.test_idx, p, yte):
                oof_rows.append({"date": dt, "model": name, "target": target,
                                "p": pi, "y": yi, "fold": fold.name})
        print(f"  [{target}] fold {fold.name} done "
              f"(train n={len(ytr)}, test n={len(yte)})", flush=True)

    return pd.DataFrame(rows), pd.DataFrame(oof_rows)
