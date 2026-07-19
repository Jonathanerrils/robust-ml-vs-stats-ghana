"""Time-series validation: expanding-window folds + stress-period holdout.

Nothing here shuffles, and every training set strictly precedes its test
set. The stress holdout trains on all data before 2024 and tests on the
2024 cocoa-shock year — the 'does it survive a regime it has never
seen?' test the paper leans on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


@dataclass
class Fold:
    name: str
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


def expanding_folds(index: pd.DatetimeIndex, n_folds: int = 5,
                    min_train_frac: float = 0.4) -> list[Fold]:
    n = len(index)
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_folds + 1, dtype=int)
    return [
        Fold(f"fold{i+1}", index[: edges[i]], index[edges[i]: edges[i + 1]])
        for i in range(n_folds)
    ]


def stress_holdout(index: pd.DatetimeIndex,
                   start: str = "2024-01-01", end: str = "2024-12-31") -> Fold:
    return Fold("stress_2024", index[index < start],
                index[(index >= start) & (index <= end)])


def score(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.nan_to_num(p, nan=float(np.nanmean(p))), 1e-6, 1 - 1e-6)
    out = {"brier": brier_score_loss(y_true, p)}
    if len(np.unique(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, p)
        thr = np.quantile(p, 1 - y_true.mean())  # match predicted to base rate
        out["balanced_acc"] = balanced_accuracy_score(y_true, p >= thr)
    else:
        out["auc"] = np.nan
        out["balanced_acc"] = np.nan
    return out


def run_models(models: list, X: pd.DataFrame, y: pd.Series,
               folds: list[Fold], returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in folds:
        Xtr, ytr = X.loc[fold.train_idx], y.loc[fold.train_idx]
        Xte, yte = X.loc[fold.test_idx], y.loc[fold.test_idx]
        if yte.sum() == 0 or ytr.sum() == 0:
            continue
        for m in models:
            m.fit(Xtr, ytr, returns=returns)
            p = np.asarray(m.predict_proba(Xte, returns=returns), dtype=float)
            rows.append({"fold": fold.name, "model": m.name,
                         "n_test": len(yte), "event_rate": float(yte.mean()),
                         **score(yte.to_numpy(), p)})
    return pd.DataFrame(rows)
