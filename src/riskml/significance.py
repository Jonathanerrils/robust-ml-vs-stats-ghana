"""Significance testing on AUC differences between models, using pooled
out-of-fold predictions from the nested-CV run.

Because outer folds are non-overlapping and time-ordered, pooling every
model's out-of-fold predictions into one series per model gives, for
each date, a matched pair of predicted probabilities from any two
models being compared, evaluated against the same realized outcome --
exactly the paired structure a significance test on AUC differences
needs (a plain unpaired test would ignore that both models are scored
on identical observations, throwing away the noise-reduction that
comes from comparing predictions on the same events).

Significance uses a moving-block bootstrap (block length 20 trading
days, matching the block length already used throughout the companion
tail-dependence project's inference stage, for consistency and because
20 days exceeds the effective autocorrelation length of these daily
series while remaining small relative to the ~3,000-observation pooled
sample) rather than a naive i.i.d. bootstrap, since these are serially
dependent daily financial observations, not independent draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _block_resample(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx[idx < n]


def auc_difference_test(oof: pd.DataFrame, target: str, model_a: str, model_b: str,
                        n_boot: int = 2000, block: int = 20,
                        seed: int = 0) -> dict:
    """Paired block-bootstrap test of AUC(model_a) - AUC(model_b) = 0,
    on the pooled out-of-fold predictions for one target."""
    da = oof[(oof.target == target) & (oof.model == model_a)].set_index("date")
    db = oof[(oof.target == target) & (oof.model == model_b)].set_index("date")
    common = da.index.intersection(db.index)
    da, db = da.loc[common], db.loc[common]
    assert (da["y"].to_numpy() == db["y"].to_numpy()).all(), \
        "matched dates must share the same realized outcome"
    y = da["y"].to_numpy()
    pa, pb = da["p"].to_numpy(), db["p"].to_numpy()

    if len(np.unique(y)) < 2:
        return {"model_a": model_a, "model_b": model_b, "target": target,
                "n": len(y), "auc_a": np.nan, "auc_b": np.nan,
                "difference": np.nan, "diff_lo": np.nan, "diff_hi": np.nan,
                "p_value": np.nan, "significant_5pct": False}

    auc_a = roc_auc_score(y, pa)
    auc_b = roc_auc_score(y, pb)
    obs_diff = auc_a - auc_b

    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_resample(n, block, rng)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            diffs[b] = np.nan
            continue
        diffs[b] = roc_auc_score(yb, pa[idx]) - roc_auc_score(yb, pb[idx])
    diffs = diffs[~np.isnan(diffs)]

    lo, hi = np.quantile(diffs, [0.025, 0.975])
    n_below = int(np.sum(diffs <= 0))
    n_above = int(np.sum(diffs >= 0))
    p_val = min(1.0, 2 * min((n_below + 1) / (len(diffs) + 1),
                             (n_above + 1) / (len(diffs) + 1)))
    return {"model_a": model_a, "model_b": model_b, "target": target,
            "n": n, "auc_a": auc_a, "auc_b": auc_b,
            "difference": obs_diff, "diff_lo": lo, "diff_hi": hi,
            "p_value": p_val, "significant_5pct": bool(lo > 0 or hi < 0)}


def top_vs_rest(oof: pd.DataFrame, target: str, n_boot: int = 2000) -> pd.DataFrame:
    """For one target, rank models by pooled AUC and test the top model
    against every other model, plus explicitly against GARCH-t as the
    natural 'does anything beat the standard statistical baseline'
    comparison regardless of rank."""
    sub = oof[oof.target == target]
    aucs = sub.groupby("model").apply(
        lambda d: roc_auc_score(d["y"], d["p"]) if d["y"].nunique() > 1 else np.nan)
    aucs = aucs.sort_values(ascending=False)
    top = aucs.index[0]
    rows = []
    for other in aucs.index:
        if other == top:
            continue
        rows.append(auc_difference_test(oof, target, top, other, n_boot=n_boot))
    if "garch_t" != top:
        rows.append(auc_difference_test(oof, target, top, "garch_t", n_boot=n_boot))
    df = pd.DataFrame(rows)
    df["pooled_auc_top"] = aucs.loc[top]
    return df
