# Decision log

This is the running record of what actually happened while building this
project — every bug we hit, every judgment call, every dead end. Kept
deliberately unpolished because the point is traceability, not prose.

## Naming
The project started life under a different name — "Interpretable Machine
Learning..." — before we settled on the title actually specified in the
original scope, "Robust Machine Learning vs Statistical Models...".
Two old directories from that earlier phase (`interpretable-ml-ghana-risk`,
`ghana-risk-ml`) never made it into this repo. If you find references to
either name lying around somewhere, they're stale.

## Data pipeline
- Real daily panel reuses the cleaned, bug-fixed canonical commodity/cedi
  panel built for the companion tail-dependence project (cocoa, gold,
  Brent, WTI, cedi), rather than rebuilding BoG integration from scratch.
- Added three new daily series: CBOE VIX and the Federal Reserve's
  Nominal Broad Dollar Index (DTWEXBGS), both from FRED. First downloads
  of both only covered a 5-year default window (July 2021 onward);
  caught before use (would have cut out the entire COVID stress episode)
  and re-fetched with full history (VIX since 1990, DTWEXBGS since 2006).
- Added Ghana Monetary Policy Rate (monthly, BoG). Confirmed via search
  it is not present in any previously-uploaded BoG table. The correct
  BoG label is "Monetary Policy Rate (%)," matching the naming
  convention of other BoG series already in use.
- Policy rate required the same reporting-gap fix already used for
  exports in the companion project: the source table encodes an
  8-month reporting gap (2023-05 to 2023-12) as literal 0.00 rather
  than missing; isolated to exactly those 8 consecutive months and
  masked before use.
- **Bug found and fixed**: the macro-feature merge (`features.py`)
  originally truncated the entire feature matrix to end mid-2023,
  silently eliminating the whole 2024 stress-test window the paper's
  design depends on. Cause: the macro source file has explicit rows
  with missing values extending to the present for series that stopped
  updating, rather than simply ending; `reindex(method="ffill")` found
  the nearest prior *row* (already missing) rather than the nearest
  prior *valid value*. Fixed by forward-filling each macro column's own
  gaps before reindexing onto the daily calendar. Verified: full 2024
  coverage (262 trading days) restored.
- `make_labels()`'s cedi-depreciation label originally hardcoded the
  column name `ghs_usd` and assumed raw (unflipped) orientation. The
  companion project's canonical panel uses a column named `cedi`,
  already sign-flipped so the lower tail denotes depreciation. Fixed to
  detect the column name and internally negate to recover the raw
  orientation the label's threshold logic needs.

## Model roster (confirmed, 11 models)
- Statistical: base rate, GARCH-t, VAR (5 core series, lag order
  AIC-selected, capped at 5), 2-state Markov-switching (states fixed at
  2, not tuned, matching the companion project's established regime
  structure for this currency).
- Machine learning: logistic, LASSO, elastic net, random forest,
  XGBoost, SVM, small MLP (deliberately weak neural benchmark).
- A GRU/LSTM sequence model on raw returns was considered and explicitly
  not built: sample-size analysis showed this dataset (under 3,000 daily
  observations, ~150 positive examples for the rarest target) is roughly
  an order of magnitude below what published sequence-model successes in
  finance require, which typically rely on cross-sectional pooling
  across hundreds of related series unavailable for a single currency.

## Bugs found and fixed during model development
- **XGBoost**: `min_child_weight=40` (carried over from a *different*
  parameter, sklearn's `min_samples_leaf`, with different semantics —
  XGBoost's parameter is a minimum-Hessian-weight threshold, not a raw
  sample count) was far too restrictive against the ~55 positive
  training examples in the smallest fold, collapsing every tree to a
  single leaf and every prediction to the training base rate. Caught
  because predictions were suspiciously degenerate (identical value
  across a heterogeneous test set), not because the AUC alone looked
  wrong. Fixed: reduced to `min_child_weight=3`.

## Nested cross-validation design
- 5 expanding-window outer folds + 2024 stress holdout (the real cocoa
  shock, not synthetic).
- Inner tuning: 3 expanding-window folds within each outer training set,
  modest grids (2-4 values per hyperparameter).
- Minimum-viable-positives fallback: verified the smallest outer fold's
  inner split would produce as few as 2 positive test examples for one
  target — too few for reliable tuning. Any outer fold whose inner split
  falls below 10 positive examples in any inner test fold skips nested
  tuning and uses a documented default instead. Verified: fallback
  triggered for 22% of tunable-model results, 100% concentrated in the
  earliest 1-2 outer folds, 0% in fold3 onward or the 2024 stress
  holdout — confirming it protects exactly the folds it was designed to
  protect without touching the folds the paper's conclusions rest on.

## Significance testing
- Paired moving-block bootstrap (block=20 days, 2000 replicates) on
  pooled out-of-fold predictions, since both models in any comparison
  are scored on identical dates and outcomes.
- Pooled AUC (all out-of-fold predictions concatenated) used as the
  primary ranking, not the fold-averaged AUC: pooling differs from
  fold-averaging for 2 of 3 targets (fold-averaging over-weights small,
  high-variance early folds). Both are reported; pooled is authoritative.
- Full family-wise Bonferroni (m=31) and independent Benjamini-Hochberg
  FDR correction applied; both reported, since a result surviving only
  one is weaker evidence than one neither touches.

## Interpretability
- An initial interpretability pass (permutation importance, SHAP) was
  run on the pre-nested-CV, un-tuned 6-model pipeline. This was
  identified as stale once the full 11-model nested-CV harness was
  built and was **discarded, not reported**: its headline finding
  (gold volatility as the top predictor of cedi depreciation) does not
  survive proper tuning and must not be cited.
- Final interpretability reruns the analysis on the actual tuned models
  from the nested-CV run (random forest for cedi depreciation and
  high-vol day, elastic net for cocoa VaR breach — the top *feature-
  using* model per target, since GARCH/VAR/Markov-switching consume no
  feature matrix and are interpreted on their own native terms instead:
  VAR's fitted lag coefficients, Markov-switching's regime parameters).
- Cross-fold stability of permutation importance is reported as a
  first-class diagnostic, not just point-estimate importance: cocoa VaR
  breach shows stability indistinguishable from zero (rho=-0.009),
  corroborating the significance-testing null from an independent angle.

## Literature grounding
- A literature search was conducted before writing the paper (not
  after), covering: ML-vs-GARCH volatility forecasting (mixed, and
  rarely significance-tested); currency-crisis EWS literature
  specifically (closer methodological match, more cautious about ML's
  advantage); feature-importance instability as a documented property of
  small-sample financial ML (validates, does not undermine, this
  paper's own near-zero-stability finding); Ghana-cedi-specific ML work
  (exists, but uses continuous-level forecasting, not binary tail-event
  classification with significance testing -- identified as the gap this
  paper fills); and one closely parallel, near-identically-designed
  contemporaneous study of the CAD/USD exchange rate reaching an almost
  identical conclusion (ensemble ML fails to significantly beat a
  statistical benchmark under formal testing), used as the primary
  external validation of this paper's central finding.
- Several literature citations (see `paper/references.bib`) have
  honestly-disclosed missing author fields: the web-search snippets used
  to locate them gave titles and venues but not always clean author
  bylines. Names were not fabricated to make the bibliography look more
  complete. These entries should be verified against primary sources
  before formal submission.

## Figures
- Four figures built specifically for the paper (none existed before
  this pass): pooled AUC comparison (all models, all targets, colored by
  model type); a significance forest plot visualizing the AUC-difference
  tests with confidence intervals and correction status (the paper's
  central finding previously existed only as a table); cross-fold
  stability heatmaps contrasting cocoa (near-zero, patternless) against
  cedi (modest, consistently positive); and the corrected SHAP
  importance plot for cedi depreciation.
- A matplotlib bug was caught and fixed while building the significance
  forest plot: passing a list of hex colors to `ecolor` in a single
  vectorized `errorbar()` call was ambiguously interpreted as a single
  RGBA tuple when the list had length 4, crashing on some target groups.
  Fixed by looping and drawing each error bar individually with an
  explicit scalar color.
