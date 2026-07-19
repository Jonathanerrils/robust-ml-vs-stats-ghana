import sys, time
sys.path.insert(0, "src")
import pandas as pd
from riskml import features, validation, nested_cv

rets = pd.read_csv("data/processed/returns_real_extended.csv", index_col=0, parse_dates=True)
macro = pd.read_csv("data/processed/ghana_macro_real_extended.csv", index_col=0, parse_dates=True)
X = features.build_features(rets, macro)
y_all = features.make_labels(rets)
idx = X.index.intersection(y_all.dropna().index)
X = X.loc[idx]

all_results = []
all_oof = []
t0 = time.time()
for target in ["cocoa_var_breach", "cedi_depreciation", "highvol_day"]:
    y = y_all.loc[idx, target]
    outer_folds = validation.expanding_folds(idx, n_folds=5)
    outer_folds.append(validation.stress_holdout(idx))
    print(f"\n=== {target} | event rate {y.mean():.3f} ===", flush=True)
    res, oof = nested_cv.run_nested_cv(X, y, rets, target, outer_folds)
    all_results.append(res)
    all_oof.append(oof)
    print(f"  cumulative time: {time.time()-t0:.1f}s")

full = pd.concat(all_results, ignore_index=True)
full.to_csv("outputs/tables/nested_cv_full_results.csv", index=False)
oof_full = pd.concat(all_oof, ignore_index=True)
oof_full.to_csv("outputs/tables/nested_cv_oof_predictions.csv", index=False)
print(f"\nTotal time: {time.time()-t0:.1f}s")

# summary: mean AUC by model x target, CV folds vs stress holdout separately
full["is_stress"] = full["fold"] == "stress_2024"
summ = full.groupby(["target", "model", "is_stress"])["auc"].mean().unstack("is_stress")
summ.columns = ["auc_cv_mean", "auc_stress"]
summ = summ.sort_values(["target", "auc_cv_mean"], ascending=[True, False])
summ.round(4).to_csv("outputs/tables/nested_cv_summary.csv")
print(summ.round(3).to_string())
