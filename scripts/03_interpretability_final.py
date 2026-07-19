import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from riskml import features, models, validation, interpret

TAB, FIG = "outputs/tables", "outputs/figures"

rets = pd.read_csv("data/processed/returns_real_extended.csv", index_col=0, parse_dates=True)
macro = pd.read_csv("data/processed/ghana_macro_real_extended.csv", index_col=0, parse_dates=True)
X = features.build_features(rets, macro)
y_all = features.make_labels(rets)
idx = X.index.intersection(y_all.dropna().index)
X = X.loc[idx]


def interpret_target(target: str, model_name: str, factory, params: dict,
                     do_shap: bool, weak_flag: bool = False):
    print(f"\n=== {target}: {model_name} (tuned params: {params}) "
          f"{'[WEAK MODEL — interpret with caution]' if weak_flag else ''} ===")
    y = y_all.loc[idx, target]
    outer_folds = validation.expanding_folds(idx, n_folds=5)
    stress = validation.stress_holdout(idx)

    # cross-fold stability of permutation importance (use the fixed tuned
    # params throughout, consistent with what the nested-CV run selected
    # on the fold used for this analysis, rather than re-tuning per fold
    # here -- this checks STABILITY of importance given a fixed model,
    # a different question from the nested-CV's own hyperparameter search)
    imp_by_fold = {}
    for fold in outer_folds:
        m = factory()
        m.set_params(**params)
        m.fit(X.loc[fold.train_idx], y.loc[fold.train_idx])
        imp_by_fold[fold.name] = interpret.perm_importance(
            m, X.loc[fold.test_idx], y.loc[fold.test_idx], n_repeats=10)
    stab = interpret.stability(imp_by_fold)
    stab.round(3).to_csv(f"{TAB}/feature_stability_{target}_{model_name}.csv")
    print(f"  Mean cross-fold stability (Spearman rho): "
          f"{stab.values[np.triu_indices_from(stab,1)].mean():.3f}")

    # final model: trained on the stress-holdout's training set (most
    # recent, most complete pre-2024 data), tested on 2024
    final = factory()
    final.set_params(**params)
    final.fit(X.loc[stress.train_idx], y.loc[stress.train_idx])
    perm = interpret.perm_importance(final, X.loc[stress.test_idx],
                                     y.loc[stress.test_idx], n_repeats=20)
    perm.round(5).to_csv(f"{TAB}/perm_importance_{target}_{model_name}.csv")
    interpret.plot_top_importance(
        perm, f"Permutation importance, {model_name} — {target}"
              f"{' (weak model)' if weak_flag else ''}",
        f"{FIG}/perm_importance_{target}_{model_name}.png")
    print("  Top 5 permutation importance:")
    print(perm.head(5).to_string())

    if do_shap:
        try:
            sh = interpret.shap_importance(final, X.loc[stress.train_idx])
            sh.round(5).to_csv(f"{TAB}/shap_importance_{target}_{model_name}.csv")
            interpret.plot_top_importance(
                sh, f"Mean |SHAP|, {model_name} — {target}",
                f"{FIG}/shap_importance_{target}_{model_name}.png")
            print("  Top 5 SHAP importance:")
            print(sh.head(5).to_string())
            top3 = list(sh.index[:3])
        except Exception as e:
            print(f"  SHAP skipped: {e}")
            top3 = list(perm.index[:3])
        interpret.plot_pdp(final, X.loc[stress.train_idx], top3,
                           f"{FIG}/pdp_{target}_{model_name}.png")
    return stab, perm


# --- cedi_depreciation: random_forest, the cleanest case (top overall AND top feature-using)
interpret_target("cedi_depreciation", "random_forest",
                 models.make_random_forest, {"min_samples_leaf": 50}, do_shap=True)

# --- cocoa_var_breach: elastic_net, top feature-using model (GARCH-t itself has no features)
interpret_target("cocoa_var_breach", "elastic_net",
                 models.make_elastic_net, {"clf__C": 1.0}, do_shap=False)
# linear model: report actual fitted coefficients directly, more precise than SHAP for a linear model
en = models.make_elastic_net()
en.set_params(**{"clf__C": 1.0})
stress = validation.stress_holdout(idx)
y = y_all.loc[idx, "cocoa_var_breach"]
en.fit(X.loc[stress.train_idx], y.loc[stress.train_idx])
coefs = pd.Series(en.named_steps["clf"].coef_.ravel(), index=X.columns).sort_values(
    key=np.abs, ascending=False)
coefs.head(15).to_csv(f"{TAB}/elastic_net_coefficients_cocoa_var_breach.csv")
print("\nTop 10 elastic-net coefficients (cocoa_var_breach, standardized scale):")
print(coefs.head(10).round(4).to_string())

# --- highvol_day: random_forest, top feature-using model but weak overall (flag explicitly)
interpret_target("highvol_day", "random_forest",
                 models.make_random_forest, {"min_samples_leaf": 10},
                 do_shap=True, weak_flag=True)

# --- VAR: report the fitted coefficient structure directly (native interpretability)
print("\n=== VAR: fitted lag-coefficient structure (cocoa equation), stress-holdout training window ===")
from statsmodels.tsa.api import VAR
core = ["cocoa", "gold", "brent", "wti", "cedi"]
r = rets.loc[stress.train_idx[0]:stress.train_idx[-1], core]
res = VAR(r).fit(2)  # matches the p typically selected in the nested-CV runs
cocoa_eq = res.params["cocoa"]
print(cocoa_eq.round(4).to_string())
cocoa_eq.to_csv(f"{TAB}/var_cocoa_equation_coefficients.csv")

# --- Markov-switching: current regime diagnostic
print("\n=== Markov-switching: regime structure per target series ===")
from hmmlearn.hmm import GaussianHMM
for tcol in ["cocoa", "cedi"]:
    rr = rets.loc[stress.train_idx, tcol].to_numpy().reshape(-1, 1)
    hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=300, random_state=0).fit(rr)
    means, stds = hmm.means_.ravel(), np.sqrt(hmm.covars_.ravel())
    quiet = int(np.argmin(stds))
    print(f"  {tcol}: quiet regime mean={means[quiet]:.4f} std={stds[quiet]:.4f} | "
          f"volatile regime mean={means[1-quiet]:.4f} std={stds[1-quiet]:.4f} | "
          f"P(stay quiet)={hmm.transmat_[quiet,quiet]:.3f} P(stay volatile)={hmm.transmat_[1-quiet,1-quiet]:.3f}")

print("\nDone.")
