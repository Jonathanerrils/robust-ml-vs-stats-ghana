import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

oof = pd.read_csv("outputs/tables/nested_cv_oof_predictions.csv", parse_dates=["date"])

MODEL_TYPE = {
    "base_rate": "baseline", "logistic": "baseline", "mlp": "baseline", "svm": "baseline",
    "garch_t": "statistical", "var": "statistical", "markov_switching": "statistical",
    "lasso": "ml", "elastic_net": "ml", "random_forest": "ml", "xgboost": "ml",
}
COLOR = {"baseline": "#B0B0B0", "statistical": "#B0413E", "ml": "#2F6B4F"}
LABEL = {"baseline": "Weak baseline", "statistical": "Statistical (GARCH/VAR/Markov-sw.)",
        "ml": "Machine learning"}
NICE = {"cocoa_var_breach": "Cocoa VaR breach", "cedi_depreciation": "Cedi depreciation",
        "highvol_day": "High-volatility day"}
MODEL_NICE = {"base_rate": "Base rate", "garch_t": "GARCH-$t$", "var": "VAR",
             "markov_switching": "Markov-sw.", "logistic": "Logistic",
             "lasso": "LASSO", "elastic_net": "Elastic net",
             "random_forest": "Random forest", "xgboost": "XGBoost",
             "svm": "SVM", "mlp": "MLP"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, target in zip(axes, ["cocoa_var_breach", "cedi_depreciation", "highvol_day"]):
    sub = oof[oof.target == target]
    aucs = sub.groupby("model").apply(
        lambda d: roc_auc_score(d["y"], d["p"]) if d["y"].nunique() > 1 else np.nan)
    aucs = aucs.sort_values(ascending=True)
    colors = [COLOR[MODEL_TYPE[m]] for m in aucs.index]
    ax.barh([MODEL_NICE[m] for m in aucs.index], aucs.values, color=colors)
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xlim(0.35, 0.75)
    ax.set_title(NICE[target], fontsize=11)
    ax.set_xlabel("Pooled AUC")

handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in ["statistical", "ml", "baseline"]]
fig.legend(handles, [LABEL[k] for k in ["statistical", "ml", "baseline"]],
          loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Pooled AUC by model, all three targets (dashed line = random)", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("outputs/figures/pooled_auc_comparison.png", dpi=160, bbox_inches="tight")
print("saved pooled_auc_comparison.png")
