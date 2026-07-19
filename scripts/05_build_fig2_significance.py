import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sig = pd.read_csv("outputs/tables/significance_tests_corrected.csv")
NICE = {"cocoa_var_breach": "Cocoa VaR breach", "cedi_depreciation": "Cedi depreciation",
        "highvol_day": "High-volatility day"}
MODEL_NICE = {"base_rate": "base rate", "garch_t": "GARCH-$t$", "var": "VAR",
             "markov_switching": "Markov-sw.", "logistic": "logistic",
             "lasso": "LASSO", "elastic_net": "elastic net",
             "random_forest": "random forest", "xgboost": "XGBoost",
             "svm": "SVM", "mlp": "MLP"}

fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
for ax, target in zip(axes, ["cocoa_var_breach", "cedi_depreciation", "highvol_day"]):
    sub = sig[sig.target == target].copy()
    top_model = sub["model_a"].iloc[0]
    sub = sub.sort_values("difference")
    labels = [f"{MODEL_NICE[top_model]} vs. {MODEL_NICE[m]}" for m in sub["model_b"]]
    y = np.arange(len(sub))

    def color(row):
        if row["sig_bonferroni"]:
            return "#8B0000"
        if row["sig_FDR_BH"]:
            return "#CC7A00"
        if row["significant_5pct"]:
            return "#D4AC0D"
        return "#999999"

    for yi, (_, row) in zip(y, sub.iterrows()):
        c = color(row)
        ax.errorbar([row["difference"]], [yi],
                   xerr=[[row["difference"] - row["diff_lo"]],
                         [row["diff_hi"] - row["difference"]]],
                   fmt="o", color=c, ecolor=c, elinewidth=2.5, capsize=3, ms=6, zorder=5)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{NICE[target]} (leader: {MODEL_NICE[top_model]})", fontsize=10)

axes[-1].set_xlabel("AUC difference (leader $-$ alternative), with 95% block-bootstrap CI")
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8)
          for c in ["#8B0000", "#CC7A00", "#D4AC0D", "#999999"]]
fig.legend(handles, ["Bonferroni-significant", "FDR-significant (not Bonferroni)",
                     "Nominal only (5%, uncorrected)", "Not significant"],
          loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9)
fig.suptitle("Significance of the leading model's AUC advantage, by comparison", fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("outputs/figures/significance_forest.png", dpi=160, bbox_inches="tight")
print("saved significance_forest.png")
