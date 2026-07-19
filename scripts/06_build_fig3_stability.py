import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cocoa = pd.read_csv("outputs/tables/feature_stability_cocoa_var_breach_elastic_net.csv", index_col=0)
cedi = pd.read_csv("outputs/tables/feature_stability_cedi_depreciation_random_forest.csv", index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, df, title, mean_rho in zip(
        axes, [cocoa, cedi],
        [r"Cocoa VaR breach (elastic net): $\bar\rho = -0.009$",
         r"Cedi depreciation (random forest): $\bar\rho = 0.142$"],
        [-0.009, 0.142]):
    im = ax.imshow(df.values, vmin=-0.3, vmax=0.4, cmap="RdBu_r")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"F{i+1}" for i in range(5)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"F{i+1}" for i in range(5)])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{df.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(df.values[i,j]) > 0.2 else "black")
    ax.set_title(title, fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.75, label="Spearman rank correlation between folds")
fig.suptitle("Cross-fold stability of permutation importance rankings", fontsize=12)
fig.savefig("outputs/figures/feature_stability_heatmaps.png", dpi=160, bbox_inches="tight")
print("saved feature_stability_heatmaps.png")
