import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import sys

# === Command-line args ===
metric = sys.argv[1] if len(sys.argv) > 1 else "interval_score"
assert metric in ["interval_score", "rmse"]

methods = ["kl_div", "alpha_renyi"]
fig, axes = plt.subplots(3, 2, figsize=(8, 7), gridspec_kw={'height_ratios': [1, 1, 0.07]}, sharex='col')
axes_flat = axes.flatten()

plot_idx = 0
for method in methods:
    result_dir = f"../results/{method}_x1d"
    results_df = pd.read_csv(os.path.join(result_dir, "merged_results.csv"))

    # Get best/worst by selected metric
    best_idx = results_df[metric].idxmin()
    worst_idx = results_df[metric].idxmax()

    for i, idx in enumerate([best_idx, worst_idx]):
        fname = os.path.join(result_dir, "plotdata", f"plot_{idx:04d}.npz")
        if not os.path.exists(fname):
            print(f"Missing plot data: {fname}")
            continue

        data = np.load(fname)
        ax = axes_flat[plot_idx]

        ax.plot(data["x_plot"], data["true_func"], color="black", linestyle="--", label="True Function")
        ax.plot(data["x_plot"], data["mean_pred"], color="blue", label="Predicted Mean")
        ax.fill_between(data["x_plot"], data["lower_pred"], data["upper_pred"], alpha=0.2, color="blue", label="90% Credible Interval")

        ax.scatter(data["X_train"], data["y_train"], color="red", marker="o", s=10, alpha=0.4, edgecolor='k', linewidth=0.2, label="Train Data")
        ax.scatter(data["X_test"], data["y_test"], color="green", marker="^", s=10, alpha=0.4, edgecolor='k', linewidth=0.2, label="Test Data")

        fit_quality = "Best" if i == 0 else "Worst"
        method_label = "KL Divergence" if method == "kl_div" else "$\\alpha$-Renyi"
        title = f"{fit_quality} Fit\n{method_label}"
        ax.set_title(title, fontsize=10)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plot_idx += 1

# Hide last row axes
for idx in [4, 5]:
    axes_flat[idx].axis('off')

# Shared legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize='x-small')

plt.tight_layout(rect=[0, 0.06, 1, 1])
outfile = f"../figures/best_worst_x1d_from_saved_{metric}.pdf"
plt.savefig(outfile, dpi=300)
print(f"Saved plot to {outfile}")
