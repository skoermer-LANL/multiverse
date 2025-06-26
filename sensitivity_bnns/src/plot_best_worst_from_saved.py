
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import sys
from config import get_noise_variance

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# === Parse command-line arguments ===
plot_mode = sys.argv[1] if len(sys.argv) > 1 else "interval_score"
method_arg = sys.argv[2] if len(sys.argv) > 2 else "both"

assert plot_mode in ["interval_score", "rmse", "both_metrics"]
assert method_arg in ["kl_div", "alpha_renyi", "both"]

# === Setup
fig, axes = plt.subplots(3, 2, figsize=(8, 6),
                         gridspec_kw={'height_ratios': [1, 1, 0.05]},
                         sharex='col')
axes_flat = axes.flatten()
plot_idx = 0
metrics_to_plot = []

# Prepare table to collect data
table_rows = []

if plot_mode == "both_metrics":
    assert method_arg in ["kl_div", "alpha_renyi"], "Must choose a single method for both_metrics mode"
    methods_to_plot = [method_arg]
    metrics_to_plot = ["interval_score", "rmse"]
else:
    methods_to_plot = ["kl_div", "alpha_renyi"] if method_arg == "both" else [method_arg]
    metrics_to_plot = [plot_mode]

for method in methods_to_plot:
    dgm = "x1d"
    result_dir = f"results/{method}_{dgm}"
    results_df = pd.read_csv(os.path.join(result_dir, "merged_results.csv"))
    lhs_df = pd.read_csv(f"lhs/{method}_{dgm}_lhs.csv")
    noise_var = get_noise_variance(dgm)

    for metric in metrics_to_plot:
        best_pos = results_df[metric].idxmin()
        best_idx = results_df.loc[best_pos, "lhs_row"]
        worst_pos = results_df[metric].idxmax()
        worst_idx = results_df.loc[worst_pos, "lhs_row"]

        for i, idx in enumerate([best_idx, worst_idx]):
            fname = os.path.join(result_dir, "plotdata", f"plot_{int(idx):04d}.npz")
            if not os.path.exists(fname):
                print(f"⚠️ Missing plot data: {fname}")
                continue

            row = results_df[results_df["lhs_row"] == idx].iloc[0]
            rmse = row["rmse"]
            iscore = row["interval_score"]
            params = lhs_df.iloc[int(idx)].to_dict()

            # Add to table
            table_row = {"method": method, "metric": metric,
                         "fit": "best" if i == 0 else "worst",
                         "lhs_row": idx, "rmse": rmse, "interval_score": iscore}
            table_row.update(params)
            table_rows.append(table_row)

            # === Plot
            data = np.load(fname)
            ax = axes_flat[plot_idx]

            # Recompute scaling
            x_train_natural = data["X_train"]
            Ey = x_train_natural**3 - x_train_natural**2 + 3
            Ey_mean = Ey.mean()
            Ey_var = Ey.var(ddof=0)
            true_var = noise_var * Ey_mean / (1 - noise_var)
            y_scaling = np.sqrt(true_var + Ey_var)
            unscale = lambda y: y * y_scaling + Ey_mean

            ax.plot(data["x_plot"], unscale(data["true_func"]), color="black", linestyle="--", label="True Function")
            ax.plot(data["x_plot"], unscale(data["mean_pred"]), color="blue", label="Predicted Mean")
            ax.fill_between(data["x_plot"],
                            unscale(data["lower_pred"]),
                            unscale(data["upper_pred"]),
                            alpha=0.2, color="blue", label="90% Credible Interval")

            ax.scatter(data["X_train"], unscale(data["y_train"]),
                       color="red", marker="o", s=10, alpha=0.4, edgecolor='k', linewidth=0.2, label="Training Data")
            ax.scatter(data["X_test"], unscale(data["y_test"]),
                       color="green", marker="^", s=10, alpha=0.4, edgecolor='k', linewidth=0.2, label="Testing Data")

            fit_quality = "Best" if i == 0 else "Worst"
            metric_label = "Interval Score" if metric == "interval_score" else "RMSE"
            title = f"{fit_quality} {metric_label}"
            ax.set_title(title, fontsize=10)

            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.set_xticks([-2, 0, 2])
            ax.set_xticklabels(["-2", "0", "2"])

            # Add metric text to plot
            ax.text(0.05, 0.95,
                    f"RMSE: {rmse:.3f}\nIS: {iscore:.3f}",
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

            plot_idx += 1

# Hide unused plots (last 2 in bottom row)
for idx in range(plot_idx, 6):
    axes_flat[idx].axis('off')

# Shared legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize='medium')

plt.tight_layout(rect=[0, 0.04, 1, 1])

# === Save figure
if plot_mode == "both_metrics":
    outfile = f"figures/best_worst_x1d_from_saved_{method_arg}_interval_and_rmse.pdf"
else:
    method_suffix = method_arg if method_arg != "both" else "all"
    outfile = f"figures/best_worst_x1d_from_saved_{plot_mode}_{method_suffix}.pdf"

plt.savefig(outfile, dpi=300)
print(f"Saved plot to {outfile}")

# === Save data table
table_df = pd.DataFrame(table_rows)
csv_outfile = outfile.replace(".pdf", ".csv").replace("figures", "results")
os.makedirs(os.path.dirname(csv_outfile), exist_ok=True)
table_df.to_csv(csv_outfile, index=False)
print(f"Saved summary CSV to {csv_outfile}")
