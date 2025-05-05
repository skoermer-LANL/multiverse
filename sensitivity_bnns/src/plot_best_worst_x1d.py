# src/plot_best_worst_x1d.py

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from evaluate import generate_predictive_samples, compute_hdr

# === Parse arguments ===
metric = "interval_score"
use_cached = False
if len(sys.argv) > 1:
    metric = sys.argv[1]
if len(sys.argv) > 2:
    use_cached = sys.argv[2].lower() == "true"

assert metric in ["interval_score", "rmse"], "Metric must be 'interval_score' or 'rmse'"

# === Setup ===
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 10})

methods = ["kl_div", "alpha_renyi"]
dgms = ["x1d", "x1d"]

# Load data safely
X_train = torch.load("../data/x1d/X_train.pt", weights_only=True)
y_train = torch.load("../data/x1d/y_train.pt", weights_only=True)
X_test = torch.load("../data/x1d/X_test.pt", weights_only=True)
y_test = torch.load("../data/x1d/y_test.pt", weights_only=True)
noise_var = torch.load("../data/x1d/noise_var.pt", weights_only=True).item()

# True function
def true_function(x):
    return x**3 - x**2 + 3

# Rescale functions
def rescale_x(x):
    return (x + 2) / 4

def rescale_y(y, mean_center, scaling_factor):
    return (y - mean_center) / scaling_factor

# Prediction function with HDR
def predict_with_hdr(model, x_grid, known_variance, level=0.90, num_samples=1000):
    predictive_samples_dict = generate_predictive_samples(
        model=model,
        x_locations=x_grid,
        num_samples=num_samples,
        known_variance=known_variance
    )

    mean_preds = []
    lower_bounds = []
    upper_bounds = []

    for x_val in x_grid:
        samples = predictive_samples_dict[float(x_val.item())]
        mean_preds.append(np.mean(samples))

        hdr_intervals, _ = compute_hdr(samples, level=level)
        if hdr_intervals:
            lower_bounds.append(hdr_intervals[0][0])
            upper_bounds.append(hdr_intervals[0][1])
        else:
            lower_bounds.append(np.percentile(samples, (1 - level) / 2 * 100))
            upper_bounds.append(np.percentile(samples, (1 + level) / 2 * 100))

    return np.array(mean_preds), np.array(lower_bounds), np.array(upper_bounds)

# Prepare true function rescaling
Ey = true_function((X_train.squeeze() * 4) - 2)
Ey_mean = Ey.mean().item()
Ey_var = Ey.var(unbiased=False).item()
true_scaling = np.sqrt(Ey_var + noise_var)

# === Prepare Plotting Grid ===
x_plot = np.linspace(-2, 2, 400)
x_plot_tensor = torch.tensor(rescale_x(x_plot)).float().unsqueeze(1)
true_function_rescaled = rescale_y(true_function(x_plot), Ey_mean, true_scaling)

# === Plotting Setup ===
fig, axes = plt.subplots(3, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1, 0.01]}, sharex='col')
axes_flat = axes.flatten()

plot_idx = 0
for j, method in enumerate(methods):
    dgm = dgms[j]

    for i, fit_quality in enumerate(["best", "worst"]):
        cache_filename = f"../figures/cache_{method}_{dgm}_{fit_quality}_{metric}.npz"

        if use_cached and os.path.exists(cache_filename):
            print(f"📦 Loading cached results: {cache_filename}")
            data = np.load(cache_filename)
            mean_pred = data["mean_pred"]
            lower_pred = data["lower_pred"]
            upper_pred = data["upper_pred"]
        else:
            # Load merged results and LHS
            results_path = f"../results/{method}_{dgm}/merged_results.csv"
            lhs_path = f"../lhs/{method}_{dgm}_lhs.csv"

            results_df = pd.read_csv(results_path)
            lhs_df = pd.read_csv(lhs_path)

            # Select best or worst
            idx = results_df[metric].idxmin() if fit_quality == "best" else results_df[metric].idxmax()

            params = lhs_df.iloc[idx]
            reg_module = importlib.import_module(method)
            model, _ = reg_module.run_regression(params, X_train, y_train, noise_var)

            mean_pred, lower_pred, upper_pred = predict_with_hdr(model, x_plot_tensor, known_variance=noise_var, level=0.90)

            # Save cached data
            np.savez(cache_filename, mean_pred=mean_pred, lower_pred=lower_pred, upper_pred=upper_pred)

        # Plotting
        ax = axes_flat[plot_idx]
        ax.plot(x_plot, true_function_rescaled, label="True Function", color="black", linestyle="--")
        ax.plot(x_plot, mean_pred, label="Predicted Mean", color="blue")
        ax.fill_between(x_plot, lower_pred, upper_pred, color="blue", alpha=0.2, label="90% Credible Interval")

        # Updated point styling
        ax.scatter((X_train.squeeze().cpu().numpy() * 4) - 2, y_train.squeeze().cpu().numpy(), 
                   label="Training Data", color="red", marker="o", s=10, alpha=0.4, edgecolor='k', linewidth=0.2)
        ax.scatter((X_test.squeeze().cpu().numpy() * 4) - 2, y_test.squeeze().cpu().numpy(), 
                   label="Test Data", color="green", marker="^", s=10, alpha=0.4, edgecolor='k', linewidth=0.2)

        method_title = r"KL Divergence" if method == "kl_div" else r"$\alpha$-Renyi"
        title_text = f"{'Best' if fit_quality == 'best' else 'Worst'} Fit\n{method_title}"
        ax.set_title(title_text, fontsize=10)

        ax.tick_params(axis='x', labelbottom=True)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        plot_idx += 1

# Hide last two empty subplots
for idx in [4, 5]:
    axes_flat[idx].axis('off')

# Create shared legend at bottom
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize='medium')

plt.tight_layout(rect=[0, 0.008, 1, 1])
plt.savefig(f"../figures/best_worst_x1d_{metric}.pdf", dpi=300)
print(f"Plot saved as: figures/best_worst_x1d_{metric}.pdf")
