# src/main.py

import sys
import os
import pandas as pd
import numpy as np
from config import get_paths, get_noise_variance
from utils import load_lhs_parameters, load_data_pt
from evaluate import calculate_rmse, assess_coverage_and_interval_score, generate_predictive_samples, compute_hdr
import importlib
import torch

def main(method, dgm, index, custom_lhs=None):
    # Get paths to data and results
    lhs_file, train_dir, test_dir, result_dir = get_paths(method, dgm)
    os.makedirs(result_dir, exist_ok=True)

    # Override LHS file path if provided
    if custom_lhs is not None:
        lhs_file = custom_lhs

    # Load parameters from LHS row
    params, idx = load_lhs_parameters(lhs_file, index)

    # Load training and testing data
    noise_var = get_noise_variance(dgm)
    X_train, y_train, X_test, y_test = load_data_pt(train_dir)

    # Dynamically load the regression module
    reg_module = importlib.import_module(method)
    model, loss = reg_module.run_regression(params, X_train, y_train, noise_var)

    # Run evaluation
    rmse = calculate_rmse(model, X_test, y_test, num_samples=1000)

    results = assess_coverage_and_interval_score(
        model=model,
        x=X_test,
        y=y_test,
        alpha_levels=[0.1], 
        num_samples=1000,
        known_variance=noise_var
    )

    # Extract clean scalar values
    coverage = float(results["coverage_rates"][0]) if isinstance(results["coverage_rates"], (list, np.ndarray)) else float(results["coverage_rates"])
    interval_score = float(results["interval_scores"][0]) if isinstance(results["interval_scores"], (list, np.ndarray)) else float(results["interval_scores"])

    # Create result dataframe
    result_df = pd.DataFrame([[idx, float(loss), float(rmse), coverage, interval_score]],
        columns=["lhs_row", "loss", "rmse", "coverage", "interval_score"])

    # Save result
    suffix = "_optimal" if custom_lhs else ""
    filename = f"result_{index}{suffix}.csv"
    result_df.to_csv(os.path.join(result_dir, filename), index=False)

    # === Save plotting data if dgm == 'x1d'
    if dgm == "x1d":
        # Generate predictive samples on fine grid
        x_plot = np.linspace(-2, 2, 400)
        x_plot_tensor = torch.tensor((x_plot + 2) / 4).float().unsqueeze(1)

        predictive_samples = generate_predictive_samples(
            model=model,
            x_locations=x_plot_tensor,
            num_samples=1000,
            known_variance=noise_var
        )

        mean_pred, lower_pred, upper_pred = [], [], []
        for x_val in x_plot_tensor:
            samples = predictive_samples[float(x_val.item())]
            mean_pred.append(np.mean(samples))
            hdr_intervals, _ = compute_hdr(samples, level=0.90)
            if hdr_intervals:
                lower_pred.append(hdr_intervals[0][0])
                upper_pred.append(hdr_intervals[0][1])
            else:
                lower_pred.append(np.percentile(samples, 5))
                upper_pred.append(np.percentile(samples, 95))

        # Compute rescaled true function for overlay
        Ey = (X_train.squeeze().numpy() * 4 - 2)**3 - (X_train.squeeze().numpy() * 4 - 2)**2 + 3
        Ey_mean = Ey.mean()
        Ey_var = Ey.var(ddof=0)
        y_scaling = np.sqrt(Ey_var + noise_var)
        true_func_rescaled = (x_plot**3 - x_plot**2 + 3 - Ey_mean) / y_scaling

        # Prepare save path
        plotdata_dir = os.path.join(result_dir, "plotdata")
        os.makedirs(plotdata_dir, exist_ok=True)
        save_path = os.path.join(plotdata_dir, f"plot_{idx:04d}.npz")

        np.savez(save_path,
                x_plot=x_plot,
                mean_pred=np.array(mean_pred),
                lower_pred=np.array(lower_pred),
                upper_pred=np.array(upper_pred),
                X_train=(X_train.squeeze().numpy() * 4 - 2),
                y_train=y_train.squeeze().numpy(),
                X_test=(X_test.squeeze().numpy() * 4 - 2),
                y_test=y_test.squeeze().numpy(),
                true_func=true_func_rescaled)
        print(f"Saved plot data to {save_path}")





if __name__ == "__main__":
    method = sys.argv[1]
    dgm = sys.argv[2]
    index = int(sys.argv[3])
    custom_lhs = sys.argv[4] if len(sys.argv) > 4 else None
    main(method, dgm, index, custom_lhs)

