import numpy as np
import torch
from scipy.stats import gaussian_kde


def calculate_rmse(model, x, y, num_samples=100):
    """
    Calculate the root mean squared error (RMSE) between the BNN predictions and the observed values,
    using repeated sampling to compute the expectation over the posterior variational distributions.

    Parameters:
    - model: Trained Bayesian Neural Network model.
    - x: Input data (tensor).
    - y: Observed data (tensor).
    - num_samples: Number of samples to use for approximating the expectation.

    Returns:
    - rmse: Root mean squared error (float).
    """
    predictive_samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Perform a forward pass with sampled weights
            predictions = model(x)  # Each forward pass samples weights from the variational posterior
            predictive_samples.append(predictions.squeeze())

        # Convert list of predictions to a tensor (num_samples, num_points)
        predictive_samples = torch.stack(predictive_samples)

        # Compute the mean predictions for each test point
        mean_predictions = predictive_samples.mean(dim=0)

    # Compute RMSE
    mse = torch.mean((mean_predictions - y.squeeze())**2)
    rmse = torch.sqrt(mse).item()

    return rmse

def assess_coverage_and_interval_score(
    model, x, y, alpha_levels=None, num_samples=1000, known_variance=1.0, hdr_resolution=1000
):
    """
    Assess coverage and calculate interval score for a Bayesian Neural Network using HDR methodology.

    Parameters:
    - model: Trained Bayesian Neural Network model.
    - x: Input data (tensor of shape (n_samples, n_features)).
    - y: Observed data (tensor of shape (n_samples,)).
    - alpha_levels: List of nominal levels for predictive intervals (e.g., [0.1, 0.05]).
                    Each level corresponds to a (1 - alpha) * 100% predictive interval.
    - num_samples: Number of Monte Carlo samples to generate.
    - known_variance: Known variance of the data-generating mechanism.
    - hdr_resolution: Number of grid points for HDR computation.

    Returns:
    - results: Dictionary containing:
        - "interval_scores": List of interval scores for each alpha level.
        - "coverage_rates": List of coverage rates for each alpha level.
        - "scores_per_point": Per-point interval scores for each alpha level.
        - "coverage_results": Per-point coverage results for each alpha level.
    """
    input_dim = x.shape[1]  # Dynamically determine input dimension
    if alpha_levels is None:
        alpha_levels = [0.05]  # Default to 95% HDR

    # Generate predictive samples
    predictive_samples_dict = generate_predictive_samples(
        model=model, x_locations=x, num_samples=num_samples, known_variance=known_variance
    )

    # Determine whether keys in predictive_samples_dict are floats or tuples
    use_float_keys = isinstance(next(iter(predictive_samples_dict.keys())), float)

    # Convert predictive samples to NumPy array for easier processing
    predictive_samples_np = np.array([
        predictive_samples_dict[float(x_val) if use_float_keys else tuple(x_val.tolist())]
        for x_val in x
    ])

    results = {
        "interval_scores": [],
        "coverage_rates": [],
        "scores_per_point": {},
        "coverage_results": {},
    }

    # Process each alpha level
    for alpha in alpha_levels:
        coverage_results = []
        scores_per_point = []
        for i, x_val in enumerate(x):
            # Extract predictive samples for the current test point
            key = float(x_val) if use_float_keys else tuple(x_val.tolist())
            samples = predictive_samples_dict[key]

            # Compute HDR intervals using compute_hdr
            level = 1 - alpha  # Convert alpha to HDR level
            hdr_intervals, _ = compute_hdr(samples, level=level, resolution=hdr_resolution)

            # Check if observed value falls within the HDR intervals
            observed_value = y[i].item()
            coverage = check_coverage(hdr_intervals, observed_value)
            coverage_results.append(coverage)

            # Calculate interval score for HDR
            interval_score = sum(u - l for l, u in hdr_intervals) + \
                             (2 / alpha) * sum(max(0, l - observed_value) + max(0, observed_value - u) for l, u in hdr_intervals)
            scores_per_point.append(interval_score)

        # Compute average interval score and coverage rate
        interval_score_avg = sum(scores_per_point) / len(scores_per_point)
        coverage_rate = sum(coverage_results) / len(coverage_results)

        # Store results
        results["interval_scores"].append(interval_score_avg)
        results["coverage_rates"].append(coverage_rate)
        results["scores_per_point"][f"{100 * (1 - alpha):.1f}%"] = scores_per_point
        results["coverage_results"][f"{100 * (1 - alpha):.1f}%"] = coverage_results

    return results

def generate_predictive_samples(model, x_locations, num_samples=1000, known_variance=1.0):
    """
    Generate predictive samples for a set of x-locations using batched evaluation.

    Parameters:
    - model: The trained BNN model.
    - x_locations: Locations at which to compute predictive samples (tensor).
    - num_samples: Number of Monte Carlo samples for each location.
    - known_variance: Known variance of the data-generating mechanism.

    Returns:
    - predictive_samples: A dictionary where keys are x-values (floats for 1D inputs, tuples for higher dimensions),
                          and values are arrays of predictive samples.
    """
    # Ensure x_locations has the correct shape (batch_size, in_features)
    if x_locations.ndimension() == 1:
        x_locations = x_locations.unsqueeze(1)  # Convert (N,) to (N, 1) if 1D

    # Determine whether to use float keys (1D) or tuple keys (higher dimensions)
    use_float_keys = x_locations.size(1) == 1

    # Initialize storage for predictive samples
    predictive_samples = {}
    for x in x_locations:
        key = float(x) if use_float_keys else tuple(x.tolist())
        predictive_samples[key] = []

    # Batch evaluation for all x_locations
    for _ in range(num_samples):
        # Perform a single forward pass for all x_locations
        pred_means = model(x_locations).detach().numpy().flatten()

        # Add Gaussian noise to each prediction based on the known variance
        pred_samples = np.random.normal(pred_means, np.sqrt(known_variance))

        # Store samples for each x_location
        for x_val, sample in zip(x_locations, pred_samples):
            key = float(x_val) if use_float_keys else tuple(x_val.tolist())
            predictive_samples[key].append(sample)

    # Convert lists to numpy arrays for each x_location
    for key in predictive_samples:
        predictive_samples[key] = np.array(predictive_samples[key])

    return predictive_samples

def compute_hdr(samples, level=0.95, resolution=1000):
    """
    Compute the highest density region (HDR) for a given set of samples.

    Parameters:
    - samples: A 1D array of predictive samples.
    - level: The desired HDR level (default is 0.95 for 95% HDR).
    - resolution: The number of grid points to evaluate the density.

    Returns:
    - hdr_intervals: A list of intervals [(lower1, upper1), (lower2, upper2), ...] representing the HDR.
    - density_function: The kernel density estimation (KDE) function for the samples.
    """
    kde = gaussian_kde(samples)
    x_grid = np.linspace(np.min(samples), np.max(samples), resolution)
    density_values = kde(x_grid)

    sorted_indices = np.argsort(density_values)[::-1]
    cumulative_density = np.cumsum(density_values[sorted_indices]) / np.sum(density_values)
    hdr_threshold_index = np.argmax(cumulative_density >= level)
    hdr_threshold = density_values[sorted_indices][hdr_threshold_index]

    hdr_intervals = []
    inside_region = False
    for i in range(len(x_grid) - 1):
        if density_values[i] >= hdr_threshold and not inside_region:
            lower = x_grid[i]
            inside_region = True
        elif density_values[i] < hdr_threshold and inside_region:
            upper = x_grid[i]
            hdr_intervals.append((lower, upper))
            inside_region = False

    if inside_region:
        hdr_intervals.append((lower, x_grid[-1]))

    return hdr_intervals, kde

def check_coverage(hdr_intervals, observed_value):
    """
    Check if an observed value falls within the HDR intervals.

    Parameters:
    - hdr_intervals: A list of intervals [(lower1, upper1), ...].
    - observed_value: The observed value to check.

    Returns:
    - bool: True if the observed value is within the HDR, False otherwise.
    """
    for lower, upper in hdr_intervals:
        if lower <= observed_value <= upper:
            return True
    return False