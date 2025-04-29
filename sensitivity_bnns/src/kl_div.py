# src/kl_div.py

from torchbnn import BayesLinear
import torch.nn as nn
import torch
import torch.optim as optim

import torchbnn as bnn
import numpy as np

# src/kl_div.py

def run_regression(params, X, y, noise_var):

    log_kl_multiplier = params["log_kl_multiplier"]
    log_sigma = params["log_sigma"]
    num_steps = int(params["num_steps"])
    num_weights = int(params["num_weights"])
    initial_samples = int(params["initial_samples"])
    log_lr = params["log_lr"]
    prior_mu = params["prior_mu"]

    kl_multiplier = 10**log_kl_multiplier
    prior_sigma = 10**log_sigma
    lr = 10**log_lr

    

    model, nll_history, kl_history, _ = fit_variational_bnn_with_kl(
        X, y,
        prior_mu = prior_mu,
        prior_sigma = prior_sigma,
        known_var = noise_var,
        num_steps = num_steps,
        lr = lr,
        initial_samples= initial_samples,
        num_weights = num_weights,
        kl_multiplier = kl_multiplier
    )

    loss = nll_history[-1] + kl_history[-1]


    return model, loss


def fit_variational_bnn_with_kl(
    x, y,
    prior_mu=0,
    prior_sigma=1,
    known_var=1.0,
    num_steps=20000,
    lr=0.01,
    initial_samples=1,
    num_weights=10,  # Unified weight configuration argument
    kl_multiplier=1.0,
    print_progress=False
):
    """
    Fit a Bayesian Neural Network (BNN) using variational inference (torchbnn).
    Includes optional dynamic sample size adjustment to reduce Monte Carlo variance.
    Uses the KL divergence instead of the alpha-Rényi divergence.

    Parameters:
    - x: Training inputs (tensor of shape (n_samples, n_features)).
    - y: Training targets (tensor).
    - prior_mu: Mean of the prior distributions for the weights and biases.
    - prior_sigma: Standard deviation of the prior distributions for the weights and biases.
    - known_var: Known variance of the data noise.
    - num_steps: Number of optimization steps.
    - lr: Learning rate for the optimizer.
    - initial_samples: Initial number of Monte Carlo samples.
    - adjust_interval: Interval (in steps) for adjusting the number of samples.
    - window_size: Number of steps in each window for variance estimation.
    - p_value_threshold: Threshold for the chi-squared test in sample adjustment.
    - use_autoregressive_adjustment: Whether to use autoregressive adjustment for sample size.
    - num_weights: Number of weights in each hidden layer.
    - kl_multiplier: Multiplier for the KL divergence term in the loss function.
    - print_progress: Whether to print optimization progress.

    Returns:
    - model: Trained Bayesian Neural Network model.
    - nll_history: History of NLL values over the optimization steps.
    - kl_history: History of KL divergence values over the optimization steps.
    - num_samples_history: History of the number of samples used at each step.
    """

    # Set seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    input_dim = x.shape[1]  # Dynamically determine input dimension

    # Normalize known_var to be a safe tensor
    if isinstance(known_var, torch.Tensor):
        known_var_tensor = known_var.clone().detach()
    else:
        known_var_tensor = torch.tensor(known_var, dtype=torch.float32)


    # Initialize the model
    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=input_dim, out_features=num_weights, bias=True),
        nn.Tanh(),
        bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=num_weights, out_features=1, bias=True),
    )

    # Define loss function and optimizer
    nllfn = nn.GaussianNLLLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track history
    nll_history = []
    kl_history = []
    num_samples_history = []
    num_samples = initial_samples

    # Training loop
    for step in range(num_steps):
        optimizer.zero_grad()

        # Monte Carlo approximation of the NLL
        nll_total = 0.0
        for _ in range(num_samples):
            pre = model(x)
            nll = nllfn(pre, y, torch.full_like(pre, known_var_tensor))
            nll_total += nll
        avg_nll = nll_total / num_samples

        # Compute KL divergence
        kl_divergence = compute_kl_divergence(model)

        # Compute total cost (with KL multiplier) and backpropagate
        cost = avg_nll + kl_multiplier * kl_divergence
        cost.backward()
        optimizer.step()

        # Track metrics
        nll_history.append(avg_nll.item())
        kl_history.append(kl_divergence.item())
        num_samples_history.append(num_samples)

        # Print progress
        if print_progress and step % 1000 == 0:
            print(f"[Variational-KL] Step {step} - Averaged NLL: {avg_nll.item():.2f}, KL Divergence: {kl_divergence.item():.2f}, Samples: {num_samples}")

    return model, nll_history, kl_history, num_samples_history


def compute_kl_divergence(model):
    """
    Compute the KL divergence between the variational posterior and prior distributions
    for a Bayesian Neural Network model.

    Parameters:
    - model: A torch.nn.Sequential model containing Bayesian layers (torchbnn.BayesLinear).

    Returns:
    - kl_divergence: The KL divergence (scalar).
    """
    kl_divergence = 0.0
    for layer in model:
        if isinstance(layer, BayesLinear):
            # Variational parameters
            variational_mu = layer.weight_mu
            variational_log_sigma = layer.weight_log_sigma
            variational_sigma = torch.exp(variational_log_sigma)

            # Prior parameters
            prior_mu = layer.prior_mu
            prior_sigma = layer.prior_sigma

            # KL divergence for weights
            kl_weights = (
                torch.log(prior_sigma / variational_sigma) +
                (variational_sigma**2 + (variational_mu - prior_mu)**2) / (2 * prior_sigma**2) - 0.5
            ).sum()

            kl_divergence += kl_weights

            # KL divergence for biases (if present)
            if layer.bias_mu is not None:
                variational_mu_bias = layer.bias_mu
                variational_log_sigma_bias = layer.bias_log_sigma
                variational_sigma_bias = torch.exp(variational_log_sigma_bias)

                kl_bias = (
                    torch.log(prior_sigma / variational_sigma_bias) +
                    (variational_sigma_bias**2 + (variational_mu_bias - prior_mu)**2) / (2 * prior_sigma**2) - 0.5
                ).sum()

                kl_divergence += kl_bias

    return kl_divergence