# src/alpha_renyi.py

import torch.nn as nn
import torch
import torch.optim as optim

import torchbnn as bnn



def run_regression(params, X, y, noise_var):
    alpha = params["alpha"]
    log_sigma = params["log_sigma"]
    num_steps = int(params["num_steps"])
    num_weights = int(params["num_weights"])
    initial_samples = int(params["initial_samples"])
    log_lr = params["log_lr"]
    prior_mu = params["prior_mu"]

    prior_sigma = 10**log_sigma
    lr = 10**log_lr

    model, loss, _ = fit_variational_arenyi_bnn(
        x = X, y = y,
        prior_mu = prior_mu,
        prior_sigma = prior_sigma,
        alpha=alpha,
        known_var = noise_var,
        num_weights_layer=num_weights,
        num_steps = num_steps,
        lr = lr,
        initial_samples =initial_samples,
        print_progress = False
    )

    loss = loss[-1]

    return model, loss

def fit_variational_arenyi_bnn(
    x, y,
    prior_mu=0,
    prior_sigma=1,
    alpha=0.1,
    known_var=1.0,
    num_weights_layer=10,
    num_steps=20000,
    lr=0.01,
    initial_samples=1,
    print_progress=True
):
    """
    Fit a Bayesian Neural Network using variational inference with the alpha-Rényi divergence from the 
    variational posterior to the true posterior as the objective.

    Parameters:
    - x: Training inputs (tensor of shape (n_samples, n_features)).
    - y: Training targets (tensor).
    - prior_mu: Mean of the prior distributions for the weights and biases.
    - prior_sigma: Standard deviation of the prior distributions for the weights and biases.
    - alpha: The alpha parameter for the Rényi divergence.
    - known_var: Known variance of the data noise.
    - num_weights_layer: Number of weights in the hidden layer.
    - num_steps: Number of optimization steps.
    - lr: Learning rate for the optimizer.
    - initial_samples: Initial number of Monte Carlo samples.
    - adjust_interval: Interval (in steps) for adjusting the number of samples.
    - window_size: Number of steps in each window for variance estimation.
    - p_value_threshold: Threshold for the chi-squared test in sample adjustment.
    - use_autoregressive_adjustment: Whether to use autoregressive adjustment for sample size.
    - print_progress: Whether to print optimization progress.

    Returns:
    - model: Trained Bayesian Neural Network model
    - loss_history: History of the loss values over the optimization steps.
    - num_samples_history: History of the number of samples used at each step.
    """
    input_dim = x.shape[1]  # Automatically set input dimension based on x

    known_var_tensor = torch.tensor(known_var, dtype=torch.float32, device=x.device)




    # Define the BNN model
    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=input_dim, out_features=num_weights_layer, bias=True),
        nn.Tanh(),
        bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=num_weights_layer, out_features=1, bias=True),
    )

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize histories
    loss_history = []
    num_samples_history = []
    num_samples = initial_samples

    # Training loop
    for step in range(num_steps):
        optimizer.zero_grad()

        # Update normalization term
        normalization_term = torch.tensor(0.0)
        num_data_points = x.shape[0]
        num_weights = sum(
            layer.weight_mu.numel() + (layer.bias_mu.numel() if layer.bias_mu is not None else 0)
            for layer in model if isinstance(layer, bnn.BayesLinear)
            )

        for layer in model:
            if isinstance(layer, bnn.BayesLinear):
                prior_sigma = torch.tensor(layer.prior_sigma, dtype=torch.float32)

                # Weights
                variational_log_sigma = layer.weight_log_sigma
                variational_sigma = torch.exp(variational_log_sigma)
                normalization_term += torch.sum(torch.log(prior_sigma / variational_sigma))

                # ✅ Biases (if present)
                if layer.bias_log_sigma is not None:
                    variational_bias_sigma = torch.exp(layer.bias_log_sigma)
                    normalization_term += torch.sum(torch.log(prior_sigma / variational_bias_sigma))


        normalization_term += 0.5 * (
            #num_data_points * torch.log(torch.tensor(known_var, dtype=torch.float32)) +
            #num_weights * torch.log(torch.tensor(2 * np.pi, dtype=torch.float32))
            num_data_points * torch.log(known_var_tensor) +
            num_weights * torch.log(torch.tensor(2 * torch.pi, dtype=torch.float32))
        )

        # Monte Carlo approximation of the integral
        combined_terms = []
        for _ in range(num_samples):
            theta_samples = {}
            for layer_idx, layer in enumerate(model):
                if isinstance(layer, bnn.BayesLinear):
                    prior_sigma = torch.tensor(layer.prior_sigma, dtype=torch.float32)

                    # Variational parameters
                    weight_mu = layer.weight_mu
                    weight_sigma = torch.exp(layer.weight_log_sigma)

                    # Reshape
                    if layer_idx == 0:
                        # Input to hidden
                        weight_mu = weight_mu.reshape(layer.out_features, layer.in_features).T
                        weight_sigma = weight_sigma.reshape_as(weight_mu)
                    elif layer_idx == 2:
                        # Hidden to output
                        weight_mu = weight_mu.reshape(layer.out_features, layer.in_features).T
                        weight_sigma = weight_sigma.reshape_as(weight_mu)

                    # Sample weights
                    weight_sample = weight_mu + weight_sigma * torch.randn_like(weight_mu)

                    # Sample bias
                    bias_mu = layer.bias_mu
                    bias_sigma = torch.exp(layer.bias_log_sigma)
                    if bias_mu is not None:
                        bias_mu = bias_mu.reshape(layer.out_features)
                        bias_sigma = bias_sigma.reshape_as(bias_mu)
                        bias_sample = bias_mu + bias_sigma * torch.randn_like(bias_mu)
                    else:
                        bias_mu = bias_sigma = bias_sample = None

                    # Store all
                    theta_samples[f"layer_{layer_idx}_weights"] = weight_sample
                    theta_samples[f"layer_{layer_idx}_weight_mu"] = weight_mu
                    theta_samples[f"layer_{layer_idx}_weight_sigma"] = weight_sigma

                    if bias_sample is not None:
                        theta_samples[f"layer_{layer_idx}_biases"] = bias_sample
                        theta_samples[f"layer_{layer_idx}_bias_mu"] = bias_mu
                        theta_samples[f"layer_{layer_idx}_bias_sigma"] = bias_sigma

            # Compute predictions
            hidden = torch.tanh(x @ theta_samples["layer_0_weights"] + theta_samples["layer_0_biases"])
            pre = hidden @ theta_samples["layer_2_weights"] + theta_samples["layer_2_biases"]

            # Quadratic terms and likelihood
            quadratic_prior = torch.tensor(0.0)
            quadratic_variational = torch.tensor(0.0)

            for layer_idx, layer in enumerate(model):
                if isinstance(layer, bnn.BayesLinear):
                    prior_mu = layer.prior_mu
                    prior_sigma = torch.tensor(layer.prior_sigma, dtype=torch.float32)

                    # Weights
                    variational_mu = theta_samples[f"layer_{layer_idx}_weight_mu"]
                    variational_sigma = theta_samples[f"layer_{layer_idx}_weight_sigma"] 
                    theta_w = theta_samples[f"layer_{layer_idx}_weights"]
                    variational_mu = variational_mu.reshape_as(theta_w)
                    variational_sigma = variational_sigma.reshape_as(theta_w)


                    qv_weights = torch.sum((theta_w - variational_mu)**2 / variational_sigma**2)
                    quadratic_variational += qv_weights

                    qp_weights = torch.sum((theta_w - prior_mu)**2 / prior_sigma**2)
                    quadratic_prior += qp_weights

                    # Biases
                    if f"layer_{layer_idx}_biases" in theta_samples:
                        bias_sample = theta_samples[f"layer_{layer_idx}_biases"]
                        prior_bias_mu = torch.ones_like(bias_sample) * layer.prior_mu
                        prior_bias_sigma = torch.ones_like(bias_sample) * prior_sigma
                        variational_bias_mu = theta_samples[f"layer_{layer_idx}_bias_mu"]
                        variational_bias_sigma = theta_samples[f"layer_{layer_idx}_bias_sigma"]

                        qv_biases = torch.sum((bias_sample - variational_bias_mu)**2 / variational_bias_sigma**2)
                        quadratic_variational += qv_biases
                        #quadratic_variational_terms["layer_biases"].append(qv_biases.item())
                        qp_biases = torch.sum((bias_sample - prior_bias_mu)**2 / prior_bias_sigma**2)
                        quadratic_prior += qp_biases

            residuals = y - pre
            likelihood_term = torch.sum((residuals**2) / known_var)

            combined_term = -0.5 * (alpha - 1) * (quadratic_variational - quadratic_prior - likelihood_term)
            combined_terms.append(combined_term)

        combined_terms = torch.stack(combined_terms)
        
        log_mc_integral = torch.logsumexp(combined_terms, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32))
        renyi_divergence = (1 / (alpha - 1)) * log_mc_integral + normalization_term

        loss = renyi_divergence
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        num_samples_history.append(num_samples)

        if print_progress and step % 1000 == 0:
            print(f"Step {step} - Loss: {loss.item():.2f}, Samples: {num_samples}")
            #print(f"Combined terms variance:  {torch.var(combined_terms_save):.1f}")

    return model, loss_history, num_samples_history

