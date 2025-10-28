import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import qmc
import argparse
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ DATA GENERATION ------------------

def generate_and_save_quad_2d(save_dir, n_train=400, n_test=400, truesd=1.0, seed=123):
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    truevar = truesd ** 2

    X_train = torch.rand((n_train, 2)).float().to(device)
    x1, x2 = X_train[:, 0], X_train[:, 1]
    Ey = 5 + 3*x1 + 4*x2 + 10*x1**2 + 1.5 * x2**2

    EEy = torch.mean(Ey)
    Eyvar = torch.var(Ey, unbiased=False)
    scaling2 = truevar + Eyvar
    scaling = torch.sqrt(scaling2)

    noise = torch.randn_like(Ey) * truesd
    y_train = (Ey + noise - EEy) / scaling
    sig2scale = truevar / scaling2

    X_test = torch.rand((n_test, 2)).float().to(device)
    x1_test, x2_test = X_test[:, 0], X_test[:, 1]
    y_test = 5 + 3*x1_test + 4*x2_test + 10*x1_test**2 + 1.5 * x2_test**2
    y_test = (y_test + torch.randn_like(y_test) * truesd - EEy) / scaling

    torch.save(X_train.cpu(), os.path.join(save_dir, "X_train.pt"))
    torch.save(y_train.unsqueeze(1).cpu(), os.path.join(save_dir, "y_train.pt"))
    torch.save(X_test.cpu(), os.path.join(save_dir, "X_test.pt"))
    torch.save(y_test.unsqueeze(1).cpu(), os.path.join(save_dir, "y_test.pt"))
    torch.save(sig2scale.clone().detach(), os.path.join(save_dir, "noise_var.pt"))

    print(f"[x2d] Data saved to {save_dir} | Scaled noise variance: {sig2scale.item():.5f}")


def generate_and_save_poly_1d(save_dir, n_train=50, n_test=200, truesd=0.5, seed=42):
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    truevar = truesd ** 2

    x_train = torch.rand(n_train, 1) * 4 - 2
    Ey = x_train.pow(3) - x_train.pow(2) + 3

    EEy = torch.mean(Ey)
    Eyvar = torch.var(Ey, unbiased=False)
    scaling2 = truevar + Eyvar
    scaling = torch.sqrt(scaling2)

    y_train = (Ey + truesd * torch.randn_like(x_train) - EEy) / scaling
    sig2scale = truevar / scaling2

    x_test = torch.rand(n_test, 1) * 4 - 2
    y_test = x_test.pow(3) - x_test.pow(2) + 3
    y_test = (y_test + truesd * torch.randn_like(x_test) - EEy) / scaling

    # Rescale inputs from [-2, 2] → [0, 1]
    x_train_rescaled = (x_train + 2) / 4
    x_test_rescaled = (x_test + 2) / 4

    torch.save(x_train_rescaled, os.path.join(save_dir, "X_train.pt"))
    # scaled
    torch.save(y_train, os.path.join(save_dir, "y_train.pt"))
    torch.save(x_test_rescaled, os.path.join(save_dir, "X_test.pt"))
    #scaled
    torch.save(y_test, os.path.join(save_dir, "y_test.pt"))
    #scaled
    torch.save(sig2scale.clone().detach(), os.path.join(save_dir, "noise_var.pt"))

    print(f"[x1d] Data saved to {save_dir} | Scaled noise variance: {sig2scale.item():.5f}")

def generate_and_save_poly_1d_oos(save_dir, n_train=50, n_test=200, truesd=0.5, seed=42):
    ## This version of the dgm is for out of sample training data
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    truevar = truesd ** 2

    # A gap in the training data space is placed on (-1.25, -0.25)
    # Code below is specific to this split, with a length of 1 out of 4
    # Resulting in the idea to place 33% of the points in [-2,-1.25]
    # and 67% of the points in (-1.25, 2]
    n1 = n_train // 3
    n2 = n_train - n1

    u1 = torch.rand(n1, 1)
    u2 = torch.rand(n2, 1)

    x1 = u1 * 0.75 - 2  # -> [-2, -1.25]
    x2 = u2 * 2.25 - 0.25 # -> [-1.25, 2]

    x_train = torch.cat([x1, x2], dim=0)
    Ey = x_train.pow(3) - x_train.pow(2) + 3

    EEy = torch.mean(Ey)
    Eyvar = torch.var(Ey, unbiased=False)
    scaling2 = truevar + Eyvar
    scaling = torch.sqrt(scaling2)

    y_train = (Ey + truesd * torch.randn_like(x_train) - EEy) / scaling
    sig2scale = truevar / scaling2

    ## Change here to expand the location of the test set
    x_test = torch.rand(n_test, 1) * 5 - 2.5
    y_test = x_test.pow(3) - x_test.pow(2) + 3
    y_test = (y_test + truesd * torch.randn_like(x_test) - EEy) / scaling

    # Rescale inputs from [-2.5, 2.5] → [0, 1]
    x_train_rescaled = (x_train + 2.5) / 5
    x_test_rescaled = (x_test + 2.5) / 5

    torch.save(x_train_rescaled, os.path.join(save_dir, "X_train.pt"))
    # scaled
    torch.save(y_train, os.path.join(save_dir, "y_train.pt"))
    torch.save(x_test_rescaled, os.path.join(save_dir, "X_test.pt"))
    #scaled
    torch.save(y_test, os.path.join(save_dir, "y_test.pt"))
    #scaled
    torch.save(sig2scale.clone().detach(), os.path.join(save_dir, "noise_var.pt"))

    print(f"[x1d] Data saved to {save_dir} | Scaled noise variance: {sig2scale.item():.5f}")


# ------------------ LHS GENERATION ------------------

'''
PARAM_SPECS = {
    ("kl_div", "x1d"): {
        "names": ["log_kl_multiplier", "log_sigma", "num_steps", "num_weights",
                  "initial_samples", "log_lr", "prior_mu"],
        "ranges": [(-1, 1), (-0.5, 0.5), (2000, 20000), (2, 100), (1, 25), (-3.3, -0.3), (-2, 2)]
    },
    ("kl_div", "x2d"): {
        "names": ["log_kl_multiplier", "log_sigma", "num_steps", "num_weights",
                  "initial_samples", "log_lr", "prior_mu"],
        "ranges": [(-1, 1), (-0.5, 0.5), (2000, 20000), (2, 100), (1, 25), (-3.3, -0.3), (-2, 2)]
    },
    ("alpha_renyi", "x1d"): {
        "names": ["alpha", "log_sigma", "num_steps", "num_weights",
                  "initial_samples", "log_lr", "prior_mu"],
        "ranges": [(0, 1), (-0.5, 0.5), (2000, 20000), (2, 100), (1, 25), (-3.3, -0.3), (-2, 2)]
    },
    ("alpha_renyi", "x2d"): {
        "names": ["alpha", "log_sigma", "num_steps", "num_weights",
                  "initial_samples", "log_lr", "prior_mu"],
        "ranges": [(0, 1), (-0.5, 0.5), (2000, 20000), (2, 100), (1, 25), (-3.3, -0.3), (-2, 2)]
    }
}

'''

with open("param_specs_v1.yaml", "r") as f:
    PARAM_SPECS = yaml.safe_load(f)

def generate_lhs(param_ranges, param_names, n_samples, filename):
    sampler = qmc.LatinHypercube(d=len(param_ranges))
    sample = sampler.random(n=n_samples)
    scaled = qmc.scale(sample,
                       [r[0] for r in param_ranges],
                       [r[1] for r in param_ranges])
    df = pd.DataFrame(scaled, columns=param_names)

    # Round specific parameters to integers
    integer_columns = {"num_steps", "num_weights", "initial_samples"}
    for col in df.columns:
        if col in integer_columns:
            df[col] = df[col].round().astype(int)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[LHS] Saved to {filename}")



def generate_lhs_for(method, dgm, n_samples=100):
    key = (method, dgm)
    if key not in PARAM_SPECS:
        raise ValueError(f"No parameter spec defined for method={method}, dgm={dgm}")
    spec = PARAM_SPECS[key]
    filename = f"lhs/{method}_{dgm}_lhs.csv"
    generate_lhs(spec["ranges"], spec["names"], n_samples, filename)


def generate_all_lhs(n_samples=100):
    for method, dgm in PARAM_SPECS:
        generate_lhs_for(method, dgm, n_samples)


# ------------------ CLI ENTRY POINT ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experimental data and LHS files.")
    parser.add_argument("--lhs", nargs=2, metavar=("METHOD", "DGM"),
                        help="Generate LHS for specific method and DGM (e.g., --lhs alpha_renyi x1d)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of LHS samples to generate (default: 100)")
    parser.add_argument("--data", action="store_true",
                        help="Generate all training/testing data (x1d and x2d)")

    args = parser.parse_args()

    if args.lhs:
        method, dgm = args.lhs
        generate_lhs_for(method, dgm, n_samples=args.samples)
    elif args.data:
        generate_and_save_poly_1d("data/x1d/")
        generate_and_save_quad_2d("data/x2d/")
    else:
        # Default: do everything
        generate_and_save_poly_1d("data/x1d/")
        generate_and_save_quad_2d("data/x2d/")
        generate_all_lhs(n_samples=args.samples)
