# src/utils.py

import torch

def load_lhs_parameters(lhs_file, index):
    import pandas as pd
    lhs = pd.read_csv(lhs_file)
    row = lhs.iloc[index]
    return row.to_dict(), index

def load_data_pt(data_dir):
    X_train = torch.load(f"{data_dir}/X_train.pt")
    y_train = torch.load(f"{data_dir}/y_train.pt")
    X_test = torch.load(f"{data_dir}/X_test.pt")
    y_test = torch.load(f"{data_dir}/y_test.pt")
    return X_train, y_train, X_test, y_test

def load_noise_var(data_dir):
    return torch.load(f"{data_dir}/noise_var.pt").item()