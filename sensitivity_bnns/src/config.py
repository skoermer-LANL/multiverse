# src/config.py

def get_paths(method, dgm):
    lhs_path = f"lhs/{method}_{dgm}_lhs.csv"
    data_dir = f"data/{dgm}/"
    result_path = f"results/{method}_{dgm}/"
    return lhs_path, data_dir, data_dir, result_path

def get_noise_variance(dgm):
    import torch
    path = f"data/{dgm}/noise_var.pt"
    return torch.load(path).item()
