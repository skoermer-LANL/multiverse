# src/config.py

def get_paths(method, dgm):
    lhs_path = f"lhs/{method}_{dgm}_lhs_rev1.csv"
    data_dir = f"data/{dgm}_rev1/"
    result_path = f"results/{method}_{dgm}_rev1/"
    #return lhs_path, data_dir, data_dir, result_path
    return lhs_path, data_dir, result_path

def get_noise_variance(dgm):
    import torch
    path = f"data/{dgm}_rev1/noise_var.pt"
    return torch.load(path).item()
