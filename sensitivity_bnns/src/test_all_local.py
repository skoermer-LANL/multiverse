# src/test_all_local.py

import subprocess

methods = ["kl_div", "alpha_renyi"]
dgms = ["x1d", "x2d"]
n_rows = 3

for method in methods:
    for dgm in dgms:
        for idx in range(n_rows):
            print(f"\n Running: {method} | {dgm} | LHS row {idx}")
            try:
                subprocess.run(
                    ["python", "src/main.py", method, dgm, str(idx)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed: {method} | {dgm} | index {idx}")
                print(e)
