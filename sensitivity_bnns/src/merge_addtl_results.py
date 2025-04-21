# src/merge_addtl_results.py

import os
import pandas as pd

# List of combinations you want to process
combinations = [
    ("kl_div", "x1d"),
    ("kl_div", "x2d"),
    ("alpha_renyi", "x1d"),
    ("alpha_renyi", "x2d")
]

# Loop over each combination
for method, dgm in combinations:
    result_dir = os.path.join("..", "results", f"{method}_{dgm}")
    merged_filename = os.path.join(result_dir, "merged_addtl_results.csv")

    # Gather all result_*.csv files EXCEPT the original 750-row ones
    result_files = [f for f in os.listdir(result_dir)
                    if f.startswith("result_") and f.endswith("optimal.csv")
                    and "addtl" not in f.lower()  # in case user has already merged
                    and "merged" not in f.lower()]  # avoid merging old merged files

    print(f"🔍 {method} {dgm} — Found {len(result_files)} result files")

    # Sort files numerically by index
    result_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Load and merge
    dfs = []
    for filename in result_files:
        df = pd.read_csv(os.path.join(result_dir, filename))
        dfs.append(df)

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(merged_filename, index=False)
        print(f"Merged to: {merged_filename}")
    else:
        print(f"No result files found for {method} {dgm}")
