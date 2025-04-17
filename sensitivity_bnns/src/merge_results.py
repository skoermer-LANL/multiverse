import pandas as pd
import glob
import os
import argparse

def merge_results(method, dgm, results_dir="results"):
    result_path = os.path.join(results_dir, f"{method}_{dgm}")
    files = sorted(glob.glob(os.path.join(result_path, "result_*.csv")))

    if not files:
        print(f"No result files found in {result_path}")
        return

    df_list = [pd.read_csv(f) for f in files]
    merged_df = pd.concat(df_list, ignore_index=True)

    output_file = os.path.join(result_path, "merged_results.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge individual result CSVs into one.")
    parser.add_argument("method", type=str, help="Regression method (e.g., kl_div)")
    parser.add_argument("dgm", type=str, help="Data generating mechanism (e.g., x1d)")

    args = parser.parse_args()
    merge_results(args.method, args.dgm)
