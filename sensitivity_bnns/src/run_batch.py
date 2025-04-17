# src/run_batch.py

import sys
import pandas as pd
import subprocess

if len(sys.argv) < 4:
    print("Usage: run_batch.py <method> <dgm> <csv_file>")
    sys.exit(1)

method = sys.argv[1]
dgm = sys.argv[2]
csv_file = sys.argv[3]


df = pd.read_csv(csv_file)

# Loop over each row and call main.py
for i in range(len(df)):
    print(f"Running row {i} of {csv_file}")
    subprocess.run([
        "python",
        "main.py",
        method,
        dgm,
        str(i),
        csv_file  
    ])
