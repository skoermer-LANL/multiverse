# Bayesian Neural Network Regression Experiments

This repository accompanies a study on Bayesian neural network (BNN) regression using variational inference with different divergence objectives and under multiple synthetic data generating mechanisms.

The experiments explore the impact of hyperparameter tuning on RMSE, interval score, and posterior coverage.


##  Directory Structure

```

.
├── src/                      # All Python and R code
│   ├── alpha\_renyi.py        # α-Rényi divergence BNN training logic
│   ├── kl\_div.py             # KL divergence BNN training logic
│   ├── config.py             # File paths and noise variance for each DGM
│   ├── utils.py              # LHS and data loading helpers
│   ├── evaluate.py           # Evaluation metrics (RMSE, interval score, HDR)
│   ├── prepare\_experiment.py # Generates datasets and LHS samples
│   ├── main.py               # Runs one regression for one set of parameters
│   ├── run\_batch.py          # Helper to run batches from custom CSVs
│   ├── merge\_results.py      # Merges result CSVs per experiment
│   ├── plot\_best\_worst\_from\_saved.py # Makes diagnostic plots from saved results
│   ├── test\_all\_local.py     # Utility for running local test sweeps
│   ├── run\_tgp\_plot.R        # Posterior surface visualization with TGP
│   ├── run\_tgp\_sens.R        # Global sensitivity analysis using TGP
│
├── lhs/                      # LHS hyperparameter grids
├── data/                     # Generated training/testing datasets (.pt)
├── results/                  # Individual and merged fit results
├── figures/                  # Publication-quality plots
├── environment.yml           # Conda environment file (minimal)
├── requirements.txt          # Python package list
├── .gitignore
└── README.md                 # This file

````


##  Setup Instructions

###  Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate bnnreg
````

Or manually:

```bash
conda create -n bnnreg python=3.10
conda activate bnnreg
pip install -r requirements.txt
```



##  1. Generate Data + LHS Designs

All synthetic data and hyperparameter samples are generated using:

```bash
python src/prepare_experiment.py
```

###  Optional Arguments:

* `--data`: generate training/testing datasets only
* `--lhs METHOD DGM`: generate LHS for method/DGM combo
* `--samples N`: number of LHS samples to generate

###  Examples

```bash
# Generate everything
python src/prepare_experiment.py

# Only generate LHS for α-Rényi on 1D input
python src/prepare_experiment.py --lhs alpha_renyi x1d --samples 100
```


##  2. Run Regression Experiments

To run a single experiment:

```bash
python src/main.py kl_div x1d 42
```

Where:

* `kl_div` or `alpha_renyi` are methods
* `x1d` or `x2d` are data generating mechanisms
* `42` is the row index in the LHS .csv file

You can run batches with:

```bash
python src/run_batch.py kl_div x1d custom_inputs.csv
```

Or for local parallel testing:

```bash
python src/test_all_local.py
```


##  3. Merge Results

After experiments complete, merge the individual CSV outputs:

```bash
python src/merge_results.py kl_div x1d
```

This creates:

```
results/kl_div_x1d/merged_results.csv
```



##  4. Plot Best/Worst Fits

Generate plots using saved posterior summaries:

```bash
python src/plot_best_worst_from_saved.py interval_score kl_div
python src/plot_best_worst_from_saved.py both_metrics alpha_renyi
```

This creates PDF and CSV summaries in `figures/` and `results/`.



##  5. Sensitivity Analysis and Visualization (R)

You may have to edit arguments in the script related to Monte Carlo methods.

###  TGP Sensitivity:

```r
source("src/run_tgp_sens.R")
```

### Predictive Plot Surface:

```r
source("src/run_tgp_plot.R")
```



##  Methods and Data

###  Regression Methods:

* `kl_div`: Variational inference using KL divergence
* `alpha_renyi`: Variational inference using α-Rényi divergence

###  Data Generating Mechanisms:

* `x1d`: $y = x^3 - x^2 + \varepsilon, \varepsilon \sim \mathcal{N}(0, \sigma^2)$
* `x2d`: Quadratic surface in 2D (defined in `prepare_experiment.py`)



##  HPC/Slurm Note

This repository omits Slurm scripts due to HPC-specific constraints. To adapt the code for parallel runs:

* Use your own batch scheduler to distribute calls to `main.py`
* Ensure paths in `config.py` and `results/` are writeable per job
* Post-process results using `merge_results.py`



##  License and Citation

If you use this codebase, please cite the associated publication or repository DOI (to be added upon acceptance).
