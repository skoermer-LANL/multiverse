# Bayesian Neural Network Regression Experiments

SUMMARY


##  1. Generating Synthetic Data & LHS Designs

All dataset and LHS generation is handled by:

```bash
python src/prepare_experiment.py
```


###  Command-Line Usage

```bash
python src/prepare_experiment.py [--lhs METHOD DGM] [--samples N] [--data]
```

###  Examples

#### Generate everything (default):
Generates both 1D and 2D datasets and all 4 LHS files with 100 samples each.

```bash
python src/prepare_experiment.py
```

#### Generate only the training/testing datasets (1D and 2D):
```bash
python src/prepare_experiment.py --data
```

#### Generate only a specific LHS (e.g., alpha_Renyi on 1D inputs):
```bash
python src/prepare_experiment.py --lhs alpha_renyi x1d --samples 50
```

#### Generate all LHS files with a specific number of samples:
```bash
python src/prepare_experiment.py --samples 25
```

##  2. Setting Up the Python Environment

You can use **Conda** (recommended) or `venv`.

### Option A: Conda (Recommended)

Create the environment:

```bash
conda create -n bnnreg python=3.10
conda activate bnnreg
```

Install packages:

```bash
pip install -r requirements.txt
```

Or recreate from YAML (if available):

```bash
conda env create -f environment.yml
conda activate bnnreg
```

### Option B: Using `venv`

```bash
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 3. Dependency Tracking

This repo supports both `requirements.txt` and `environment.yml`.

### ✅ `requirements.txt`

Install with:

```bash
pip install -r requirements.txt
```

Sample contents:

```text
# Requires Python >= 3.10
torch>=2.0
numpy
pandas
scipy
```

### `environment.yml` (optional)

Export from Conda:

```bash
conda env export --from-history > environment.yml
```

Recreate environment:

```bash
conda env create -f environment.yml
```

## 5. Methods and Data Mechanisms

### Regression Methods:

- `kl_div` — Variational BNN with KL divergence
- `alpha_renyi` — Posterior approximation using alpha-Rényi divergence

### Data Generating Mechanisms:

- `x1d` — Univariate input in [0, 1]
- `x2d` — Bivariate input in [0, 1]^2

---

## 6. Merging Results After Experiment Runs

Each experiment writes its results to separate files for safe parallel execution.

To merge them into a single CSV after all runs complete:

```bash
python src/merge_results.py METHOD DGM
```

