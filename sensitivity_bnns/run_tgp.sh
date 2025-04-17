#!/bin/bash -l
#SBATCH --job-name=tgp_analysis
#SBATCH --output=logs/tgp_%x_%j.out
#SBATCH --error=logs/tgp_%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=compute


module reset
export PATH=$HOME/R-mkl/bin:$PATH  # MKL-optimized R

METHOD=$1
DGM=$2
RESPONSE=$3

cd $SLURM_SUBMIT_DIR/src
Rscript run_tgp_analysis.R $METHOD $DGM $RESPONSE
