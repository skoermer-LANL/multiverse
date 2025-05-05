#!/bin/bash -l
#SBATCH --job-name=tgp_pair
#SBATCH --output=logs/tgp_pair_%j.out
#SBATCH --error=logs/tgp_pair_%j.err
#SBATCH --partition=shared-spr 

# === Environment ===
module reset
export PATH=$HOME/R-mkl/bin:$PATH  

# === Parse arguments for both jobs ===
METHOD1=$1
DGM1=$2
RESPONSE1=$3

METHOD2=$4
DGM2=$5
RESPONSE2=$6

cd $SLURM_SUBMIT_DIR/src

# Run both jobs in parallel
Rscript run_tgp_sens.R $METHOD1 $DGM1 $RESPONSE1 &
Rscript run_tgp_sens.R $METHOD2 $DGM2 $RESPONSE2 &

wait
echo "Both sensitivity analyses completed on node $(hostname)"
