#!/bin/bash -l
#SBATCH --job-name=tgp_one
#SBATCH --output=logs/tgp_%x_%j.out
#SBATCH --error=logs/tgp_%x_%j.err
#SBATCH --time=10:00:00
# Constraint added dynamically via sbatch --constraint=...

module purge
module load intel-mkl/2024.0.0

export LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:$LD_LIBRARY_PATH"

export PATH=$HOME/R-mkl/bin:$PATH


#module reset
#export PATH=$HOME/R-mkl/bin:$PATH

# Match R thread count to Slurm allocation
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

METHOD=$1
DGM=$2
RESPONSE=$3
ANALYSIS=$4  # sens | plot | improv

cd $SLURM_SUBMIT_DIR/src

case "$ANALYSIS" in
  sens)   Rscript run_tgp_sens.R $METHOD $DGM $RESPONSE ;;
  plot)   Rscript run_tgp_plot.R $METHOD $DGM $RESPONSE ;;
  improv) Rscript run_tgp_improv.R $METHOD $DGM $RESPONSE ;;
  *)      echo "Unknown analysis type: $ANALYSIS"; exit 1 ;;
esac
