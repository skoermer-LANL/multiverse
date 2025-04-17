#!/bin/bash -l
#SBATCH --job-name=addtl_job
#SBATCH --output=logs/addtl_%j_%x.out
#SBATCH --error=logs/addtl_%j_%x.err
#SBATCH --qos=debug
#SBATCH --time=04:00:00

# === Load environment for Python ===
module reset
module load miniconda3
source activate varbnnenv

# === Parse input arguments ===
METHOD=$1
DGM=$2
START_IDX=$3
END_IDX=$4

cd $SLURM_SUBMIT_DIR

# === Detect resources ===
TOTAL_CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
ESTIMATED_THREADS_PER_PROCESS=6
TARGET_UTILIZATION=80

NUM_PROCS_TO_USE=$(( (TOTAL_CPUS * TARGET_UTILIZATION / 100) / ESTIMATED_THREADS_PER_PROCESS ))
if (( NUM_PROCS_TO_USE < 1 )); then
    NUM_PROCS_TO_USE=1
fi

echo "----------------------------------------"
echo "Node CPU cores available         : $TOTAL_CPUS"
echo "Estimated threads per process    : $ESTIMATED_THREADS_PER_PROCESS"
echo "Target CPU utilization           : ${TARGET_UTILIZATION}%"
echo "Launching $NUM_PROCS_TO_USE parallel Python processes"
echo "Processing rows $START_IDX to $END_IDX"
echo "----------------------------------------"

# === Semaphore setup ===
SEMAPHORE=/tmp/sema$$
mkfifo $SEMAPHORE
exec 3<>$SEMAPHORE
rm $SEMAPHORE

for ((n=0; n<NUM_PROCS_TO_USE; n++)); do
  echo >&3
done

# === Run in parallel ===
for ((i=START_IDX; i<=END_IDX; i++)); do
  read -u 3
  {
    echo "[$(date)] Running: $METHOD $DGM index $i"
    python src/main.py $METHOD $DGM $i addtl_runs/${METHOD}_${DGM}_extra.csv
    echo "[$(date)] Done: $METHOD $DGM index $i"
    echo >&3
  } &
done

wait
exec 3>&-

echo "[$(date)] All tasks complete."
