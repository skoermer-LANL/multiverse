#!/bin/bash

TOTAL=11
BATCH_SIZE=2  # Adjust as needed

METHODS=("kl_div" "kl_div" "alpha_renyi" "alpha_renyi")
DGMS=("x1d" "x2d" "x1d" "x2d")

for idx in "${!METHODS[@]}"; do
  METHOD=${METHODS[$idx]}
  DGM=${DGMS[$idx]}

  for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    START=$i
    END=$((i + BATCH_SIZE - 1))
    if (( END >= TOTAL )); then
        END=$((TOTAL - 1))
    fi

    echo "Submitting $METHOD $DGM: rows $START–$END"
    sbatch run_addtl_job.sh $METHOD $DGM $START $END
  done
done
