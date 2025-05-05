#!/bin/bash

METHOD=alpha_renyi
DGM=x1d
TOTAL=750
BATCH_SIZE=25

for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    START=$i
    END=$((i + BATCH_SIZE - 1))
    if (( END >= TOTAL )); then
        END=$((TOTAL - 1))
    fi

    sbatch run_job.sh $METHOD $DGM $START $END
done
