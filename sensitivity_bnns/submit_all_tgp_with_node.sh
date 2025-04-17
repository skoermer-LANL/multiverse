#!/bin/bash

METHOD=kl_div
DGM=x1d
METRICS=("loss" "rmse" "coverage" "interval_score")
USE_NODE_SELECTION=true  # Set to false for debugging

if $USE_NODE_SELECTION; then
    NODE_CONSTRAINT=$(bash choose_node_constraint.sh)
    if [[ "$NODE_CONSTRAINT" == "none" ]]; then
        echo "No preferred nodes available — submitting to general pool."
        NODE_FLAG=""
    else
        echo "Using preferred node type: $NODE_CONSTRAINT"
        NODE_FLAG="--constraint=$NODE_CONSTRAINT"
    fi
else
    echo "Debug mode: no constraint"
    NODE_FLAG=""
fi

for METRIC in "${METRICS[@]}"; do
    sbatch $NODE_FLAG run_tgp.sh $METHOD $DGM $METRIC
done
