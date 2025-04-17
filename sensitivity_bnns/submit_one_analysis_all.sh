#!/bin/bash

RESPONSE="interval_score"      # Set your response of interest here
ANALYSIS_TYPE="sens"           # Options: sens, plot, improv
USE_NODE_SELECTION=true        # Set false for debug/general testing

METHODS=("kl_div" "kl_div" "alpha_renyi" "alpha_renyi")
DGMS=("x1d" "x2d" "x1d" "x2d")

if $USE_NODE_SELECTION; then
    NODE_CONSTRAINT=$(bash choose_node_constraint.sh)
    if [[ "$NODE_CONSTRAINT" == "none" ]]; then
        echo "No preferred nodes available — using default partition."
        NODE_FLAG=""
    else
        echo "Using node constraint: $NODE_CONSTRAINT"
        NODE_FLAG="--constraint=$NODE_CONSTRAINT"
    fi
else
    echo "Debug mode: no node constraint"
    NODE_FLAG=""
fi

for i in "${!METHODS[@]}"; do
  METHOD=${METHODS[$i]}
  DGM=${DGMS[$i]}
  echo "Submitting: $METHOD $DGM $RESPONSE $ANALYSIS_TYPE"
  sbatch $NODE_FLAG run_tgp_one.sh $METHOD $DGM $RESPONSE $ANALYSIS_TYPE
done
