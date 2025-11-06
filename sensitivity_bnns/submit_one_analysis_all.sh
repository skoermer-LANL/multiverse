#!/bin/bash

#RESPONSE="interval_score"
RESPONSE1="interval_score"
RESPONSE2="rmse"
ANALYSIS_TYPE="sens"

# 4 combinations
METHODS=("kl_div" "kl_div" "alpha_renyi" "alpha_renyi")
DGMS=("x1d" "x2d" "x1d" "x2d")

# Define partitions — modify as needed
PARTITIONS=("shared-spr" "shared-spr" "shared-spr" "shared-spr")

for i in "${!METHODS[@]}"; do
  METHOD=${METHODS[$i]}
  DGM=${DGMS[$i]}
  PARTITION=${PARTITIONS[$i]}

  #echo "🚀 Submitting: $METHOD $DGM $RESPONSE $ANALYSIS_TYPE to partition $PARTITION"
  echo "🚀 Submitting: $METHOD $DGM $RESPONSE1 $RESPONSE2 $ANALYSIS_TYPE to partition $PARTITION"

  #sbatch --partition=$PARTITION run_tgp_one.sh $METHOD $DGM $RESPONSE $ANALYSIS_TYPE
  sbatch --partition=$PARTITION run_tgp_pair.sh $METHOD $DGM $RESPONSE1 $RESPONSE2
done


