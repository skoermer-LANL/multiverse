#!/bin/bash

# RESPONSE="interval_score"
# ANALYSIS_TYPE="sens"

# # 4 combinations
# METHODS=("kl_div" "kl_div" "alpha_renyi" "alpha_renyi")
# DGMS=("x1d" "x2d" "x1d" "x2d")

# # Define partitions — modify as needed
# PARTITIONS=("shared-spr" "shared-spr" "shared-spr" "shared-spr")

# for i in "${!METHODS[@]}"; do
#   METHOD=${METHODS[$i]}
#   DGM=${DGMS[$i]}
#   PARTITION=${PARTITIONS[$i]}

#   echo "🚀 Submitting: $METHOD $DGM $RESPONSE $ANALYSIS_TYPE to partition $PARTITION"

#   sbatch --partition=$PARTITION run_tgp_one.sh $METHOD $DGM $RESPONSE $ANALYSIS_TYPE
# done

RESPONSE_METRICS=("interval_score" "rmse")
METHODS=("kl_div" "alpha_renyi")
DGMS=("x1d" "x2d")

# === Flatten combinations into pairs of two ===
COMBOS=()
for response in "${RESPONSE_METRICS[@]}"; do
  for method in "${METHODS[@]}"; do
    for dgm in "${DGMS[@]}"; do
      COMBOS+=("$method $dgm $response")
    done
  done
done

# === Submit jobs two at a time ===
for ((i=0; i<${#COMBOS[@]}; i+=2)); do
  CMD1="${COMBOS[$i]}"
  CMD2="${COMBOS[$((i+1))]}"
  echo "Submitting job pair:"
  echo "  $CMD1"
  echo "  $CMD2"
  sbatch run_tgp_pair.sh $CMD1 $CMD2
done

