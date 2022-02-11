#!/bin/bash

# ==================================================================================
# Single policy
# ==================================================================================
Terrains="flat"
declare -a Experiments=(
                        "1"
                        )
declare -a Arguments=(
                      ""
                      )

for terrain in $Terrains; do 
    for (( i=0; i<${#Arguments[@]}; i++ )); do 
      sbatch ./base_csiro_single.sh $terrain ${Experiments[$i]} "${Arguments[$i]}"
    done
done