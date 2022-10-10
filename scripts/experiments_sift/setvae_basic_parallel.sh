#!/bin/bash

MAX_SEED=$1  
# seed, if seed =< 0 it is considered concrete single seed to train with
DATASET=$2
CONTAMINATION=$3
# training data contamination rate


# FIX RANDOM SEED FOR THIS CALL
RANDOM_SEED=$RANDOM
# random seed for sample_params function (to be able to train multile seeds in parallel)

module load Julia/1.7.2-linux-x86_64
julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '

LOG_DIR="${HOME}/logs/SetVAE-parts/mvtec-sift/"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

for((i=1; i<=$MAX_SEED; i++ ))
do 
    sbatch \
    --output="${LOG_DIR}/${DATASET}-${RANDOM_SEED}_${i}.out" \
    ./setvae_basic.sh -$i ${DATASET} ${CONTAMINATION} ${RANDOM_SEED}
    #julia --project ./setvae_basic.jl -$i $DATASET -$j $METHOD $CONTAMINATION ${RANDOM_SEED}
    # i/j is negative => special case in setvae_basic.jl => parallel training
done
