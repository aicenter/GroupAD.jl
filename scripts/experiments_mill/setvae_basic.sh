#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpulong
#SBATCH --mem=30G


MAX_SEED=$1  
# seed, if seed =< 0 it is considered concrete single seed to train with
DATASET=$2
CONTAMINATION=$3
# training data contamination rate
RANDOM_SEED=$4
# random seed for sample_params function (to be able to train multile seeds in parallel)

module load Julia/1.7.2-linux-x86_64
julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '

julia --project ./setvae_basic.jl ${MAX_SEED} $DATASET $CONTAMINATION ${RANDOM_SEED}

# ##SBATCH --gres=gpu:1
# ##SBATCH --partition=gpu