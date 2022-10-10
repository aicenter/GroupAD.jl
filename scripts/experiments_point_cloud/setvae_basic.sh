#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=30G

MAX_SEED=$1  
# seed, if seed =< 0 it is considered concrete single seed to train with
DATASET=$2
ANOMALY_CLASSES=$3
# number of anomaly classes, if =< 0 then considered as single anomaly class to train with
METHOD=$4
# method for data creation -> \"leave-one-out\" or \"leave-one-in\" 
CONTAMINATION=$5
# training data contamination rate
RANDOM_SEED=$6
# random seed for sample_params function (to be able to train multile seeds in parallel)

module load Julia/1.7.2-linux-x86_64
julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '

julia --project ./setvae_basic.jl ${MAX_SEED} $DATASET ${ANOMALY_CLASSES} $METHOD $CONTAMINATION ${RANDOM_SEED}