#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=cpufast
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=20G

MAX_SEED=$1
DATASET=$2
CONTAMINATION=$3

# module load Julia
module load Julia/1.6.4-linux-x86_64

julia ./bagkNN.jl ${MAX_SEED} $DATASET $CONTAMINATION