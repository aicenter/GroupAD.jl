#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=40G

MAX_SEED=$1
DATASET=$2
CONTAMINATION=$3

# module load Julia
module load Julia/1.6.4-linux-x86_64

julia ./SMMC.jl ${MAX_SEED} $DATASET $CONTAMINATION