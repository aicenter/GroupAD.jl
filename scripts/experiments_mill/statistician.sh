#!/bin/bash
#SBATCH --partition=cpulong
#SBATCH --time=35:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=20G

MAX_SEED=$1
DATASET=$2
CONTAMINATION=$3

module load Julia/1.7.2-linux-x86_64

julia ./statistician.jl ${MAX_SEED} $DATASET $CONTAMINATION