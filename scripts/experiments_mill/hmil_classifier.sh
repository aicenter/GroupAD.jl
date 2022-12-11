#!/bin/bash
#SBATCH --partition=cpulong
#SBATCH --time=35:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=12G

MAX_SEED=$1
DATASET=$2
CONTAMINATION=$3

module load Julia/1.7.3-linux-x86_64

julia --project ./hmil_classifier.jl ${MAX_SEED} $DATASET $CONTAMINATION