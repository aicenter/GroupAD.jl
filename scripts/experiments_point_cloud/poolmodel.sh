#!/bin/bash
#SBATCH --partition=cpulong
#SBATCH --time=30:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=10
#SBATCH --mem=100G

MAX_SEED=$1
DATASET=$2
ANOMALY_CLASSES=$3
METHOD=$4
CONTAMINATION=$5

module load Julia/1.7.2-linux-x86_64

julia --threads 10 ./poolmodel.jl ${MAX_SEED} $DATASET ${ANOMALY_CLASSES} $METHOD $CONTAMINATION
