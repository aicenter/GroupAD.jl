#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=32G

MODEL=$1
DATASET=$2

module load Python/3.8
module load Julia/1.7.3-linux-x86_64

julia --project ./knn_basic.jl ${MAX_SEED} events_anomalydetection_v2.h5 $CONTAMINATION