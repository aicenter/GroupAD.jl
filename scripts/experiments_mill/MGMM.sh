#!/bin/bash
#SBATCH --partition=cpufast
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=20G

MAX_SEED=$1
DATASET=$2
CONTAMINATION=$3

module load Python/3.8
module load Julia/1.7.3-linux-x86_64

julia --project ./MGMM.jl ${MAX_SEED} $DATASET $CONTAMINATION