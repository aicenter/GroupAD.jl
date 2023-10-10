#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2
#SBATCH --mem=10G

MAX_SEED=$1
DATASET=$2
ANOMALY_CLASSES=$3
METHOD=$4
CONTAMINATION=$5

module load Julia/1.7.3-linux-x86_64

# julia --threads 10 ./vae_basic.jl ${MAX_SEED} $DATASET ${ANOMALY_CLASSES} $METHOD $CONTAMINATION
julia --project ./vae_basic.jl ${MAX_SEED} $DATASET ${ANOMALY_CLASSES} $METHOD $CONTAMINATION