#!/bin/bash
#SBATCH --partition=cpulong
#SBATCH --time=36:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2
#SBATCH --mem=12G

MAX_SEED=$1
DATASET=$2
ANOMALY_CLASSES=$3
METHOD=$4
CONTAMINATION=$5

module load Julia/1.7.3-linux-x86_64
# module load Python/3.8

# julia --project --threads 5 ./vae_instance_chamfer.jl ${MAX_SEED} $DATASET ${ANOMALY_CLASSES} $METHOD $CONTAMINATION
julia --project ./vae_instance_chamfer.jl ${MAX_SEED} $DATASET ${ANOMALY_CLASSES} $METHOD $CONTAMINATION
