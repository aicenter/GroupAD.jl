#!/bin/bash
# This runs MNIST on slurm.
# USAGE EXAMPLE
#   ./run_parallel_lhco.sh vae_basic 50 10 10 0
# Run from this folder only.
MODEL=$1 		# which model to run
NUM_SAMPLES=$2	# how many repetitions
MAX_SEED=$3		# how many folds over dataset
NUM_CONC=$4		# number of concurrent tasks in the array job
CONTAMINATION=$5

LOG_DIR="${HOME}/logs/lhco/${MODEL}"
echo "$LOG_DIR"

if [ ! -d "$LOG_DIR" ]; then
	mkdir -p  $LOG_DIR
fi

# submit to slurm
sbatch \
--array=1-${NUM_SAMPLES}%${NUM_CONC} \
--output="${LOG_DIR}/lhco-%A_%a.out" \
    ./${MODEL}.sh $MAX_SEED $CONTAMINATION