#!/bin/bash
# This runs MNIST on slurm.
# USAGE EXAMPLE
#   ./run_parallel_mnist.sh statistician 50 10 2 leave-one-in 0
# Run from this folder only.
MODEL=$1 		# which model to run
NUM_SAMPLES=$2	# how many repetitions
MAX_SEED=$3		# how many folds over dataset
NUM_CONC=$4		# number of concurrent tasks in the array job
METHOD=$5	    # method: leave-one-in or leave-one-out
CONTAMINATION=$6

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi


# submit to slurm
sbatch \
--array=1-${NUM_SAMPLES}%${NUM_CONC} \
--output="${LOG_DIR}/MNIST_$METHOD-%A_%a.out" \
    ./${MODEL}.sh $MAX_SEED "MNIST" 10 $METHOD $CONTAMINATION

# for local testing    
# ./${MODEL}_run.sh $MAX_SEED "MNIST" $METHOD 10