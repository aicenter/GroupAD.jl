# Experiments - point clouds (currently only MNIST)

To run a model on the RCI cluster via slurm, run
```
./run_mnist.sh statistician 20 10 5 leave-one-in
```
for 20 repetitions of the experiment over 10 seeds with 5 concurrent tasks in the array job in "leave-one-in" setting. The dataset is currently fixed at MNIST since MNIST is the only point cloud dataset currently used.

All models have their own scripts to define parameters, the `fit()` function and calculated scores. All the parameters are passed to the `point_cloud_experimental_loop` which is different to `basic_experimental_loop` (for MIL datasets). The loop makes use of Julia multithreading feature and runs experiments in parallel over seeds. The recommended setting is 10 seeds and 10 CPU cores for each repetition of the experiment. (For different number of CPU cores, do not forget to change the run scripts, e.g. `statistician_run.sh`, and training time in julia scripts.)

## Training specifications

### VAE and Neural Statistician

All models are trained for 24 hours over 10 cores to ensure 24 training hours for a model over 10 seeds. The overall time is 48 hours to ensure that all scores are calculates. Still, sometimes the sampled likelihood calculations take too long. If the calculation time (of sampled likelihood) should take more than an hour, it is not computed.

### Other models - aggVAE, kNN

Models acting on aggregated bags do not need that much training time, it is reduced to 24 hours and 20G of memory.