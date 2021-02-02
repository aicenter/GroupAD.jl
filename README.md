# GroupAD.jl
Benchmarking of Generative Anomaly Detection for Multiple Instance Learning problems. Inspired by [GenerativeAD.jl](https://github.com/aicenter/GenerativeAD.jl).

## Installation

1. Clone this repo somewhere.
2. Run Julia in the cloned dir.
```bash
cd path/to/repo/GroupAD.jl
julia --project
```
3. Install all packages and download datasets.
```julia
]instantiate
using GroupAD
data = GroupAD.load_data("Fox")
# the last line should ask for permission to download datasets
```
4. Running a single experiment of VAE with 5-fold crossvalidation on the Tiger dataset.
```bash
cd scripts/experiments
julia vae_basic.jl 5 Tiger
```
5. You can quickly evaluate the results using this recursive script.
```bash
julia GroupAD.jl/scripts/evaluate_performance_single.jl path/to/results
```

## Running experiments on the RCI cluster

0. First, load Julia and Python modules.
```bash
ml Julia
ml Python
```
1. Install the package somewhere on the RCI cluster.
2. Then the experiments can be run via `slurm`. This will run 20 experiments with the basic VAE model, each with 5 crossvalidation repetitions on all datasets in the text file with 10 parallel processes for each dataset.
```bash
cd GroupAD.jl/scripts/experiments
./run_parallel.sh vae_basic 20 5 10 datasets_mill.txt
```
