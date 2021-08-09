# Experiments - MI problems (Multiple-Instance learning)

To run a model on the RCI cluster via slurm, run
```
./run_parallel_mill.sh statistician 20 10 5 datasets_mill.txt
```
for 20 repetitions of the experiment over 10 seeds with 5 concurrent tasks in the array job. This scripts loops over all MIL datasets in the text file `datasets_mill.txt`.

## Models
Models working on aggregated bags (VAE, kNN) get 24 hours of runtime, 23 hours for training over all seeds and 1 hours for score calculations.

Models working on bags (VAE, Neural Statistician) get 24 hours training time, 48 hours of total runtime to safely calculate all scores (sampled likelihood takes a bit more time).

## List of datasets
- BrownCreeper
- CorelAfrican
- CorelBeach
- Elephant
- Fox
- Musk1
- Musk2
- Mutagenesis1
- Mutagenesis2
- Newsgroups1
- Newsgroups2
- Newsgroups3
- Protein
- Tiger
- UCSBBreastCancer
- Web1
- Web2
- Web3
- Web4
- WinterWren