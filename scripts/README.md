# Scripts

Every dataset type has its own folder for run scripts due to different setting for MIL, MNIST and toy datasets.

# Evaluation of results

This folder contains evaluation scripts.

## One-sample evaluation

To evaluate a single results file (or all results file in a folder), use `evaluate_performance_single.jl` script which computes AUC-ROC, AUC-PR, F score and others for given results file.

To find the best model for a particular dataset, use `evaluate_single_model.jl`. 

### Examples
```
# evaluation of vae_basic model on Fox dataset
$ julia scripts/evaluate_single_model.jl vae_basic Fox

# evaluation of knn_basic model on MNIST dataset, leave-one-in method, normal class 1
$ julia scripts/evaluate_single_model.jl knn_basic MNIST nothing 1 leave-one-in
```

To find the best model based on some criteria (aggregation, score, distance etc.), use the third argument, groupkey (should be `Symbol` type).

```
# evaluation of vae_basic model on Fox dataset based on aggregation
$ julia scripts/evaluate_single_model.jl vae_basic Fox aggregation

# evaluation of knn_basic model on MNIST dataset, leave-one-in method, normal class 1 based on distance
$ julia scripts/evaluate_single_model.jl knn_basic MNIST distance 1 leave-one-in
```

## MIL results

