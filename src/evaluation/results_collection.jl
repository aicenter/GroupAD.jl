using DrWatson
@quickactivate

using GroupAD
using GroupAD: Evaluation

# model names as they are saved
modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel"]
# pretty version of modelnames
model_names = ["kNNagg", "VAEagg", "VAE", "NS", "PoolModel"]

modelname = "vae_basic"
model_score = :aggregation
dataset = "BrownCreeper"

df = mill_results(modelname, dataset)