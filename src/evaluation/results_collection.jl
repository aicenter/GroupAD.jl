using DrWatson
@quickactivate
using GroupAD
using GroupAD.Evaluation

# model names as they are saved
modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel"]
# pretty version of modelnames
model_names = ["kNNagg", "VAEagg", "VAE", "NS", "PoolModel"]

###########
### MIL ###
###########

### single model, single dataset
modelname = "vae_basic"
modelscore = :aggregation
dataset = "BrownCreeper"

# only the best score
df = mill_results(modelname, [dataset])
safesave(datadir("results", "MIL", "models", "$(modelname)_$(dataset).bson"),Dict(:df => df))
# the best scores based on modelscore
df = mill_results(modelname, [dataset], modelscore)
safesave(datadir("results", "MIL", "models", "$(modelname)_$(dataset)_$(String(modelscore)).bson"),Dict(:df => df, :score => modelscore))

### single model, all datasets
modelname = "vae_basic"
modelscore = :aggregation
mill_datasets = [
    "BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2",
    "Mutagenesis1", "Mutagenesis2", "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein",
    "Tiger", "UCSBBreastCancer", "Web1", "Web2", "Web3", "Web4", "WinterWren"
]

# only the best score
df = mill_results(modelname, mill_datasets)
safesave(datadir("results", "MIL", "models", "$(modelname).bson"),Dict(:df => df))
# the best scores based on modelscore
df = mill_results(modelname, mill_datasets, modelscore)
safesave(datadir("results", "MIL", "models", "$(modelname)_$(String(modelscore)).bson"),Dict(:df => df, :score => modelscore))