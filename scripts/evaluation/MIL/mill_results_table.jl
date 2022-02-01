"""
This script is very untidy and could be optimized.
"""

using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics
using BSON

# Milldata sets names
mill_datasets = [
    "BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2",
    "Mutagenesis1", "Mutagenesis2", "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein",
    "Tiger", "UCSBBreastCancer", "Web1", "Web2", "Web3", "Web4", "WinterWren"
]

# load results dataframes
modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel", "MGMM"]
mill_results_collection = load(datadir("results", "MIL", "mill_results_collection.bson"))
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map(key -> mill_results_collection[key], modelnames)
modelvec = [knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm]
# add modelname
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map((d, m) -> insertcols!(d, :model => m), modelvec, modelnames)
df = vcat(knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm, cols=:union)

model_names = ["kNNagg", "VAEagg", "VAE", "NS", "PoolModel", "MGMM"]

df_red = df[:, [:dataset, :model, :test_AUC_mean]]
sort!(df_red, :dataset)

g = groupby(df_red, :dataset)
g = map(x -> rename(x, :test_AUC_mean => x[1,:dataset]), g)
g = hcat(map(x -> x[:, 3], g)...)
df1 = DataFrame(g')
rename!(df1, model_names)
df1[:, :dataset] = mill_datasets
df_new = df1[:, [7,1,2,3,4,5,6]]

avg = map(x -> typeof(x) == Array{Float64,1} ? mean(x) : "Average", eachcol(df_new))
#avg_rank maybe do it if there is time
push!(df_new, avg)

using PrettyTables

l_max = LatexHighlighter(
    (data, i, j) -> (data[i,j] == maximum(df_new[i, 2:7])) && typeof(data[i,j])!==String,
    ["textbf", "textcolor{blue}"]
)
l_min = LatexHighlighter(
    (data, i, j) -> (data[i,j] == minimum(df_new[i, 2:7])) && typeof(data[i,j])!==String,
    ["textcolor{red}"]
)

t = pretty_table(
    df_new,
    highlighters = (l_max, l_min),
    formatters = ft_printf("%5.3f"),
    backend=:latex, tf=tf_latex_booktabs, nosubheader=true
)