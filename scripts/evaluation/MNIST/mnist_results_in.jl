using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics
using BSON

using Plots
using StatsPlots
ENV["GKSwstype"] = "100"

include(scriptsdir("evaluation", "MIL", "workflow.jl"))

####################
### leave-one-in ###
####################

# first empty Dictionary
#mnist_results_in = Dict()
mnist_results_in = load(datadir("dataframes", "mnist_results_in.bson"))

modelname = "knn_basic"
modelname = "vae_basic"
modelname = "vae_instance"
modelname = "statistician"
modelname = "PoolModel"
method = "leave-one-in"
class = 10
folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")

results = DataFrame[]
for class in 1:10
    folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")
    #df = find_best_model(folder, [:aggregation, :distance])
    df = find_best_model(folder) |> DataFrame
    push!(results, df)
end
rdf = vcat(results...)
push!(mnist_results_in, modelname => rdf)
save(datadir("dataframes", "mnist_results_in.bson"), mnist_results_in)

# add :model columns
modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel"]
model_names = ["kNNagg", "VAEagg", "VAE", "NS", "PoolModel"]
knn_basic, vae_basic, vae_instance, statistician, poolmodel = map(m-> insertcols!(mnist_results_in[m], :model => m), modelnames)

df_all = vcat(knn_basic, vae_basic, vae_instance, statistician, poolmodel, cols=:union)
df_red = df_all[:, [:model, :class, :test_AUC_mean]]
#sort!(df_red, [:class, :model])
groupnames, M, labels = groupedbar_matrix(df_red, group=:class, cols=:model, value=:test_AUC_mean)
idx = [2,4,5,3,1]
vcat(hcat(labels[idx]...), hcat(model_names...))

mnist_barplots(
    df_red, "all_models-in", model_names; ind = idx, 
    group=:class, cols=:model, value=:test_AUC_mean,
    w1=0.8, w2=0.85,legend_title="Model"
)

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, ylims=(0,1),
    ylabel="test AUC", xlabel="digit", legend=:outerright,
)
savefig(plotsdir("MNIST", "groupedbar_leave-in.png"))


########################
### groupedby scores ###
########################
### leave-one-in #######
########################

#mnist_results_in_scores = Dict()
mnist_results_in_scores = load(datadir("dataframes", "mnist_results_in_scores.bson"))

modelname = "knn_basic"
modelname = "vae_basic"
modelname = "vae_instance"
modelname = "statistician"
modelname = "PoolModel"
method = "leave-one-in"
class = 10
folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")

results = DataFrame[]
for class in 1:10
    folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")
    df = find_best_model(folder, :poolf)
    #df = find_best_model(folder) |> DataFrame
    push!(results, df)
end
rdf = vcat(results...)
push!(mnist_results_in_scores, modelname => rdf)
save(datadir("dataframes", "mnist_results_in_scores.bson"), mnist_results_in_scores)

# groupedbarplot for :aggregation, or :score, :type etc.
# kNN
modelname = "knn_basic"
knn_basic = mnist_results_in_scores[modelname]
g = groupby(sort(knn_basic, :val_AUC_mean, rev=true), [:class,:aggregation])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:aggregation, value=:test_AUC_mean)
groupnames

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, legend=:bottomright,
    ylims=(0,1), ylabel="test AUC", xlabel="digit"
)
savefig(plotsdir("MNIST", "png", "$(modelname)_agg-in.png"))
savefig(plotsdir("MNIST", "pdf", "$(modelname)_agg-in.pdf"))

# VAEagg
modelname = "vae_basic"
vae_basic = mnist_results_in_scores[modelname]
g = groupby(sort(vae_basic, :val_AUC_mean, rev=true), [:class,:aggregation])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:aggregation, value=:test_AUC_mean)

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, legend=:bottomright,
    ylims=(0,1), ylabel="test AUC", xlabel="digit"
)
savefig(plotsdir("MNIST", "png", "$(modelname)_agg-in.png"))
savefig(plotsdir("MNIST", "pdf", "$(modelname)_agg-in.pdf"))


# VAE
modelname = "vae_instance"
vae_instance = mnist_results_in_scores[modelname]
g = groupby(sort(vae_instance, :val_AUC_mean, rev=true), [:class,:type])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:type, value=:test_AUC_mean)

idx = [11,8,7,4,5,6,9,10,1,2,3]
new_labels = ["sum" "mean" "maximum" "logU" "LN" "LN + logU" "Po" "Po + logU" "MMD-G" "MMD-IMQ" "Chamfer"]
groupnames
vcat(hcat(labels[idx]...), new_labels)

mnist_barplots(
    gdf, "vae-in", new_labels; ind = idx, 
    group=:class, cols=:type, value=:test_AUC_mean,
    w1=0.8, w2=0.85
)

# NS
modelname = "statistician"
statistician = mnist_results_in_scores[modelname]
g = groupby(sort(statistician, :val_AUC_mean, rev=true), [:class,:type])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:type, value=:test_AUC_mean)
groupnames

idx = [11,8,7,4,5,6,9,10,1,2,3]
new_labels = ["sum" "mean" "maximum" "logU" "LN" "LN + logU" "Po" "Po + logU" "MMD-G" "MMD-IMQ" "Chamfer"]
vcat(hcat(labels[idx]...), new_labels)

mnist_barplots(
    gdf, "statistician-in", new_labels; ind = idx, 
    group=:class, cols=:type, value=:test_AUC_mean,
    w1=0.8, w2=0.85
)

# PoolModel
modelname = "PoolModel"
poolmodel = mnist_results_in_scores[modelname]
g = groupby(sort(poolmodel, :val_AUC_mean, rev=true), [:class,:poolf])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:poolf, value=:test_AUC_mean)
hcat(groupnames...)

new_labels = ["maximum" "mean" "meanmax" "meanmax + card" "sumstat" "sumstat + card"]
vcat(labels, new_labels)

mnist_barplots(
    gdf, "poolmodel-in", new_labels; legend_title="Pool function",
    group=:class, cols=:poolf, value=:test_AUC_mean,
    w1=0.8, w2=0.85
)