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

#####################
### leave-one-out ###
#####################
mnist_results_out = Dict()
mnist_results_out = load(datadir("dataframes", "mnist_results_out.bson"))

modelname = "knn_basic"
modelname = "vae_basic"
modelname = "vae_instance"
method = "leave-one-out"
folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")

# create the dataframe and save it to dictionary
results = DataFrame[]
for class in 1:10
    folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")
    #df = find_best_model(folder, [:aggregation, :distance])
    df = find_best_model(folder) |> DataFrame
    push!(results, df)
end
rdf = vcat(results...)
push!(mnist_results_out, modelname => rdf)
save(datadir("dataframes", "mnist_results_out.bson"), mnist_results_out)

# add :model columns
modelnames = ["knn_basic", "vae_basic", "vae_instance"]
model_names = ["kNNagg", "VAEagg", "VAE"]
knn_basic, vae_basic, vae_instance = map(m-> insertcols!(mnist_results_out[m], :model => m), modelnames)

# groupedbarplot for more models
df_all = vcat(knn_basic, vae_basic, vae_instance, cols=:union)
df_red = df_all[:, [:model, :class, :test_AUC_mean]]
groupnames, M, labels = groupedbar_matrix(df_red, group=:class, cols=:model, value=:test_AUC_mean)
groupnames
idx = [1,2,3]
vcat(hcat(labels[idx]...), hcat(model_names...))

mnist_barplots(
    df_red, "all_models-out", model_names; ind = idx, 
    group=:class, cols=:model, value=:test_AUC_mean,
    w1=0.8, w2=0.85,legend_title="Model"
)


########################
### groupedby scores ###
########################
### leave-one-out ######
########################

#mnist_results_out_scores = Dict()
mnist_results_out_scores = load(datadir("dataframes", "mnist_results_out_scores.bson"))

modelname = "knn_basic"
modelname = "vae_basic"
modelname = "vae_instance"
method = "leave-one-out"

# calculating the results for a single model
results = DataFrame[]
for class in 1:10
    folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")
    df = find_best_model(folder, :type)
    #df = find_best_model(folder) |> DataFrame
    push!(results, df)
end
rdf = vcat(results...)
push!(mnist_results_out_scores, modelname => rdf)
save(datadir("dataframes", "mnist_results_out_scores.bson"), mnist_results_out_scores)


knn_basic, vae_basic, vae_instance = map(m-> mnist_results_out_scores[m], modelnames)
# groupedbarplot for :aggregation, or :score, :type etc.
# kNN
modelname = "knn_basic"
#knn_basic = mnist_results_out_scores["knn_basic"]
g = groupby(sort(knn_basic, :val_AUC_mean, rev=true), [:class,:aggregation])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:aggregation, value=:test_AUC_mean)
hcat(groupnames...)

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, legend=:bottomright,
    ylims=(0,1), ylabel="test AUC", xlabel="digit"
)
savefig(plotsdir("MNIST", "png", "$(modelname)_agg-out.png"))
savefig(plotsdir("MNIST", "pdf", "$(modelname)_agg-out.pdf"))

# VAEagg
modelname = "vae_basic"
#vae_basic = mnist_results_out_scores["vae_basic"]
g = groupby(sort(vae_basic, :val_AUC_mean, rev=true), [:class,:aggregation])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:aggregation, value=:test_AUC_mean)

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, legend=:bottomright,
    ylims=(0,1), ylabel="test AUC", xlabel="digit"
)
savefig(plotsdir("MNIST", "png", "$(modelname)_agg-out.png"))
savefig(plotsdir("MNIST", "pdf", "$(modelname)_agg-out.pdf"))

# VAE
modelname = "vae_instance"
#vae_instance = mnist_results_out_scores["vae_instance"]
g = groupby(sort(vae_instance, :val_AUC_mean, rev=true), [:class,:type])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = vcat(gm...)
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:type, value=:test_AUC_mean)

idx = [11,8,7,4,5,6,9,10,1,2,3]
new_labels = ["sum" "mean" "maximum" "logU" "LN" "LN + logU" "Po" "Po + logU" "MMD-G" "MMD-IMQ" "Chamfer"]
groupnames
vcat(hcat(labels[idx]...), new_labels)

mnist_barplots(
    gdf, "vae-out", new_labels; ind = idx, 
    group=:class, cols=:type, value=:test_AUC_mean,
    w1=0.8, w2=0.85
)