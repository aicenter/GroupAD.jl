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
mnist_results_in = Dict()

modelname = "knn_basic"
modelname = "vae_basic"
modelname = "vae_instance"
modelname = "statistician"
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
mnist_results_in = load(datadir("dataframes", "mnist_results_in.bson"))

bar(map(i -> "$i", 0:9), rdf[:, :test_AUC_mean], legend=:none,xlabel="digit", ylabel="AUC",ylims=(0,1))
savefig(plotsdir("MNIST", "$(modelname)_leave-in.png"))

knn_basic = insertcols!(mnist_results_in["knn_basic"], :model => "knn_basic")
knn_basic = mnist_results_in["knn_basic"]
vae_basic = insertcols!(mnist_results_in["vae_basic"], :model => "vae_basic")
vae_basic = mnist_results_in["vae_basic"]
vae_instance = insertcols!(mnist_results_in["vae_instance"], :model => "vae_instance")
vae_instance = mnist_results_in["vae_instance"]
statistician = insertcols!(mnist_results_in["statistician"], :model => "statistician")
statistician = mnist_results_in["statistician"]

knn_basic, vae_basic, vae_instance, statistician = map(modelname -> mnist_results_in[modelname], ["knn_basic", "vae_basic", "vae_instance", "statistician"])

df_all = vcat(knn_basic, vae_basic, vae_instance, statistician, cols=:union)
df_red = df_all[:, [:model, :class, :test_AUC_mean]]
sort!(df_red, [:class, :model])
groupnames, M, labels = groupedbar_matrix(df_red, group=:class, cols=:model, value=:test_AUC_mean)

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, ylims=(0,1),
    ylabel="test AUC", xlabel="digit", legend=:outerright
)
savefig(plotsdir("MNIST", "groupedbar_leave-in.png"))


########################
### groupedby scores ###
########################
### leave-one-in #######
########################

mnist_results_in_scores = Dict()

modelname = "knn_basic"
modelname = "vae_basic"
modelname = "vae_instance"
method = "leave-one-in"
class = 10
folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")

results = DataFrame[]
for class in 1:10
    folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")
    df = find_best_model(folder, :type)
    #df = find_best_model(folder) |> DataFrame
    push!(results, df)
end
rdf = vcat(results...)
push!(mnist_results_in_scores, modelname => rdf)
save(datadir("dataframes", "mnist_results_in_scores.bson"), mnist_results_in_scores)
mnist_results_in_scores = load(datadir("dataframes", "mnist_results_in_scores.bson"))

# groupedbarplot for :aggregation, or :score, :type etc.
# kNN
modelname = "knn_basic"
knn_basic = mnist_results_in_scores["knn_basic"]
g = groupby(sort(knn_basic, :val_AUC_mean, rev=true), [:class,:aggregation])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = sort(vcat(gm...), [:class, :aggregation])
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:aggregation, value=:test_AUC_mean)

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, legend=:bottomright,
    ylims=(0,1), ylabel="test AUC", xlabel="digit"
)
savefig(plotsdir("MNIST", "$(modelname)_groupedbar_leave-in.png"))

# VAEagg
modelname = "vae_basic"
vae_basic = mnist_results_in_scores["vae_basic"]
g = groupby(sort(vae_basic, :val_AUC_mean, rev=true), [:class,:aggregation])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = sort(vcat(gm...), [:class, :aggregation])
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:aggregation, value=:test_AUC_mean)

groupedbar(
    map(i -> "$i", 0:9), M, labels=labels, legend=:bottomright,
    ylims=(0,1), ylabel="test AUC", xlabel="digit"
)
savefig(plotsdir("MNIST", "$(modelname)_groupedbar_leave-in.png"))


# VAE
modelname = "vae_instance"
vae_instance = mnist_results_in_scores["vae_instance"]
g = groupby(sort(vae_instance, :val_AUC_mean, rev=true), [:class,:type])
gm = map(x -> DataFrame(x[1,:]), g)
gdf = sort(vcat(gm...), [:class, :type])
groupnames, M, labels = groupedbar_matrix(gdf, group=:class, cols=:type, value=:test_AUC_mean)

idx = [11,8,7,4,5,6,9,10,3,1,2]
new_labels = ["sum" "mean" "max" "logU" "Po" "Po + logU" "LN" "LN + logU" "Chamfer" "MMD-G" "MMD-IMQ"]

groupedbar(
    map(i -> "$i", 0:9), M[:, idx], labels=hcat(labels[idx]...), legend=:bottomright,
    ylims=(0,1), ylabel="test AUC", xlabel="digit", color_palette=:tab20,
    size=(800,400),
)
savefig(plotsdir("MNIST", "$(modelname)_groupedbar_leave-in2.png"))

mnist_barplots(
    gdf, "vae_in", new_labels; ind = idx, 
    group=:class, cols=:type, value=:test_AUC_mean,
    w1=0.8, w2=0.85
)