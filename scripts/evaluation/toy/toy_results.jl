using DrWatson
@quickactivate
using GroupAD
using GroupAD.Evaluation
using DataFrames
using Statistics
using EvalMetrics

using Plots
using StatsPlots
ENV["GKSwstype"] = "100"

include(scriptsdir("evaluation", "workflow.jl"))
include(srcdir("evaluation", "utils.jl"))
include(scriptsdir("evaluation", "toy", "workflow.jl"))

modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel", "MGMM"]
modelscores = [:distance, :score, :type, :type, :type, :score]

# for one model
dataset = "toy"
modelname = modelnames[1]

# create results collection
toy_results_collection = Dict()
for (modelname, group) in map((x, y) -> (x, y), modelnames, modelscores)
    df = vcat(map(x -> find_best_model_scores(modelname, dataset, x; groupsymbol=group), 1:3)...)
    push!(toy_results_collection, modelname => df)
end
for (modelname, group) in map((x, y) -> (x, y), modelnames, modelscores)
    dfs = DataFrame[]
    for scenario in 1:3
        folder = datadir("experiments", "contamination-0.0", modelname, dataset, "scenario=$scenario")
        df = find_best_model(folder, group)
        push!(dfs, df)
    end
    vcat(dfs...)
    push!(toy_results_collection, modelname => df)
end
# save it
safesave(datadir("dataframes", "toy_results_collection.bson"), toy_results_collection)
toy_results_names_scores = Dict(map((x, y) -> x => y, modelnames, modelscores))
safesave(datadir("dataframes", "toy_results_names_scores.bson"), toy_results_names_scores)

# load results collection
toy_results_collection = load(datadir("results/toy", "toy_results_collection.bson"))
toy_results_names_scores = load(datadir("results/toy", "toy_results_names_scores.bson"))


### BARPLOTS
# for already created dataframes
for (modelname, group) in pairs(toy_results_names_scores)
    df = toy_results_collection[modelname]
    groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=group, value=:test_AUC_mean, groupnamefull=false)

    groupedbar(
        groupnames, M, labels=labels, size=(800,400), color_palette=:tab20,
        ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
        left_margin=5Plots.mm, bottom_margin=5Plots.mm, ylims=(0,1)
    )
    savefig(plotsdir("toy", "groupedbar_$modelname.png"))
end

# calculate the dataframes 
for (modelname, group) in map((x, y) -> (x, y), modelnames, modelscores)
    df = vcat(map(x -> find_best_model_scores(modelname, dataset, x; groupsymbol=group), 1:3)...)
    groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=group, value=:test_AUC_mean, groupnamefull=false)

    groupedbar(
        groupnames, M, labels=labels, size=(800,400), color_palette=:tab20,
        ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
        left_margin=5Plots.mm, bottom_margin=5Plots.mm, ylims=(0,1)
    )
    savefig(plotsdir("toy", "groupedbar_$modelname.png"))
end

### ALL MODELS TOGETHER
# this is not formated the best
df_all = DataFrame[]
for (modelname, group) in map((m, s) -> (m, s), modelnames, modelscores)
    df = sort(toy_results_collection[modelname], :val_AUC_mean, rev=true)
    df[!, :modelname] = repeat([modelname], nrow(df))
    best = vcat(map(x -> DataFrame(x[1,:]), groupby(df, :scenario))...)
    push!(df_all, best)
end
df_all = vcat(df_all..., cols=:union)
sort!(df_all, :scenario)

# better formating
groupnames, M, labels = groupedbar_matrix(df_all, group=:scenario, cols=:modelname, value=:test_AUC_mean, groupnamefull=false)
modellabels = ["kNNagg" "VAEagg" "VAE" "NS" "PoolModel" "MGMM"]
groupedbar(
    groupnames, M, labels=modellabels, size=(800,400), color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Model", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_all_models.png"))

### for individual models by hand
modelname = "knn_basic"
df = vcat(map(x -> find_best_model_scores("knn_basic", dataset, x; groupsymbol=:distance), 1:3)...)
df = vcat(map(x -> find_best_model_scores("vae_instance", dataset, x; groupsymbol=:type), 1:3)...)
modelname = "PoolModel"
df = vcat(map(x -> find_best_model_scores("PoolModel", dataset, x; groupsymbol=:type), 1:3)...)
modelname = "statistician"
scorelabels = [""]
df = vcat(map(x -> find_best_model_scores("statistician", dataset, x; groupsymbol=:type), 1:3)...)
toy_results_collection[modelname] = df

groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:type, value=:test_AUC_mean, groupnamefull=false)

groupedbar(
    groupnames, M, labels=labels, size=(800,400), color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$modelname.png"))

######################################
### Formating of individual models ###
######################################

# for NS and VAE
modelname = "vae_instance"
modelname = "statistician"
df = toy_results_collection[modelname]
groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:type, value=:test_AUC_mean, groupnamefull=false)
idx = [8, 5, 4, 1, 7, 6, 3, 2, 11, 9, 10]
new_labels = labels[idx]
new_labels = ["sum" "mean" "max" "logU" "Po" "Po + logU" "LN" "LN + logU" "Chamfer" "MMD-G" "MMD-IMQ"]
MM = M[:, idx]

groupedbar(
    groupnames, MM, labels=new_labels, size=(800,400), color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$modelname.png"))

# for PoolModel
modelname = "PoolModel"
df = toy_results_collection[modelname]
groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:type, value=:test_AUC_mean, groupnamefull=false)
idx = [3,1,2]
new_labels = labels[idx]
new_labels = ["Chamfer" "MMD-G" "MMD-IMQ"]
MM = M[:, idx]

groupedbar(
    groupnames, MM, labels=new_labels, color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$modelname.png"))

# for knn model
modelname = "knn_basic"
df = toy_results_collection[modelname]
groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:distance, value=:test_AUC_mean, groupnamefull=false)

groupedbar(
    groupnames, MM, labels=labels, color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$modelname.png"))

# for vae_basic
modelname = "vae_basic"
df = toy_results_collection[modelname]
groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:score, value=:test_AUC_mean, groupnamefull=false)
idx = [3,1,2]
new_labels = labels[idx]
new_labels = ["rec" "rec-sampled" "rec-mean"]
MM = M[:, idx]

groupedbar(
    groupnames, MM, labels=new_labels, color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$modelname.png"))

# for MGMM
modelname = "MGMM"
df = toy_results_collection[modelname]
groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:score, value=:test_AUC_mean, groupnamefull=false)
idx = [1,3,2]
new_labels = labels[idx]
new_labels = ["point" "topic" "point + topic"]
MM = M[:, idx]

groupedbar(
    groupnames, MM, labels=new_labels, color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$modelname.png"))

# PoolModel on its own


# for PoolModel
modelname = "PoolModel"
df = vcat(map(x -> find_best_model_scores("PoolModel", dataset, x; groupsymbol=:poolf), 1:3)...)
groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:poolf, value=:test_AUC_mean, groupnamefull=false)
idx = [6,5,1,2,4,3]
new_labels = labels[idx]
new_labels = ["mean" "max" "meanmax" "meanmax + card" "sumstat" "sumstat + card"]
MM = M[:, idx]

groupedbar(
    groupnames, MM, labels=new_labels, color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$(modelname)_poolf.png"))