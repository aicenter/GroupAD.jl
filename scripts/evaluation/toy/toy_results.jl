using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics

using Plots
using StatsPlots
#using PlotlyJS
ENV["GKSwstype"] = "100"

include(scriptsdir("evaluation", "workflow.jl"))
include(srcdir("evaluation", "utils.jl"))
include(scriptsdir("evaluation", "toy", "workflow.jl"))

modelnames = ["knn_basic", "vae_basic", "MGMM", "vae_instance", "statistician", "PoolModel"]
modelscores = [:distance, :score, :score, :type, :type, :type]

"""
    groupedbar_matrix(df::DataFrame; group::Symbol, cols::Symbol, value::Symbol, groupnamefull=true)

Create groupnames, matrix and labels for given dataframe.
"""
function groupedbar_matrix(df::DataFrame; group::Symbol, cols::Symbol, value::Symbol, groupnamefull=true)
    gdf = groupby(df, group)
    gdf_keys = keys(gdf)
    gdf_name = String(group)
    gdf_values = map(k -> values(k)[1], gdf_keys)
    if groupnamefull
        groupnames = map((name, value) -> "$name = $value", repeat([gdf_name], length(gdf_values)), gdf_values)
    else
        groupnames = map(value -> "$value", gdf_values)
    end
    colnames = gdf[1][:, cols]
    M = hcat(map(x -> x[:, value], gdf)...)'

    groupnames, M, String.(hcat(colnames...))
end
function groupedbar_matrix(df::DataFrame; group::Symbol, cols::Array{Symbol,1}, value::Symbol, groupnamefull=true)
    gdf = groupby(df, group)
    gdf_keys = keys(gdf)
    gdf_name = String(group)
    gdf_values = map(k -> values(k)[1], gdf_keys)
    if groupnamefull
        groupnames = map((name, value) -> "$name = $value", repeat([gdf_name], length(gdf_values)), gdf_values)
    else
        groupnames = map(value -> "$value", gdf_values)
    end
    _colnames = gdf[1][:, cols]
    colnames = map((x, y) -> "$x & $y", _colnames[:, 1], _colnames[:, 2])
    M = hcat(map(x -> x[:, value], gdf)...)'

    groupnames, M, String.(hcat(colnames...))
end

# for one model
dataset = "toy"
modelname = modelnames[1]

# create results collection
toy_results_collection = Dict()
for (modelname, group) in map((x, y) -> (x, y), modelnames, modelscores)
    df = vcat(map(x -> find_best_model_scores(modelname, dataset, x; groupsymbol=group), 1:3)...)
    push!(toy_results_collection, modelname => df)
end
# save it
safesave(datadir("dataframes", "toy_results_collection.bson"), toy_results_collection)
toy_results_names_scores = Dict(map((x, y) -> x => y, modelnames, modelscores))
safesave(datadir("dataframes", "toy_results_names_scores.bson"), toy_results_names_scores)

# load results collection
toy_results_collection = load(datadir("dataframes", "toy_results_collection.bson"))
toy_results_names_scores = load(datadir("dataframes", "toy_results_names_scores.bson"))


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
df_all = DataFrame[]
for (modelname, group) in pairs(toy_results_names_scores)
    df = sort(toy_results_collection[modelname], :val_AUC_mean, rev=true)
    df[!, :modelname] = repeat([modelname], nrow(df))
    best = vcat(map(x -> DataFrame(x[1,:]), groupby(df, :scenario))...)
    push!(df_all, best)
end
df_all = vcat(df_all..., cols=:union)

groupnames, M, labels = groupedbar_matrix(df_all, group=:scenario, cols=:modelname, value=:test_AUC_mean, groupnamefull=false)
groupedbar(
    groupnames, M, labels=labels, size=(800,400), color_palette=:tab20,
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
df = vcat(map(x -> find_best_model_scores("statistician", dataset, x; groupsymbol=:type), 1:3)...)
toy_results_collection[modelname] = df

groupnames, M, labels = groupedbar_matrix(df, group=:scenario, cols=:type, value=:test_AUC_mean, groupnamefull=false)
groupedbar(
    groupnames, M, labels=labels, size=(800,400), color_palette=:tab20,
    ylabel="test AUC", xlabel="scenario", legendtitle="Score", legend=:outerright,
    left_margin=5Plots.mm, bottom_margin=5Plots.mm
)
savefig(plotsdir("toy", "groupedbar_$modelname.png"))