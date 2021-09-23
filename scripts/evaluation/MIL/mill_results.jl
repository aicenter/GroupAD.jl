using DrWatson
@quickactivate
using GroupAD
using GroupAD.Evaluation
using DataFrames
using Statistics
using EvalMetrics
using BSON

using Plots
using StatsPlots
ENV["GKSwstype"] = "100"

#include(scriptsdir("evaluation", "MIL", "workflow.jl"))

mill_datasets = [
    "BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2",
    "Mutagenesis1", "Mutagenesis2", "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein",
    "Tiger", "UCSBBreastCancer", "Web1", "Web2", "Web3", "Web4", "WinterWren"
]
mill_names = [
    "BrownCreeper", "CorelAfrican", "CorelBeach", "Elephant", "Fox", "Musk1", "Musk2",
    "Mut1", "Mut2", "News1", "News2", "News3", "Protein",
    "Tiger", "UCSB-BC", "Web1", "Web2", "Web3", "Web4", "WinterWren"
]

modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel", "MGMM"]
modelscores = [:distance, :score, :type, :type, :type, :score]

# MIL results - finding the best model
# if calculated for the first time
mill_results_collection = Dict()
for modelname in modelnames
    df = mill_results(modelname, mill_datasets)
    push!(mill_results_collection, modelname => df)
end
save(datadir("dataframes", "mill_results_collection.bson"), mill_results_collection)
names_scores = Dict(:modelnames => modelnames, :modelscores => modelscores)
save(datadir("dataframes", "mill_names_scores.bson"),names_scores)

# MIL results finding the best model based on score function
# if calculated for the first time
mill_results_scores = Dict()
for (modelname, score) in map((x, y) -> (x, y), modelnames, modelscores)
    df = mill_results(modelname, mill_datasets, score)
    push!(mill_results_scores, modelname => df)
end
save(datadir("dataframes", "mill_results_scores.bson"), mill_results_scores)

# MIL results finding the best model based on score function
# aggregation for kNN and VAEagg
# if calculated for the first time
modelscores_agg = [:aggregation, :aggregation, :type, :type, :poolf, :score]
mill_results_scores_agg = Dict()
for (modelname, score) in map((x, y) -> (x, y), modelnames, modelscores_agg)
    df = mill_results(modelname, mill_datasets, score)
    push!(mill_results_scores_agg, modelname => df)
end
save(datadir("dataframes", "mill_results_scores_agg.bson"), mill_results_scores_agg)


# if already calculated, just load the data
mill_results_collection = load(datadir("results", "MIL", "mill_results_collection.bson"))
mill_results_scores_agg = load(datadir("results", "MIL", "mill_results_scores_agg.bson"))


###################################################
# create full dataframe for mill_results_collection
###################################################
# load the dataframe in corresponding variables
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map(key -> mill_results_collection[key], modelnames)
modelvec = [knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm]
# add modelname
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map((d, m) -> insertcols!(d, :model => m), modelvec, modelnames)
df = vcat(knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm, cols=:union)

# full barplot
new_labels = ["kNNagg" "VAEagg" "VAE" "NS" "PoolModel" "MGMM"]
groupnames, M, labels = groupedbar_matrix(df; group=:dataset, cols=:model, value=:test_AUC_mean)
idx = [3,5,6,4,2,1]
vcat(hcat(labels[idx]...), new_labels)

mill_barplots(df, "mill_models", new_labels; ind=idx, group=:dataset, cols=:model, value=:test_AUC_mean)

# results from mill_results_scores
# results for each model
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map(key -> mill_results_scores[key], modelnames)

# for VAE
modelname = "vae_instance"
df = vae_instance
idx = [11,8,7,4,5,6,9,10,1,2,3]
new_labels = ["sum" "mean" "maximum" "logU" "LN" "LN + logU" "Po" "Po + logU" "MMD-G" "MMD-IMQ" "Chamfer"]
groupnames, M, labels = groupedbar_matrix(df; group=:dataset, cols=:type, value=:test_AUC_mean)
vcat(hcat(labels[idx]...), new_labels)

mill_barplots(
    df, modelname, new_labels;
    group=:dataset, cols=:type, value=:test_AUC_mean, ind=idx, w1=0.55, w2=0.85,
    legend_title="Score"
    )

# for NS
modelname = "statistician"
df = statistician
idx = [11,8,7,4,5,6,9,10,1,2,3]
new_labels = ["sum" "mean" "max" "logU" "LN" "LN + logU" "Po" "Po + logU" "MMD-G" "MMD-IMQ" "Chamfer"]
groupnames, M, labels = groupedbar_matrix(df; group=:dataset, cols=:type, value=:test_AUC_mean)
vcat(hcat(labels[idx]...), new_labels)

mill_barplots(
    df, modelname, new_labels;
    group=:dataset, cols=:type, value=:test_AUC_mean, ind=idx, w1=0.55, w2=0.85,
    legend_title="Score"
    )

# for PoolModel
modelname = "PoolModel"
df = poolmodel
idx = [3,1,2]
new_labels = ["Chamfer" "MMD-G" "MMD-IMQ"]
groupnames, M, labels = groupedbar_matrix(df; group=:dataset, cols=:type, value=:test_AUC_mean)
vcat(hcat(labels[idx]...), new_labels)

mill_barplots(
    df, modelname, new_labels;
    group=:dataset, cols=:type, value=:test_AUC_mean, ind=idx,
    legend_title="Score"
    )

# for knn model
modelname = "knn_basic"
df = knn_basic
groupnames, M, labels = groupedbar_matrix(df; group=:dataset, cols=:distance, value=:test_AUC_mean)

mill_barplots(
    df, modelname;
    group=:dataset, cols=:distance, value=:test_AUC_mean,
    legend_title="Distance"
    )

# for vae_basic
modelname = "vae_basic"
df = vae_basic
idx = [1,3,2]
new_labels = ["rec" "rec-sampled" "rec-mean"]
groupnames, M, labels = groupedbar_matrix(df; group=:dataset, cols=:score, value=:test_AUC_mean)
vcat(hcat(labels[idx]...), new_labels)

mill_barplots(
    df, modelname, new_labels; ind = idx,
    group=:dataset, cols=:score, value=:test_AUC_mean,
    legend_title="Score"
    )

# for MGMM
modelname = "MGMM"
df = mgmm
new_labels = ["point" "topic" "point + topic"]
groupnames, M, labels = groupedbar_matrix(df; group=:dataset, cols=:score, value=:test_AUC_mean)
vcat(labels, new_labels)

mill_barplots(
    df, modelname, new_labels;
    group=:dataset, cols=:score, value=:test_AUC_mean,
    legend_title="Score"
    )

##############################
# for vae and knn aggregations
##############################

knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map(key -> mill_results_scores_agg[key], modelnames)

# for knn model
modelname = "knn_basic"
df = knn_basic
g = groupby(df, :dataset)
ix = findall(x -> size(x,1) != 3, g)
gix = g[ix]
vcat(gix[1], DataFrame(:aggregation => maximum), DataFrame(:aggregation => median), cols=:union)

df_new = vcat(
    DataFrame(:aggregation => "maximum", :dataset => "Web4", :test_AUC_mean => 0),    
    df,
    DataFrame(:aggregation => "median", :dataset => "Web4", :test_AUC_mean => 0), cols=:union)
sort!(df_new, :dataset)

mill_barplots(
    df_new, "$(modelname)_agg";
    group=:dataset, cols=:aggregation, value=:test_AUC_mean,
    legend_title="Aggregation"
    )

# for vae_basic
modelname = "vae_basic"
df = vae_basic

mill_barplots(
    df, "$(modelname)_agg";
    group=:dataset, cols=:aggregation, value=:test_AUC_mean,
    legend_title="Aggregation"
    )

    
# for PoolModel
modelname = "PoolModel"
df = poolmodel
#poolmodel = mill_results("PoolModel", mill_datasets, :poolf)
g = groupby(poolmodel, :dataset)
gs = map(x -> sort(x, :poolf), g)

ix = findall(x -> size(x,1) != 6, gs)
gix = gs[ix]
gix[1]

#using GroupAD.Models: sum_stat, sum_stat_card
df_new = vcat(
    vcat(gs...),
    DataFrame(:poolf => ["sum_stat", "sum_stat_card"], :dataset => ["Musk2", "Musk2"], :test_AUC_mean => [0,0]),
    DataFrame(:poolf => ["sum_stat", "sum_stat_card"], :dataset => ["Tiger", "Tiger"], :test_AUC_mean => [0,0]),
    cols=:union
)
sort!(df_new, :dataset)

new_labels = ["maximum", "mean", "meanmax", "meanmax + card", "sumstat", "sumstat + card"]

mill_barplots(
    df_new, "$(modelname)_poolf", new_labels;
    group=:dataset, cols=:poolf, value=:test_AUC_mean,
    legend_title="Pooling function"
)



    