using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics

using Plots
using StatsPlots
ENV["GKSwstype"] = "100"

include(scriptsdir("evaluation", "workflow.jl"))
include(scriptsdir("evaluation", "downsampling_src.jl"))

"""
    find_best_model_scores(modelname::String, dataset::String, method::String, class_ind::Int; metric=:val_AUC)

Returns the rows of a results dataframe with the best result based on chosen
metric for all scores calculated for the model.
"""
function find_best_model_scores(modelname::String, dataset::String, method::String, class_ind::Int; metric=:val_AUC)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset, method, "class_index=$(class_ind)")
    data = GroupAD.Evaluation.results_dataframe(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
    params = point[:parameters]

    g_score = groupby(data, :score)
    g = map(x -> groupby(x, [keys(params)...]), g_score)
    un = unique(vcat(map(x -> unique(map(y -> size(y), x)), g)...))

    if length(un) != 1
        idx = findall.(x -> size(x,1) > 5, g)
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing $(sum(length.(g)) - sum(length.(idx))) groups out of $(sum(length.(g))) with less than 6 seeds."
        g = map(i -> g[i][idx[i]], 1:length(g))
    end

    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]
    cdf = map(y -> combine(y, map(x -> x => mean, metricsnames)), g)
    cdf_sorted = map(x -> sort(x, :val_AUC_mean, rev=true), cdf)
    best_models = vcat(map(x -> DataFrame(x[1,:]), cdf_sorted)...)
end

function best_model_files_scores(best_models)
    pr = best_models[:, Not([:val_AUC_mean, :val_AUPRC_mean, :test_AUC_mean, :test_AUPRC_mean])]
    params = map(x -> Dict(names(x) .=> values(x)), eachrow(pr))
    files = map(x -> savename(x, "bson", digits=5), params)
    return files
end

function best_model_files_models(best_models)
    mpath = GroupAD.Evaluation.collect_models(datadir("experiments", "contamination-0.0", modelname, dataset, method, "class_index=$(class_ind)"))[1]
    mdata = load(mpath)
    mpars = mdata["parameters"]
    pr = best_models[:, [keys(mpars)...]]
    params = map(x -> Dict(names(x) .=> values(x)), eachrow(pr))
    files = map(x -> savename("model", x, "bson", digits=5), params)
    return files
end

function mnist_paths(modelname, dataset, method, class_ind, files)
    paths = []
    for f in files
        seed_paths = String[]
        for seed in 1:10
            folder = datadir("experiments", "contamination-0.0", modelname, dataset, method, "class_index=$(class_ind)", "seed=$seed")
            path = joinpath(folder, f)
            push!(seed_paths, path)
        end
        push!(paths, seed_paths)
    end
    return paths
end

function evaluate_model_at_downsampled(modelname, dataset, method; classes=1:10)
    # load and find the best models for each score
    models = map(c -> find_best_model_scores(modelname, dataset, method, c), classes)
    files = best_model_files_models.(models)
    modelpaths = map(c -> mnist_paths(modelname, dataset, method, c, files[c]), classes)
    paths_unloaded = vcat(vcat(modelpaths...)...)
    r1 = map(x -> match(r"(.*\/seed=[0-9]{1,2})(\/.*)", x).captures[1], paths_unloaded)

    mdf = unique(DataFrame(
        :path => paths_unloaded,
        :group => r1
    ))

    gdf = groupby(mdf, :group)
    arr = map(x -> x[:, :path], gdf)

    for ar in arr
        evaluate_at_downsampled(ar)
    end
end

# for testing purposes
modelname = "vae_basic"
dataset = "MNIST"
method = "leave-one-in"

# evaluate and save new scores
evaluate_model_at_downsampled("vae_basic", "MNIST", "leave-one-out")