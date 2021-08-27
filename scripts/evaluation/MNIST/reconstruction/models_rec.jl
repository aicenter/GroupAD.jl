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

using GenerativeModels
using Distributions, DistributionsAD, ConditionalDists
using ValueHistories
using Flux

include(scriptsdir("evaluation", "MIL", "workflow.jl"))
include(scriptsdir("plotting", "mnist.jl"))
scatter = Plots.scatter
scatter! = Plots.scatter!

using GroupAD.Models: unpack_mill
using GroupAD.Models: reconstruct
import GroupAD.Models: reconstruct
reconstruct(model::NeuralStatistician, x) = GroupAD.Models.reconstruct_input(model, x)

# parameters
modelname = "vae_instance"
method = "leave-one-in"
class = 1

# load data
data = GroupAD.load_data("MNIST", method=method, anomaly_class_ind=class)
tr_x, tr_l = unpack_mill(data[1])
val_x, val_l = unpack_mill(data[2])
test_x, test_l = unpack_mill(data[3])



function best_model_files(best_models, modelname)
    mpath = GroupAD.Evaluation.collect_models(datadir("experiments", "contamination-0.0", modelname, "MNIST", "leave-one-in", "class_index=1", "seed=1"))[1]
    mdata = load(mpath)
    mpars = mdata["parameters"]
    pr = best_models[:, [keys(mpars)...]]
    params = map(x -> Dict(names(x) .=> map(i -> x[i], 1:ncol(pr))), eachrow(pr))
    files = map(x -> savename("model", x, "bson", digits=5), params)
    return files
end

function mnist_paths(modelname, method, class, files)
    paths = []
    for f in files
        seed_paths = String[]
        for seed in 1:10
            folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$(class)", "seed=$seed")
            path = joinpath(folder, f)
            push!(seed_paths, path)
        end
        push!(paths, seed_paths)
    end
    if length(paths) == 1
        return paths[1]
    else
        return paths
    end
end

function collect_mnist_models(modelname, method)
    models = Dict()

    for class in 1:10
        folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$class")
        b, best_seed = find_best_model(folder; save_best_seed=true)
        files = best_model_files(b, modelname)
        paths = mnist_paths(modelname, method, class, files)
        model = load(paths[best_seed])["model"]
        push!(models, Symbol(class) => model)
    end
    wsave(datadir("results", "MNIST", method, "models", "$(modelname).bson"), models)
end


##############################################
################ leave-one-in ################
##############################################

modelname = "vae_instance"
modelname = "statistician"
modelname = "PoolModel"
method = "leave-one-in"
#collect_mnist_models(modelname, method)
models = load(datadir("results", "MNIST", method, "models", "$(modelname).bson"))

for class in 1:10
    # load data
    data = GroupAD.load_data("MNIST", method=method, anomaly_class_ind=class)
    test_x, _ = unpack_mill(data[3])

    # load model
    model = models[Symbol(class)]

    # plot row of normal and anomalous digits and their reconstruction
    p = plot_na(test_x, model; an_color=2, k=5, layout=(1,2))
    wsave(
        plotsdir(
            "MNIST_reconstruction",
            method, modelname,
            "reconstruction_class=$(class-1).png"
        ), p)
end

###############################################
################ leave-one-out ################
###############################################

modelname = "vae_instance"
method = "leave-one-out"
#collect_mnist_models(modelname, method)
vae_models_out = load(datadir("results", "MNIST", method, "models", "$(modelname).bson"))

for class in 1:10
    # load data
    data = GroupAD.load_data("MNIST", method=method, anomaly_class_ind=class)
    test_x, _ = unpack_mill(data[3])

    # load model
    model = vae_models_out[Symbol(class)]

    # plot row of normal and anomalous digits and their reconstruction
    p = plot_na(test_x, model; an_color=2, k=5, layout=(1,2))
    wsave(
        plotsdir(
            "MNIST_reconstruction",
            method, modelname,
            "reconstruction_class=$(class-1).png"
        ), p)
end