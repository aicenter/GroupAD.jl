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

"""
    best_model_files(best_models, modelname)

Given a DataFrame of best models and their parameters, returns the file names for the models.
"""
function best_model_files(best_models, modelname)
    mpath = GroupAD.Evaluation.collect_models(datadir("experiments", "contamination-0.0", modelname, "MNIST", "leave-one-in", "class_index=1", "seed=1"))[1]
    mdata = load(mpath)
    mpars = mdata["parameters"]
    pr = best_models[:, [keys(mpars)...]]
    params = map(x -> Dict(names(x) .=> map(i -> x[i], 1:ncol(pr))), eachrow(pr))
    files = map(x -> savename("model", x, "bson", digits=5), params)
    return files
end

"""
    mnist_paths(modelname, method, class, files)

Joins the files paths with the datapath and returns the file paths for all models over the possible seeds.
"""
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

"""
    collect_mnist_models(modelname, method)

Collects the results, finds the best model for each class and saves a Dictionary
of models given the anomaly class index.
"""
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
    wsave(datadir("results", "MNIST", method, "$(modelname).bson"), models)
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

# context
class = 1
data = GroupAD.load_mnist_point_cloud(;anomaly_class_ind=class)
X = cat(data[:normal], data[:anomaly]);
dt, _ = unpack_mill((X, []));
labels = vcat(data[:l_normal], data[:l_anomaly]);

function mean_context(m::NeuralStatistician, x::AbstractArray)
    # instance network
    v = m.instance_encoder(x)
    p = mean(v, dims=2)

    # sample latent for context
    c = mean(m.encoder_c, p)
end

using GroupAD.Models: PoolModel
function pool_context(m::PoolModel, x::AbstractArray)
    v = m.prepool_net(x)
    # pooling
    p = m.poolf(v)
    # post-pool
    p_post = m.postpool_net(p)
end

idx = sample(1:70000, 5000, replace=false)
d = dt[idx]
l = Int.(labels[idx])

# statistician
model = models[Symbol(class)]
C = hcat(map(x -> mean_context(model, x), d)...)

p = scatter(C[1,:], C[2,:], color=l)
wsave(plotsdir("context", "context_in_class=$(class-1).png"), p)

using UMAP
# PoolModel
for class in 1:10
    modelname = "PoolModel"
    models = load(datadir("results", "MNIST", method, "models", "$(modelname).bson"))
    model = models[Symbol(class)]
    C = hcat(map(x -> pool_context(model, x), d)...)

    if size(C,1) > 2
        emb = umap(C, 2)
    else
        emb = C
    end

    nix = l .!= class-1
    aix = l .== class-1

    p = scatter(emb[1,nix], emb[2,nix], label="normal")
    p = scatter!(emb[1,aix], emb[2,aix], label="anomalous")
    wsave(plotsdir("context", modelname, "in-class=$(class-1).png"), p)
end

for class in 1:10
    modelname = "PoolModel"
    models = load(datadir("results", "MNIST", method, "models", "$(modelname).bson"))
    model = models[Symbol(class)]
    C = hcat(map(x -> pool_context(model, x), d)...)

    nix = l .!= class-1
    aix = l .== class-1

    p = scatter(C[1,nix], C[2,nix], C[3,nix], label="normal")
    p = scatter!(C[1,aix], C[2,aix], C[3,aix], label="anomalous")
    wsave(plotsdir("context", modelname, "in-3D_class=$(class-1).png"), p)
end