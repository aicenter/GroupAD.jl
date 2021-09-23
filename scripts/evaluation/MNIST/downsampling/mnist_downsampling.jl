using DrWatson
using GroupAD
using GroupAD: load_data
using GroupAD.Models: unpack_mill
using BSON
using DataFrames
using Latexify
using Mill
using ValueHistories

using Plots
using Plots: scatter, scatter!
using StatsPlots
ENV["GKSwstype"] = "100"
include(scriptsdir("plotting", "mnist.jl"))

using Flux
using DistributionsAD
using ConditionalDists
using GenerativeModels
using Random
using StatsBase
using Statistics


"""
    downsample(x::AbstractArray, ratio=0.5; seed = nothing)
    downsample(data::Mill.BagNode, labels; seed = nothing)
    downsample(data::Tuple; seed = nothing)

The `downsample` function is used to reduce cardinality of some bags in the data.
The input dataset is randomly divided into 4 parts of same length. First part is left
as it is. Bags in the other parts are donwsampled to 90%, 75% and 50% of instances.
"""
function downsample(x::AbstractArray, ratio=0.5; seed = nothing)
    # set seed
    (seed === nothing) ? nothing : Random.seed!(seed)

    n = size(x,2)
    nd = round(Int, n*ratio)
    idx = sample(1:n, nd)

    # reset seed
    (seed === nothing) ? nothing : Random.seed!()

    return x[:, idx]
end
function downsample(x::Mill.BagNode, labels; seed = nothing)
    d, _ = GroupAD.Models.unpack_mill((x, []))
    n = length(d)

    (seed === nothing) ? nothing : Random.seed!(seed)
    bag_idx = sample(1:n,n)
    indices = round.(Int, n ./ [4,3,2,1])

    down0 = d[bag_idx[1:indices[1]]]
    down90 = map(x -> downsample(x, 0.9; seed = seed), d[bag_idx[indices[1]+1 : indices[2]]])
    down75 = map(x -> downsample(x, 0.75; seed = seed), d[bag_idx[indices[2]+1 : indices[3]]])
    down50 = map(x -> downsample(x, 0.5; seed = seed), d[bag_idx[indices[3]+1 : indices[4]]])

    l0 = labels[bag_idx[1:indices[1]]]
    l90 = labels[bag_idx[indices[1]+1 : indices[2]]]
    l75 = labels[bag_idx[indices[2]+1 : indices[3]]]
    l50 = labels[bag_idx[indices[3]+1 : indices[4]]]

    new = vcat(down0, down90, down75, down50)
    sz = size.(new, 2)
    ids = vcat(0, map(i -> sum(sz[1:i]), 1:n))
    bagids = map(i -> ids[i]+1:ids[i+1], 1:n)

    new_labels = vcat(l0,l90,l75,l50)
    if length(new) != n != length(new_labels)
        error("Error in length of new data.")
    end

    M = hcat(new...)
    (Mill.BagNode(ArrayNode(M), bagids), new_labels)
end
function downsample(data::Tuple; seed = nothing)
    train, val, test = data
    tr = (downsample(train..., seed = seed))
    v = (downsample(val..., seed = seed))
    t = (downsample(test..., seed = seed))

    return (tr, v, t)
end

modelpath = "data/experiments/contamination-0.0/vae_basic/MNIST/leave-one-in/class_index=1/seed=1/model_activation=relu_aggregation=maximum_batchsize=32_class=1_hdim=4_init_seed=64636347_lr=0.0001_method=leave-one-in_nlayers=3_var=diagonal_zdim=1.bson"


"""
    evaluate_at_downsampled(modelpath::String; all_results = all_results)

This function takes the modelpath and calculates all scores for the model on downsampled MNIST dataset.
The function extracts all necessary parameters from the modelpath so nothing else is needed.

Steps:
1. Load the model.
2. Extract all parameters.
3. Load data given parameters, prepare (leave-one-in or leave-one-out downsampling), and downsample
    the instances in the bags.
4. Load results functions and send them to `experiment` or `experiment_bag` functions to calculate
    scores on downsampled data.

New result files are saved to the folder data/contamination-0.0/model/MNIST_downsampled/...
Therefore the evaluation of this dataset can be done easily with the functions for evaluation
already prepared, the only thing to change is the dataset="MNIST_downsampled" name.

Note: the calculation of scores for some models (vae_instance, statistician) might take a while because
of sampled likelihood function.
"""
function evaluate_at_downsampled(modelpath::String)
    # seed extraction from modelpath
    m = match(r"/contamination-0.0\/(.*)\/MNIST.*\/seed=([0-9]*)\/.*", modelpath)
    modelname = m.captures[1]
    seed = parse(Int, m.captures[2])

    # load model
    training_info = load(modelpath)
    model = training_info["model"]
    parameters = training_info["parameters"]
    method, class_ind = parameters[:method], parameters[:class]

    # this doesn't work because it is a Dict and not NamedTuple
    # save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = "MNIST"))
    # this should probably be enough
    save_entries = (model = model, modelname = modelname, seed = seed, dataset = "MNIST")

    # load data, prepare and downsample
    data = GroupAD.load_data("MNIST", anomaly_class_ind = class_ind, method = method, contamination = 0)
    if method == "leave-one-in"
        data = GroupAD.leave_one_in(data; seed=seed)
    elseif method == "leave-one-out"
        data = GroupAD.leave_one_out(data; seed=seed)
    else
        error("This evaluation only works on MNIST!")
    end
    ds = downsample(data; seed = seed)

    # any difference to path? like MNIST_downsampled?
    # _savepath = datadir("experiments/contamination-0.0", modelname, "MNIST", method, "class_index=$(class)/seed=$(seed)")
    _savepath = datadir("experiments/contamination-0.0", modelname, "MNIST_downsampled", method, "class_index=$(class_ind)/seed=$(seed)")

    # to do: add results (the anomalous functions)
    # this is results for vae_instance
    results = results_functions(modelname, model, parameters)

    for result in results
        if modelname in ["vae_instance", "statistician", "PoolModel"]
            experiment_bag(result..., ds, _savepath; save_entries...)
        else
            GroupAD.experiment(result..., ds, _savepath; save_entries...)
        end
    end
end

# save this as a contant NamedTuple
# choose from all_results based on modelname
# results = all_results[Symbol(modelname)]
function results_functions(modelname, model, parameters)
    if haskey(parameters, :aggregation)
        agf = eval(:($(Symbol(parameters[:aggregation]))))
    end
    all_results = (
        vae_basic = [
            (x -> GroupAD.Models.reconstruction_score(model,x,agf), 
                merge(parameters, (score = "reconstruction",))),
            (x -> GroupAD.Models.reconstruction_score_mean(model,x,agf), 
                merge(parameters, (score = "reconstruction-mean",))),
            (x -> GroupAD.Models.reconstruction_score(model,x,agf,100), 
                merge(parameters, (score = "reconstruction-sampled", L=100)))		
        ],
        vae_instance = [
            (x -> GroupAD.Models.likelihood(model,x), 
                merge(parameters, (score = "reconstruction",))),
            (x -> GroupAD.Models.mean_likelihood(model,x), 
                merge(parameters, (score = "reconstruction-mean",))),
            (x -> GroupAD.Models.likelihood(model,x,50), 
                merge(parameters, (score = "reconstruction-sampled", L=50))),
            (x -> GroupAD.Models.reconstruct(model,x), 
                merge(parameters, (score = "reconstructed_input",)))
        ],
        statistician = [
            (x -> GroupAD.Models.likelihood(model,x), 
                merge(parameters, (score = "reconstruction",))),
            (x -> GroupAD.Models.mean_likelihood(model,x), 
                merge(parameters, (score = "reconstruction-mean",))),
            (x -> GroupAD.Models.likelihood(model,x,50), 
                merge(parameters, (score = "reconstruction-sampled", L=50))),
            (x -> GroupAD.Models.reconstruct_input(model, x),
                merge(parameters, (score = "reconstructed_input",)))
        ],
        MGMM = [
            (x -> GroupAD.Models.topic_score(model,x), 
                merge(parameters, (score = "topic",))),
            (x -> GroupAD.Models.point_score(model,x), 
                merge(parameters, (score = "point",))),
            (x -> GroupAD.Models.MGMM_score(model,x), 
                merge(parameters, (score = "topic+point",)))
        ],
        PoolModel = [
            (x -> GroupAD.Models.reconstruct(model, x),
                merge(parameters, (score = "reconstructed_input",)))
        ]
    )
    return all_results[Symbol(modelname)]
end

"""
For given model and method, `collect_models(folder)` goes recursively through the given
folder and returns all the model paths in that folder.

Then, all it takes is to start a loop and calculate the scores for downsampled instances
with the function `evaluate_at_downsampled()`.
"""
modelname = "vae_basic"
method = "leave-one-in"
folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method)
model_paths = GroupAD.Evaluation.collect_models(folder)

evaluate_at_downsampled.(model_paths)