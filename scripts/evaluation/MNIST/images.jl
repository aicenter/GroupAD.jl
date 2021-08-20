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
function downsample(data::Mill.BagNode, labels; seed = nothing)
    d, _ = GroupAD.Models.unpack_mill((data, []))
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
    new_labels = vcat(l0,l90,l75,l50)
    if length(new) != n != length(new_labels)
        error("Error in length of new data.")
    end

    M = hcat(new...)
    (Mill.BagNode(ArrayNode(M), data.bags), new_labels)
end
function downsample(data::Tuple; seed = nothing)
    train, val, test = data
    tr = (downsample(train..., seed = seed))
    v = (downsample(val..., seed = seed))
    t = (downsample(test..., seed = seed))

    return (tr, v, t)
end
    
data = load_data("MNIST")
d = unpack_mill(data[1])
dt = d[1]

plot_numbers(10,dt)
savefig(plotsdir("MNIST", "numbers.png"))

mnist_down_90 = map(x -> downsample(x, 0.9; seed = 2), dt)
mnist_down_75 = map(x -> downsample(x, 0.75; seed = 2), dt)
mnist_down_50 = map(x -> downsample(x, 0.5; seed = 2), dt)

k = 50
d_title = ["100%" "" "" "" "" "90%" "" "" "" "" "75%" "" "" "" "" "50%" "" "" "" ""]
plot(
    plot_number_row(k, dt),
    plot_number_row(k, mnist_down_90),
    plot_number_row(k, mnist_down_75),
    plot_number_row(k, mnist_down_50),
    layout = (1,4), size=(600,600), title=d_title
)
savefig(plotsdir("MNIST", "downsampled.png"))