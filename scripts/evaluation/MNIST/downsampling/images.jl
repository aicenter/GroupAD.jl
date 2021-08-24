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

# load data    
data = load_data("MNIST")
d = unpack_mill(data[1])
dt = d[1]

# downsample train data
mnist_down_90 = map(x -> downsample(x, 0.9; seed = 2), dt)
mnist_down_75 = map(x -> downsample(x, 0.75; seed = 2), dt)
mnist_down_50 = map(x -> downsample(x, 0.5; seed = 2), dt)

# plot downsampled data in colums
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