module GroupAD

using DrWatson
using Statistics
using Mill
using Random
using StatsBase
using DelimitedFiles
using FileIO
using BSON
using DataDeps
using Distributions

include("data.jl")
include("mnist_data.jl")
include("experiments.jl")
include("models/Models.jl")

end #module
