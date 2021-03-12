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
using Mmap

include("data.jl")
include("experiments.jl")
include("experiments_mnist.jl")
include("models/Models.jl")

end #module
