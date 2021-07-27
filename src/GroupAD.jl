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
using Distributions

include("data.jl")
include("toy.jl")
include("experiments.jl")
include("experiments_point_cloud.jl")
include("models/Models.jl")
include("evaluation/Evaluation.jl")

end #module
