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
# using PyCall

export GenerativeModels

include("data.jl")
include("toy.jl")
include("experiments.jl")
include("experimental_loops.jl")
include("experiments_point_cloud.jl")
include("ipmeasures/IPMeasures.jl")
include("generative_models/GenerativeModels.jl")
include("models/Models.jl")
#include("evaluation/Evaluation.jl")

end #module
