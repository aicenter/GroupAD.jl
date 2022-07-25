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
if occursin("Python/3.8.6-GCCcore-10.2.0", read(`which python`, String))
    using PyCall
end

export GenerativeModels

include("data.jl")
include("toy.jl")
include("experiments.jl")
include("experimental_loops.jl")
#include("experiments_point_cloud.jl")
include("ipmeasures/IPMeasures.jl")
include("generative_models/GenerativeModels.jl")
include("models/Models.jl")
#include("evaluation/Evaluation.jl")

const mill_datasets = [
    "BrownCreeper", "CorelAfrican", "CorelBeach", "Elephant", "Fox", "Musk1", "Musk2",
    "Mutagenesis1", "Mutagenesis2", "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein",
    "Tiger", "UCSBBreastCancer", "Web1", "Web2", "Web3", "Web4", "WinterWren"
]

export mill_datasets

end #module
