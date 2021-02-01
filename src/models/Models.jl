module Models

using NearestNeighbors
using StatsBase
using Statistics
using LinearAlgebra
using Mill

include("utils.jl")
include("aggregation.jl")
include("knn.jl")
include("vae.jl")

end # module