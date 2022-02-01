module Evaluation

using DrWatson
using Distributions
using Statistics
using EvalMetrics
using DataFrames

export results_dataframe, find_best_model
export mill_results
export groupedbar_matrix, mill_barplots, mnist_barplots

include("utils.jl")
include("plotting.jl")

end # module