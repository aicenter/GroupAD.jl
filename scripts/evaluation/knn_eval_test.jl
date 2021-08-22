using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics

modelname = "knn_basic"
dataset = "BrownCreeper"
folder = datadir("experiments", "contamination-0.0", modelname, dataset)

data = GroupAD.Evaluation.results_dataframe(folder)

using BSON
point = load(GroupAD.Evaluation.collect_scores(folder)[1])
println(keys(point))
params = point[:parameters]
keys(params)

g = groupby(data, [keys(params)...])
un = unique(map(x -> size(x), g))
length(un)

metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]

cdf = combine(g, map(x -> x => mean, metricsnames))
sort!(cdf, :val_AUC_mean, rev=true)

best_model = c[1,:]

using Plots
ENV["GKSwstype"] = "100"

function category_to_int(x)
    categories = unique(x)
    n = length(categories)
    levs = ones(Int, length(x))

    for i in 2:n
        levs[x .== categories[i]] .= i
    end

    return levs
end

"""
    category_plot(df::DataFrame, x::Symbol, y::Symbol, c::Symbol; kwargs...)

Plots two columns x, y of a dataframe `df` with color coding based on
chosen category `c`. Accepts any keyword parameters for plotting.
"""
function category_plot(df::DataFrame, x::Symbol, y::Symbol, c::Symbol; kwargs...)
    categories = unique(df[:, c])
    n = length(categories)

    p = plot()
    for j in 1:n
        dt = df[df[:, c] .== categories[j],:]
        p = plot!(dt[:, x], dt[:, y], label=String(categories[j]); kwargs...)
    end
    return p
end

category_plot(
    cdf, :k, :val_AUC_mean, :aggregation;
    seriestype=:scatter,
    xlabel="k", ylabel="validation AUC"
)
savefig(plotsdir("$(dataset)_$(modelname)_aggregation.png"))

category_plot(
    cdf, :k, :val_AUC_mean, :distance;
    seriestype=:scatter,
    xlabel="k", ylabel="validation AUC"
)
savefig(plotsdir("$(dataset)_$(modelname)_distance.png"))
