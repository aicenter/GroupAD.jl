using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics

modelname = "vae_basic"
dataset = "BrownCreeper"
df = combined_dataframe(modelname, dataset)

category_plot(
    df, :zdim, :val_AUC_mean, :activation;
    seriestype=:scatter,
    xlabel="zdim", ylabel="validation AUC",
    legend=:outerright
)
savefig(plotsdir("$(dataset)_$(modelname)_zdim_activation.png"))


category_plot(
    df, :hdim, :val_AUC_mean, :activation;
    seriestype=:scatter,
    xlabel="zdim", ylabel="validation AUC",
    legend=:outerright
)
savefig(plotsdir("$(dataset)_$(modelname)_hdim_activation.png"))

function parameter_mean(df::DataFrame, par::Symbol; metric = :val_AUC_mean)
    g = groupby(df, par)
    means = map(x -> mean(x[:, metric]), g)
    sigma = map(x -> std(x[:, metric]), g)
    count = map(x -> size(x,1), g)

    k = keys(g)
    v = map(x -> x[par], k)

    res = DataFrame(par => v, :mean => means, :std => sigma, :count => count)
    return sort(res, :mean, rev=true)
end
function parameter_mean(df::DataFrame, par::Array{Symbol,1}; metric = :val_AUC_mean)
    g = groupby(df, par)
    means = map(x -> mean(x[:, metric]), g)
    sigma = map(x -> std(x[:, metric]), g)
    count = map(x -> size(x,1), g)

    k = keys(g)
    if length(par) > 2
        error("Currently unsupported more than 2 parameters.")
    end
    v1 = map(x -> x[par[1]], k)
    v2 = map(x -> x[par[2]], k)

    res = DataFrame(par[1] => v1, par[2] => v2, :mean => means, :std => sigma, :count => count)
    return sort(res, :mean, rev=true)
end

parameter_mean(df, :aggregation)
parameter_mean(df, :activation)
parameter_mean(df, :zdim)
parameter_mean(df, :hdim)
parameter_mean(df, :nlayers)
parameter_mean(df, :score)
parameter_mean(df, :var)
parameter_mean(df, [:zdim, :hdim])

scatter(df[:, :zdim], df[:, :hdim], df[:, :val_AUC_mean])
savefig("$(dataset)_$(modelname)_hdim+zdim")