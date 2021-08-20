using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics

using Plots
using StatsPlots
ENV["GKSwstype"] = "100"

include(scriptsdir("evaluation", "workflow.jl"))

mill_datasets = ["BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2", "Mutagenesis1", "Mutagenesis2",
                    "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein", "Tiger", "UCSBBreastCancer",
                    "Web1", "Web2", "Web3", "Web4", "WinterWren"]

# for a single dataset
modelname = "statistician"
dataset = "BrownCreeper"
r = combined_dataframe(modelname, dataset)
boxplot(r[:, :type], r[:, :val_AUC_mean],xrotation=55, ylims=(0,1))
savefig(plotsdir("boxplot_$(modelname)_$(dataset).png"))

# for all MIL datasets
statistician_df = DataFrame[]

for d in mill_datasets
    r = combined_dataframe(modelname, d)
    rr = hcat(r, DataFrame(:dataset => repeat([d], size(r,1))))
    push!(statistician_df, rr)
end

metricsnames = [:val_AUC_mean, :test_AUC_mean]
NS_df = vcat(statistician_df...)
g = groupby(NS_df, :type)
gg = map(x -> groupby(x, :dataset), g)

CDF = map(x -> combine(x, :test_AUC_mean => mean => Symbol("$(x[1][1,:type])")), gg)
final = hcat(CDF..., makeunique=true)

prep = final[:, [:sum, :mean, Symbol("maximum,"), :poisson, :lognormal, :logU, Symbol("poisson+logU"), Symbol("lognormal+logU"), :chamfer, Symbol("MMD-GaussianKernel"), Symbol("MMD-IMQKernel")]] |> Array

groupedbar(
    mill_datasets[1:10],prep[1:10,:],xrotation=15,
    label=["sum" "mean" "maximum" "Poisson" "LogNormal" "log U" "Poisson + log U" "LogNormal + log U" "Chamfer distance" "MMD Gaussian kernel" "MMD IMQ kernel"],
    ylims=(0,1), size=(1800,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerright, legendtitle="Score"
    )
savefig(plotsdir("test_mean_1-10.png"))
groupedbar(
    mill_datasets[11:20],prep[11:20,:],xrotation=15,
    label=["sum" "mean" "maximum" "Poisson" "LogNormal" "log U" "Poisson + log U" "LogNormal + log U" "Chamfer distance" "MMD Gaussian kernel" "MMD IMQ kernel"],
    ylims=(0,1), size=(1800,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerright, legendtitle="Score"
    )
savefig(plotsdir("test_mean_11-20.png"))

# needs to be divided into two barplots


# choose the best model on validation and get corresponding test AUC
# first index is score
# second index is dataset
# i want to find best model in each dataset
filter(row -> row[:val_AUC_mean] < maximum(gg[1][10][:, :val_AUC_mean]), gg[1][10])[1,:]

CDF = map(
    x -> map(
        y -> DataFrame(filter(
            row -> row[:val_AUC_mean] == maximum(y[:, :val_AUC_mean]), y
            )[1,:]),
        x),
    gg);

redCDF = map(x -> map(y -> y[:, [:dataset, :type, :val_AUC_mean, :test_AUC_mean]], x), CDF);
vCDF = vcat(vcat(redCDF...)...)
new_g = groupby(vCDF, :type)
new_g2 = map(x -> rename(x, :test_AUC_mean => x[1,:type]),new_g)

final_max_test = hcat(new_g2..., makeunique=true)
prep_max = final_max_test[:, [:sum, :mean, Symbol("maximum,"), :poisson, :lognormal, :logU, Symbol("poisson+logU"), Symbol("lognormal+logU"), :chamfer, Symbol("MMD-GaussianKernel"), Symbol("MMD-IMQKernel")]] |> Array

groupedbar(
    mill_datasets[1:10],prep_max[1:10,:],xrotation=15,
    label=["sum" "mean" "maximum" "Poisson" "LogNormal" "log U" "Poisson + log U" "LogNormal + log U" "Chamfer distance" "MMD Gaussian kernel" "MMD IMQ kernel"],
    ylims=(0,1), size=(1800,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerright, legendtitle="Score"
    )
savefig(plotsdir("test_max_1-10.png"))

groupedbar(
    mill_datasets[11:20],prep_max[11:20,:],xrotation=15,
    label=["sum" "mean" "maximum" "Poisson" "LogNormal" "log U" "Poisson + log U" "LogNormal + log U" "Chamfer distance" "MMD Gaussian kernel" "MMD IMQ kernel"],
    ylims=(0,1), size=(1800,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerright, legendtitle="Score"
    )
savefig(plotsdir("test_max_11-20.png"))