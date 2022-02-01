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
include(scriptsdir("evaluation", "validation_anomalies.jl"))

models = ["knn_basic", "vae_basic", "MGMM", "vae_instance", "statistician", "PoolModel"]
model = "vae_basic"
d = "Elephant"

R = evaluate_at_val(models, d)
p = plot_at_val(R, models, d)
savefig(plotsdir("all_models_$(d).png"))

mill_datasets = ["BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2", "Mutagenesis1", "Mutagenesis2",
                    "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein", "Tiger", "UCSBBreastCancer",
                    "Web1", "Web2", "Web3", "Web4", "WinterWren"]

mill_best = []
for d  in mill_datasets
    R = evaluate_at_val(models, d)
    plot_at_val(R, models, d; only_test = true)
    savefig(plotsdir("mill_min_6_seeds", "all_models_$(d).png"))
    push!(mill_best, R)
end


R = evaluate_at_val_over_seeds(models, d)
p = plot_at_val(R, models, d)
savefig(plotsdir("all_models_$(d)_seeds=50.png"))

mill_best_seeds = []
for d  in mill_datasets[11:20]
    R = evaluate_at_val_over_seeds(models, d)
    plot_at_val(R, models, d; only_test = true)
    savefig(plotsdir("mill_seeds", "all_models_$(d).png"))
    push!(mill_best_seeds, R)
end

