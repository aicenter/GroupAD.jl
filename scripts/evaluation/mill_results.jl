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

"""
    mill_results(modelname, mill_datasets; info = true)

Calculates the results dataframe for full validation dataset for all
MIL datasets and chosen model.
"""
function mill_results(modelname, mill_datasets; info = true)
    res = []

    for d in mill_datasets
        model = find_best_model(modelname, d) |> DataFrame
        #@info d
        insertcols!(model, :dataset => d)
        push!(res, model)
        if info
            @info "Best $modelname model for $d found."
        end
    end

    results = vcat(res...)
end

"""
    barplot_mill(modelname, results; sorted = false)

Plots and saves a barplot of all MIL datasets and given model.
"""
function barplot_mill(modelname, results; sorted = false, savef = false)
    if sorted
        res_sort = sort(results, :val_AUC_mean, rev=true)
    else
        res_sort = results
    end
    r = res_sort[:, [:val_AUC_mean, :test_AUC_mean]] |> Array
    p = groupedbar(
        res_sort[:, :dataset],r,xrotation=55,legendtitle=modelname,
        label=["val-AUC" "test-AUC"], ylabel="AUC", legend=:bottomright, ylims=(0,1))
    if savef
        savefig(plotsdir("barplot_$(modelname).png"))
    end
    return p
end

# calculate results dataframe (full validation)
res_knn = mill_results("knn_basic",mill_datasets)
res_vae_basic = mill_results("vae_basic",mill_datasets)
res_vae_instance = mill_results("vae_instance",mill_datasets)
res_mgmm = mill_results("MGMM",mill_datasets)
res_statistician = mill_results("statistician", mill_datasets)
res_pool = mill_results("PoolModel", mill_datasets)

results_full_validation = Dict(
    :knn_basic => res_knn,
    :vae_basic => res_vae_basic,
    :vae_instance => res_vae_instance,
    :mgmm => res_mgmm,
    :statistician => res_statistician,
    :poolmodel => res_pool
)
safesave(datadir("dataframes", "results_full_validation.bson"), results_full_validation)
R = load(datadir("dataframes", "results_full_validation.bson"))

@unpack knn_basic, vae_basic, vae_instance, mgmm, statistician, poolmodel = R

# barplots for each model
p_knn = barplot_mill("kNN (agg)", knn_basic)
savefig(plotsdir("barplots", "knn.png"))
p_mgmm = barplot_mill("MGMM", mgmm)
savefig(plotsdir("barplots", "MGMM.png"))
p_vae_b = barplot_mill("VAE (agg)", vae_basic)
savefig(plotsdir("barplots", "vae_basic.png"))
p_vae_i = barplot_mill("VAE", vae_instance)
savefig(plotsdir("barplots", "vae_instance.png"))
p_ns = barplot_mill("Neural Statistician", statistician)
savefig(plotsdir("barplots", "statistician.png"))
p_pool = barplot_mill("PoolModel", poolmodel)
savefig(plotsdir("barplots", "PoolModel.png"))

knn_basic_b = rename(knn_basic[:, [:dataset, :test_AUC_mean]], :test_AUC_mean => :knn_basic)
vae_basic_b = rename(vae_basic[:, [:dataset, :test_AUC_mean]], :test_AUC_mean => :vae_basic)
mgmm_b = rename(mgmm[:, [:dataset, :test_AUC_mean]], :test_AUC_mean => :mgmm)
statistician_b = rename(statistician[:, [:dataset, :test_AUC_mean]], :test_AUC_mean => :statistician)
vae_instance_b = rename(vae_instance[:, [:dataset, :test_AUC_mean]], :test_AUC_mean => :vae_instance)
pool_b = rename(poolmodel[:, [:dataset, :test_AUC_mean]], :test_AUC_mean => :PoolModel)

bar_mill = hcat(knn_basic_b, vae_basic_b, mgmm_b, statistician_b, vae_instance_b, pool_b, makeunique=true)
modelnames = ["kNN (agg)" "VAE (agg)" "MGMM" "Neural Statistician" "VAE" "PoolModel"]
mat = bar_mill[:, [:knn_basic, :vae_basic, :mgmm, :statistician, :vae_instance, :PoolModel]] |> Array

groupedbar(
    mill_datasets[1:10], mat[1:10,:], xrotation=15,
    label=modelnames, ylabel="AUC",
    ylims=(0,1), size=(1850,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerbottom
    )
savefig(plotsdir("barplots", "barplot_1-10.png"))
groupedbar(
    mill_datasets[11:20],mat[11:20,:],xrotation=15,
    label=modelnames, ylabel="AUC",
    ylims=(0,1), size=(1850,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerright
    )
savefig(plotsdir("barplots", "barplot_11-20.png"))