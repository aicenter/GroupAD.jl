using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics
using PrettyTables

using Plots
using StatsPlots
#using PlotlyJS
ENV["GKSwstype"] = "100"

modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel", "MGMM"]
modelscores = [:distance, :score, :type, :type, :type, :score]

# load results collection
toy_results_collection = load(datadir("results/toy", "toy_results_collection.bson"))

df_vec = map(name -> toy_results_collection[name], modelnames)
df_vec2 = map(name -> insertcols!(toy_results_collection[name], :model => name), modelnames)
df_full = vcat(df_vec2..., cols=:union)
sort!(df_full, :val_AUC_mean, rev=true)
g = groupby(df_full, [:model, :scenario])
df_best = map(df -> DataFrame(df[1,[:model, :scenario, :test_AUC_mean]]), g)
df_red = vcat(df_best...)

s1 = filter(:scenario => scenario -> scenario == 1, df_red)[:, [:model, :test_AUC_mean]]
s2 = filter(:scenario => scenario -> scenario == 2, df_red)[:, [:model, :test_AUC_mean]]
s3 = filter(:scenario => scenario -> scenario == 3, df_red)[:, [:model, :test_AUC_mean]]

H = []
for modelname in modelnames
    v1 = s1[s1[:, :model] .== modelname, :test_AUC_mean]
    v2 = s2[s2[:, :model] .== modelname, :test_AUC_mean]
    v3 = s3[s3[:, :model] .== modelname, :test_AUC_mean]
    V = vcat(v1,v2,v3)
    push!(H, V)
end

H2 = hcat(H...)
H3 = vcat(H2, mean(H2, dims=1))
_final = DataFrame(hcat(["1","2","3","Average"],H3))
nice_modelnames = ["scenario", "kNNagg", "VAEagg", "VAE", "NS", "PoolModel", "MGMM"]
final = rename(_final, nice_modelnames)


l_max = LatexHighlighter(
    (data, i, j) -> (data[i,j] == maximum(final[i, 2:7])) && typeof(data[i,j])!==String,
    ["textbf", "textcolor{blue}"]
)
l_min = LatexHighlighter(
    (data, i, j) -> (data[i,j] == minimum(final[i, 2:7])) && typeof(data[i,j])!==String,
    ["textcolor{red}"]
)

t = pretty_table(
    final,
    highlighters = (l_max, l_min),
    formatters = ft_printf("%5.3f"),
    backend=:latex, tf=tf_latex_booktabs, nosubheader=true
)