using DrWatson
@quickactivate
using GroupAD
using GroupAD: Evaluation
using DataFrames
using Statistics
using EvalMetrics
using BSON

include(scriptsdir("evaluation", "MIL", "workflow.jl"))

##########################################
############## leave-one-in ##############
##########################################

mnist_results_in = load(datadir("results", "MNIST", "mnist_results_in.bson"))

model_names = ["kNNagg", "VAEagg", "VAE", "NS", "PoolModel", "MGMM"]
modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel", "MGMM"]
modelvec = map(key -> mnist_results_in[key], modelnames)
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map(key -> mnist_results_in[key], modelnames)
# add modelname
#knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map((d, m) -> insertcols!(d, :model => m), modelvec, modelnames)
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map((d, m) -> insertcols!(d, :model => m), modelvec, modelnames)
df = vcat(knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm, cols=:union)

# create dataframe
df_red = df[:, [:class, :model, :val_AUC_mean, :test_AUC_mean, :val_AUPRC_mean, :test_AUPRC_mean]]
df_red = df[:, [:class, :model, :test_AUC_mean]]
sort!(df_red, [:class, :model])

g = groupby(df_red, :class)
nm = g[1][:, :model] |> Array{String,1}
g = map(x -> rename(x, :test_AUC_mean => Symbol(x[1,:class])), g)
g = hcat(map(x -> x[:, 3], g)...)
df1 = DataFrame(g')
rename!(df1, nm)
df1[:, :digit] = map(i -> "$i", 0:9)
df1
df_new = df1[:, [7,3,5,6,4,2,1]]
rename!(df_new, vcat("digit", model_names))

avg = map(x -> typeof(x) == Array{Float64,1} ? mean(x) : "Average", eachcol(df_new))
#avg_rank maybe do it if there is time
push!(df_new, avg)

using PrettyTables

l_max = LatexHighlighter(
    (data, i, j) -> (data[i,j] == maximum(df_new[i, 2:7])) && typeof(data[i,j])!==String,
    ["textbf", "textcolor{blue}"]
)
l_min = LatexHighlighter(
    (data, i, j) -> (data[i,j] == minimum(df_new[i, 2:7])) && typeof(data[i,j])!==String,
    ["textcolor{red}"]
)

t = pretty_table(
    df_new,
    highlighters = (l_max, l_min),
    formatters = ft_printf("%5.3f"),
    backend=:latex, tf=tf_latex_booktabs, nosubheader=true
)

###########################################
############## leave-one-out ##############
###########################################

mnist_results_out = load(datadir("results", "MNIST", "mnist_results_out.bson"))

model_names = ["kNNagg", "VAEagg", "VAE", "NS", "PoolModel", "MGMM"]
modelnames = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel", "MGMM"]
modelvec = map(key -> mnist_results_out[key], modelnames)
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map(key -> mnist_results_out[key], modelnames)
# add modelname
#knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map((d, m) -> insertcols!(d, :model => m), modelvec, modelnames)
knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm = map((d, m) -> insertcols!(d, :model => m), modelvec, modelnames)
df = vcat(knn_basic, vae_basic, vae_instance, statistician, poolmodel, mgmm, cols=:union)

# create dataframe
df_red = df[:, [:class, :model, :val_AUC_mean, :test_AUC_mean, :val_AUPRC_mean, :test_AUPRC_mean]]
df_red = df[:, [:class, :model, :test_AUC_mean]]
sort!(df_red, [:class, :model])

g = groupby(df_red, :class)
nm = g[1][:, :model] |> Array{String,1}
g = map(x -> rename(x, :test_AUC_mean => Symbol(x[1,:class])), g)
g = hcat(map(x -> x[:, 3], g)...)
df1 = DataFrame(g')
rename!(df1, nm)
df1[:, :digit] = map(i -> "$i", 0:9)
df_new = df1[:, [7,3,5,6,4,2,1]]
rename!(df_new, vcat("digit", model_names))

avg = map(x -> typeof(x) == Array{Float64,1} ? mean(x) : "Average", eachcol(df_new))
#avg_rank maybe do it if there is time
push!(df_new, avg)

using PrettyTables

l_max = LatexHighlighter(
    (data, i, j) -> (data[i,j] == maximum(df_new[i, 2:7])) && typeof(data[i,j])!==String,
    ["textbf", "textcolor{blue}"]
)
l_min = LatexHighlighter(
    (data, i, j) -> (data[i,j] == minimum(df_new[i, 2:7])) && typeof(data[i,j])!==String,
    ["textcolor{red}"]
)

t = pretty_table(
    df_new,
    highlighters = (l_max, l_min),
    formatters = ft_printf("%5.3f"),
    backend=:latex, tf=tf_latex_booktabs, nosubheader=true
)