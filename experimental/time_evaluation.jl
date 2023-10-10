using DataFrames, DrWatson, BSON
using Statistics
using CSV

using Plots
using StatsPlots
ENV["GKSwstype"] = "100"

#####################################################################
###                             TIMES                             ###
#####################################################################

# df = collect_results(datadir("times"))
# df_matej = CSV.read(datadir("times/times-all.csv"), DataFrame)
df = collect_results("/home/maskomic/projects/GroupAD.jl/data/times/")
df_matej = CSV.read(datadir("/home/maskomic/projects/GroupAD.jl/data/times/times-all.csv"), DataFrame)

df_matej.dataset = map(x -> split(x, "-")[end], df_matej.dataset)
df.dataset = map(x -> split(x, "-")[end], df.dataset)

# MEAN
df.fit_t_mean = map(x -> mean(log10.(1 .+ unique(x))), df.fit_t)
df.tst_eval_t_mean = map(x -> mean(log10.(1 .+ unique(x))), df.tst_eval_t)

cdf = combine(
    groupby(
        df_matej, [:model, :dataset]
    ),
    :fit_t => x -> mean(log10.(1 .+ unique(x))),
    :tst_eval_t => x -> mean(log10.(1 .+ unique(x)))
)
rename!(cdf, [:model, :dataset, :fit_t_mean, :tst_eval_t_mean])

# MEDIAN
df.fit_t_mean = map(x -> median(log10.(1 .+ unique(x))), df.fit_t)
df.tst_eval_t_mean = map(x -> median(log10.(1 .+ unique(x))), df.tst_eval_t)

cdf = combine(
    groupby(
        df_matej, [:model, :dataset]
    ),
    :fit_t => x -> median(log10.(1 .+ unique(x))),
    :tst_eval_t => x -> median(log10.(1 .+ unique(x)))
)
rename!(cdf, [:model, :dataset, :fit_t_mean, :tst_eval_t_mean])

# MEDIAN without log
df.fit_t_mean = map(x -> median(unique(x)), df.fit_t)
df.tst_eval_t_mean = map(x -> median(unique(x)), df.tst_eval_t)

cdf = combine(
    groupby(
        df_matej, [:model, :dataset]
    ),
    :fit_t => x -> median(unique(x)),
    :tst_eval_t => x -> median(unique(x))
)
rename!(cdf, [:model, :dataset, :fit_t_mean, :tst_eval_t_mean])

misa_df = df[:, [:dataset, :model, :fit_t_mean, :tst_eval_t_mean]]
full_df = vcat(cdf, misa_df)

@df full_df boxplot(:model, :fit_t_mean, fillalpha=0.75, linewidth=2, label="", marker=nothing)
# @df full_df violin(:model, :fit_t_mean, fillalpha=0.75, linewidth=2, markerstrokewidth=0.1, markersize=3, label="")
@df full_df dotplot!(:model, :fit_t_mean, markerstrokewidth=0, markersize=3, color=:black, label="")
savefig("plot.png")


gdf = groupby(full_df, :dataset)
gdf2 = filter(x -> nrow(x) > 5, gdf)
using StatsBase

# modelnames = gdf[1].model
modelnames = gdf2[1].model

train_ranks = []
for g in gdf2
	x = g.fit_t_mean
	print(tiedrank(x))
	push!(train_ranks, tiedrank(x))
end
train_rank = mean(hcat(train_ranks...), dims=2)

test_ranks = []
for g in gdf2
	x = g.tst_eval_t_mean
	print(tiedrank(x))
	push!(test_ranks, tiedrank(x))
end
test_rank = mean(hcat(test_ranks...), dims=2)

results = DataFrame(
	:model => modelnames,
	:train_rank => train_rank[:],
	:test_rank => test_rank[:],
)

using PrettyTables
pretty_table(results, tf=tf_latex_booktabs)

stuff = vcat(gdf2...)
final = combine(
	groupby(stuff, :model),
	:fit_t_mean => mean,
	:fit_t_mean => median,
	:fit_t_mean => std,
	:tst_eval_t_mean => mean,
	:tst_eval_t_mean => median,
	:tst_eval_t_mean => std,
)
rename!(final, ["model", "train_time_mean", "train_time_median", "train_time_std", "inference_time_mean", "inference_time_median", "inference_time_std"])
combine(groupby(stuff, :model), :fit_t_mean => median)
combine(groupby(stuff, :model), :fit_t_mean => std)
combine(groupby(stuff, :model), :tst_eval_t_mean => mean)
combine(groupby(stuff, :model), :tst_eval_t_mean => median)
combine(groupby(stuff, :model), :tst_eval_t_mean => std)

auc_ranks = transpose([2.9 5.1 2.3 5.7 3.8 5.6 5.4])
modelnames = [
	"IVAE",
	"VAE-B",
	"NS",
	"KNN",
	"PoolM.",
	"SetVAE",
	"FN-VAE",
]

modelnames = [
	"vae_instance",
	"vae_basic",
	"statistician",
	"knn_basic",
	"PoolModel",
	"setvae",
	"foldingnet_vae",
]

ranks_df = DataFrame(:model => modelnames, :ranks => auc_ranks[:])

data = leftjoin(final, ranks_df, on=:model)
# markers = [:circle, :square, :dtriangle, :utriangle, :x, :hexagon, :+]
markers = [:circle, :square, :dtriangle, :utriangle, :diamond, :star4, :hexagon]
ms = 4

p1 = plot()
p2 = plot()
for i in 1:7
    p1 = scatter!(p1, [data.train_time_mean[i]], [data.ranks[i]], label=data.model[i], marker=markers[i], markersize=ms)
    p2 = scatter!(p2, [data.inference_time_mean[i]], [data.ranks[i]], label=data.model[i], marker=markers[i], markersize=ms)
end
p1 = plot!(p1, xlabel="mean training time (s)", ylabel="AUC rank", legend=nothing, yticks=[1,2,3,4,5,6])
p2 = plot!(p2, xlabel="mean inference time (s)", yticks=[1,2,3,4,5,6])

using Plots.Measures
plot(
	p1, p2, layout = (1,2), size=(800,300), xguidefontsize=8, yguidefontsize=8, legendfontsize=8,
	label=["IVAE" "VAE-B" "NS" "KNN" "PoolModel" "SetVAE" "FN-VAE"],
	bottom_margin=20Plots.px,
	left_margin=10Plots.px,
	yticks=[1,2,3,4,5,6],
	ylims=[2,6]
)

savefig("plot.pdf")