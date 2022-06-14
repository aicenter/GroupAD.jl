using DrWatson
@quickactivate
using DataFrames

using EvalMetrics
using FileIO
using BSON
using ValueHistories
using LinearAlgebra
using PrettyTables
using Statistics

# for generative models
using DistributionsAD, GenerativeModels, Flux, GroupAD

function compute_stats(row::DataFrameRow)
    if typeof(row[:tst_scores]) <: BitVector
        return (0.0,0.0,0.0,0.0)
    end
	scores_labels = [(row[:val_scores], row[:val_labels]), (row[:tst_scores], row[:tst_labels])]
	setnames = ["validation", "test"]

	results = []
	for (scores, labels) in scores_labels
		if all(isnan.(scores))
			@info "score is NaN"
			# return (NaN, NaN, NaN, NaN)
            return (0.0,0.0,0.0,0.0)
		end
		scores = vec(scores)
		roc = EvalMetrics.roccurve(labels, scores)
		auc = EvalMetrics.auc_trapezoidal(roc...)
		prc = EvalMetrics.prcurve(labels, scores)
		auprc = EvalMetrics.auc_trapezoidal(prc...)

		#t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		#cm5 = ConfusionMatrix(labels, scores, t5)
		#tpr5 = EvalMetrics.true_positive_rate(cm5)
		#f5 = EvalMetrics.f1_score(cm5)

		#push!(results, [auc, auprc, tpr5, f5])
        push!(results, [auc, auprc, 0, 0])
	end

	# DataFrame(measure = ["AUC", "AUPRC", "TPR@5", "F1@5"], validation = results[1], test = results[2])
    results[1][1:2]..., results[2][1:2]...
end

"""
	findmaxs(gdf::GroupedDataFrame, metric::Symbol)

Iterates over a grouped dataframe of results. Based on the chosen `metric`,
chooses the best model (should be validation AUC or AUPRC).
"""
function findmaxs(gdf::GroupedDataFrame, metric::Symbol)
    res = []
	# get parameter names for resulting dataframe
    p_names = keys(gdf[1].parameters[1])

    for g in gdf
		# find the maximum value based on `metric`
        vauc, ix = findmax(g[:, metric])
        tauc = g[ix, :test_AUC]

        # export parameters
        p = g.parameters[ix]
        pdf = DataFrame(keys(p) .=> values(p))

        push!(res, [g.dataset[1] values(g.parameters[ix])... round(vauc, digits=3) round(tauc, digits=3)])       
    end

	# create a resulting dataframe
	df = DataFrame(vcat(res...), [:dataset, p_names..., :val_AUC, :test_AUC])
    sort(df, :dataset)
end

model = "knn_basic"
model = "SMM"
model = "SMMC"
model = "vae_basic"
# load results collection
_df = collect_results(datadir("experiments", "contamination-0.0", model, "MIL"), subfolders=true)
# filter model files (for vae, statistician...)
df = filter(:path => x -> !occursin("model", x), _df)

# preallocate and calculate metrics
n = nrow(df)
metrics = zeros(n, 4)
for i in 1:n
    row = df[i,:]
    metrics[i, :] .= compute_stats(row)
end

# add metrics to a dataframe
res_df = DataFrame(["val_AUC", "val_AUPRC", "test_AUC", "test_AUPRC"] .=> eachcol(metrics))

# create results dataframe
df_results = df[:, [:dataset, :parameters, :seed]]
df_results = hcat(df_results, res_df)

# groupby parameters and dataset to average over seeds
g = groupby(df_results, [:parameters, :dataset])

# filter out groups with less than 10 seeds
k = length(g)
b = map(i -> nrow(g[i]) == 10, 1:k)
g = g[b]

# get mean values of metrics over seeds
cdf = combine(g, ["val_AUC", "val_AUPRC", "test_AUC", "test_AUPRC"] .=> mean, renamecols=false)

# group by dataset
g2 = groupby(cdf, :dataset)
# find the best model based on metric (validation AUC)
R = findmaxs(g2, :val_AUC)#[:, Not(:init_seed)]

# reorder columns
c = ncol(R)
R2 = R[:, vcat([1,c-1,c], setdiff(1:c, [1,c,c-1]))]

# create a pretty table
pretty_table(R2, nosubheader=true, crop=:none)
pretty_table(R2, nosubheader=true, backend=:text, tf = tf_markdown)