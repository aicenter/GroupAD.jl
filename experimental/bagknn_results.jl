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

		t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		tpr5 = EvalMetrics.true_positive_rate(cm5)
		f5 = EvalMetrics.f1_score(cm5)

		push!(results, [auc, auprc, tpr5, f5])
	end

	# DataFrame(measure = ["AUC", "AUPRC", "TPR@5", "F1@5"], validation = results[1], test = results[2])
    results[1][1:2]..., results[2][1:2]...
end

function findmaxs(gdf::GroupedDataFrame, metric::Symbol)
    res = []
    p_names = keys(gdf[1].parameters[1])
    for g in gdf
        vauc, ix = findmax(g[:, metric])
        tauc = g[ix, :test_AUC]

        # export parameters
        p = g.parameters[ix]
        pdf = DataFrame(keys(p) .=> values(p))

        push!(res, [g.dataset[1] values(g.parameters[ix])... round(vauc, digits=3) round(tauc, digits=3)])       
    end
    df = DataFrame(vcat(res...), [:dataset, p_names..., :val_AUC, :test_AUC])
    sort(df, :dataset)
end


df = collect_results!(datadir("experiments", "contamination-0.0", "bag_knn"), subfolders=true)
df_results = df[:, [:dataset, :parameters, :seed]]

metrics = []
for row in eachrow(df)
    vauc, vaupr, tauc, taupr = compute_stats(row)
    push!(metrics, [vauc vaupr tauc taupr])
    #if any([vauc, vaupr, tauc, taupr] .!= 0)
    #    push!(metrics, [vauc vaupr tauc taupr])
    #end
end
res = vcat(metrics...)

res_df = DataFrame(["val_AUC", "val_AUPRC", "test_AUC", "test_AUPRC"] .=> eachcol(res))
df_results = hcat(df_results, res_df)
# filtered_df = filter(["val_AUC", "val_AUPRC", "test_AUC", "test_AUPRC"] => (a,b,c,d) -> !(a == b == c == d == 0.0), df_results)

g = groupby(df_results, [:parameters, :dataset])

# filter out groups with < 5 seeds
b = map(x -> nrow(x) >= 5, g)
# b = map(i -> nrow(g[i]) >= 5, 1:length(g))
g = g[b]
cdf = combine(g, ["val_AUC", "val_AUPRC", "test_AUC", "test_AUPRC"] .=> mean, renamecols=false)

g2 = groupby(cdf, :dataset)
pretty_table(findmaxs(g2, :val_AUC), nosubheader=true, crop=:none)


