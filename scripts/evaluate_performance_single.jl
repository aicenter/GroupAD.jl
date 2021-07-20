# run this script as `julia evaluate_performance_single.jl res.bson` or `julia evaluate_performance_single.jl dir`
# in the second case, it will recursively search for all compatible files in subdirectories
target = ARGS[1]

using DrWatson
@quickactivate
using EvalMetrics
using FileIO
using BSON
using DataFrames
using ValueHistories
using LinearAlgebra

function compute_stats(f::String)
	data = load(f)
	println(abspath(f))
	scores_labels = [(data[:val_scores], data[:val_labels]), (data[:tst_scores], data[:tst_labels])]
	setnames = ["validation", "test"]

	results = []
	for (scores, labels) in scores_labels
		if all(isnan.(scores))
			@info "score is NaN"
			return
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

	@unpack dataset, seed, modelname, npars, parameters = data
	score = parameters[:score]
	d = @dict dataset seed modelname npars score
	measures_val = Symbol.(["val_AUC", "val_AUPRC", "val_TPR_5", "val_F1_5"])
	measures_test = Symbol.(["test_AUC", "test_AUPRC", "test_TPR_5", "test_F1_5"])
	
	for i in 1:length(measures_val)
		d[measures_val[i]] = results[1][i]
	end
	
	for i in 1:length(measures_test)
		d[measures_test[i]] = results[2][i]
	end
	
	safesave(datadir("experiments/contamination-0.0/vae_basic/MIL_results", savename(d, "bson")), d);
end


function query_stats(target::String)
	if isfile(target)
		try
			println(compute_stats(target))
			println("")
		catch e
			@info "$target not compatible"
		end
	else
		query_stats.(joinpath.(target, filter(x->!(occursin("model_", string(x))), readdir(target))))
	end
end

query_stats(target)
