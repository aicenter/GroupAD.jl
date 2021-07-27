function collect_files!(target::String, files)
	if isfile(target)
		push!(files, target)
	else
		for file in readdir(target, join=true)
			collect_files!(file, files)
		end
	end
	files
end

"""
	collect_files(target)
Walks recursively the `target` directory, collecting all files only along the way.
"""
collect_files(target) = collect_files!(target, String[])

"""
    collect_scores(target::String)

Walk recursively the `target` directory and collects all score files along the way.
(Filters out the model files - "model_.*")
"""
function collect_scores(target::String)
    files = collect_files(target)
    return filter(x -> !(occursin("model_", string(x))), files)
end

"""
    results_dataframe_row(target::String)

Returns a DataFrame (single row) from the results file.
If there are different parameters exported to the DataFrame,
missing columns are filled with `missing`.
"""
function results_dataframe_row(target::String)
    data = load(target)

    scores_labels = [(data[:val_scores], data[:val_labels]), (data[:tst_scores], data[:tst_labels])]
	setnames = ["validation", "test"]

    # calculate the scores (AUC-ROC, AUC-PR etc.)
	results = []
	for (scores, labels) in scores_labels
		if all(isnan.(scores))
			@info "score is NaN"
			return
		end
		scores = Base.vec(scores)
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

    # create a dictionary for the future dataframe
	@unpack dataset, seed, modelname, npars = data
	d = @dict dataset seed modelname npars 
	measures_val = Symbol.(["val_AUC", "val_AUPRC", "val_TPR_5", "val_F1_5"])
	measures_test = Symbol.(["test_AUC", "test_AUPRC", "test_TPR_5", "test_F1_5"])
	
	for i in 1:length(measures_val)
		d[measures_val[i]] = results[1][i]
	end
	
	for i in 1:length(measures_test)
		d[measures_test[i]] = results[2][i]
	end

    # add the model parameters
	pars = data[:parameters]
	df_params = DataFrame(hcat(values(pars)...), vcat(keys(pars)...))

    # create the resulting DataFrame (row)
	hcat(DataFrame(d), df_params)
end

function results_dataframe(target::Array{String,1})
    df = results_dataframe_row(target[1])
    for i in 2:length(target)
        df = vcat(df, results_dataframe_row(target[i]), cols=:union)
    end
    df
end

"""
    results_dataframe(target::String)

Collects all result files from the folder specified and returns a DataFrame
with calculated scores and parameters of the model (and data).
"""
function results_dataframe(target::String)
    files = collect_scores(target)
    results_dataframe(files)
end