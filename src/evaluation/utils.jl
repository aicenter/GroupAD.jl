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
    collect_models(target::String)

Walk recursively the `target` directory and collects all model files along the way.
"""
function collect_models(target::String)
    files = collect_files(target)
    return filter(x -> (occursin("model_", string(x))), files)
end

"""
    results_dataframe_row(target::String)

Returns a DataFrame (single row) from the results file.
If there are different parameters exported to the DataFrame,
missing columns are filled with `missing`.
"""
function results_dataframe_row(target::String, verb = false)
    data = load(target)

    scores_labels = [(data[:val_scores], data[:val_labels]), (data[:tst_scores], data[:tst_labels])]
	setnames = ["validation", "test"]

    # calculate the scores (AUC-ROC, AUC-PR etc.)
	results = []
	for (scores, labels) in scores_labels
		if all(isnan.(scores))
			if verb
				@info "score is NaN"
			end
			return
		end
		
		scores = Base.vec(scores)
		roc = roccurve(labels, scores)
		auc = auc_trapezoidal(roc...)
		prc = prcurve(labels, scores)
		auprc = auc_trapezoidal(prc...)

		t5 = threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		tpr5 = true_positive_rate(cm5)
		f5 = f1_score(cm5)

		push!(results, [auc, auprc, tpr5, f5])
	end

    # create a dictionary for the future dataframe
	@unpack dataset, seed, modelname = data
	d = @dict dataset seed modelname
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
function results_dataframe_row(target::String, verb = false)
	data = []
	# if the file is not found, returns nothing
	try
    	data = load(target)
	catch e
		@warn "Model not found."
		return 
	end

    scores_labels = [(data[:val_scores], data[:val_labels]), (data[:tst_scores], data[:tst_labels])]
	setnames = ["validation", "test"]

    # calculate the scores (AUC-ROC, AUC-PR etc.)
	results = []
	for (scores, labels) in scores_labels
		if all(isnan.(scores))
			if verb
				@info "score is NaN"
			end
			return
		end
		
		scores = Base.vec(scores)
		roc = roccurve(labels, scores)
		auc = auc_trapezoidal(roc...)
		prc = prcurve(labels, scores)
		auprc = auc_trapezoidal(prc...)

		push!(results, [auc, auprc])
	end

    # create a dictionary for the future dataframe
	@unpack dataset, seed, modelname = data
	d = @dict dataset seed modelname
	measures_val = Symbol.(["val_AUC", "val_AUPRC"])
	measures_test = Symbol.(["test_AUC", "test_AUPRC"])
	
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
	j = 1
	df = []
	for i in 1:length(target)
		r = results_dataframe_row(target[i])
		if !isnothing(r)
			df = results_dataframe_row(target[i])
			break
		end
		j += 1
	end

    for i in j:length(target)
		r = results_dataframe_row(target[i])
		if !isnothing(r)
        	df = vcat(df, r, cols=:union)
		end
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

"""
    find_best_model(folder::String [, groupkey]; metric=:val_AUC)

Recursively goes through given folder and finds the best model based on
chosen metric, default is validation AUC.

If `groupkey` is present, returns the best model for each category of groupkey.
Group key can be both a symbol or an array of symbols.
"""
function find_best_model(folder::String; metric=:val_AUC, save_best_seed=false)
    #folder = datadir("experiments", "contamination-0.0", modelname, dataset)
    data = results_dataframe(folder)
    point = load(collect_scores(folder)[1])
    params = point[:parameters]

    g = groupby(data, [keys(params)...])
    un = unique(map(x -> size(x), g))
    if length(un) != 1
        idx = findall(x -> size(x,1) > 5, g)
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing $(length(g) - length(idx)) groups out of $(length(g)) with less than 6 seeds."
        g = g[idx]
    end

    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]
    cdf = combine(g, map(x -> x => mean, metricsnames))
    sort!(cdf, :val_AUC_mean, rev=true)
    best_model = cdf[1,:]

    if save_best_seed
        _nm = names(best_model)
        nm = _nm[occursin.("mean", _nm) .== 0]
        nm = nm[occursin.("L", nm) .== 0]
        values = best_model[[nm...]]

        idx = findall(row -> row[nm] == values, eachrow(data))
        s = sort(data[idx, :], :val_AUC, rev = true)[1,:][:seed]
        return DataFrame(best_model), s
    else
        return best_model
    end
end
function find_best_model(folder, groupkey, metric=:val_AUC)
    #folder = datadir("experiments", "contamination-0.0", modelname, dataset, "scenario=$scenario")
    data = results_dataframe(folder)
    point = load(collect_scores(folder)[1])
    params = point[:parameters]

    g_score = groupby(data, groupkey)
    g = map(x -> groupby(x, [keys(params)...]), g_score)
    un = unique(vcat(map(x -> unique(map(y -> size(y), x)), g)...))

    if length(un) != 1
        idx = findall.(x -> size(x,1) > 5, g)
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing $(sum(length.(g)) - sum(length.(idx))) groups out of $(sum(length.(g))) with less than 6 seeds."
        g = map(i -> g[i][idx[i]], 1:length(g))
    end

    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]
    cdf = map(y -> combine(y, map(x -> x => mean, metricsnames)), g)
    cdf_sorted = map(x -> sort(x, :val_AUC_mean, rev=true), cdf)
    best_models = vcat(map(x -> DataFrame(x[1,:]), cdf_sorted)...)
end

"""
    mill_results(modelname, mill_datasets [, groupkey]; info = true)

Calculates the results dataframe for full validation dataset for all
MIL datasets and chosen model.

If `groupkey` is included, finds best model for each category of groupkey.
"""
function mill_results(modelname, mill_datasets; info = true)
    res = []
    # full results folder
    folder = datadir("experiments", "contamination-0.0", modelname)

    for d in mill_datasets
        # dataset folder
        dfolder = joinpath(folder, d)
        model = find_best_model(dfolder) |> DataFrame
        #@info d
        insertcols!(model, :dataset => d)
        push!(res, model)
        if info
            @info "Best $modelname model for $d found."
        end
    end

    results = vcat(res...)
end
function mill_results(modelname, mill_datasets, groupkey; info = true)
    res = []
    # full results folder
    folder = datadir("experiments", "contamination-0.0", modelname)

    for d in mill_datasets
        # dataset folder
        dfolder = joinpath(folder, d)
        model = find_best_model(dfolder, groupkey) |> DataFrame
        #@info d
        insertcols!(model, :dataset => d)
        push!(res, model)
        if info
            @info "Best $modelname model for $d found."
        end
    end

    results = vcat(res...)
end