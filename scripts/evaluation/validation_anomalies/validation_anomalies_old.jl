using Random
using StatsBase
using BSON
using Plots
ENV["GKSwstype"] = "100"


function results_dataframe_row_at_val(target::String, verb = false)
    data = load(target)
    seed = data[:seed]
    no_anomalies = Int(sum(data[:val_labels]))
    val_an = [1, 2, 5, 10, 20, 50, 100]

    scores_labels = [(data[:val_scores], data[:val_labels]), (data[:tst_scores], data[:tst_labels])]
	setnames = ["validation", "test"]

    results_val = []
    results_test = []
    for no in val_an
        if no < no_anomalies
            # calculate the scores (AUC-ROC, AUC-PR etc.)
            k = 1
            for (scores, labels) in scores_labels
                if all(isnan.(scores))
                    if verb
                        @info "score is NaN"
                    end
                    return
                end
                
                an_idx = findall(x -> x == 1, labels)
                Random.seed!(seed)
                chosen_idx = sample(an_idx, no)

                sc = vcat(
                    scores[labels .== 0],
                    scores[chosen_idx]
                )

                lb = vcat(
                    labels[labels .== 0],
                    labels[chosen_idx]
                )

                sc = Base.vec(sc)
                roc = roccurve(lb, sc)
                auc = auc_trapezoidal(roc...)
                prc = prcurve(lb, sc)
                auprc = auc_trapezoidal(prc...)

                if k == 1
                    push!(results_val, [no, auc, auprc])
                else
                    push!(results_test, [auc, auprc])
                end
                k += 1
            end
        end
    end

    # create a dictionary for the future dataframe
	@unpack dataset, seed, modelname = data
	d = @dict dataset seed modelname
	measures_val = Symbol.(["no_anomalous", "val_AUC", "val_AUPRC"])
	measures_test = Symbol.(["test_AUC", "test_AUPRC"])

    val_df = DataFrame(hcat(results_val...)', measures_val)
    test_df = DataFrame(hcat(results_test...)', measures_test)
    df = hcat(val_df, test_df)

    # add the model parameters
	pars = data[:parameters]
	df_params = repeat(DataFrame(hcat(values(pars)...), vcat(keys(pars)...)), size(df, 1))

    # create the resulting DataFrame (row)
	hcat(df, df_params)
end

function results_dataframe_row_at_val_over_seeds(target::String; verb = false, seeds = 50)
    data = load(target)
    no_anomalies = Int(sum(data[:val_labels]))
    val_an = [1, 2, 5, 10, 20, 50, 100]
    if no_anomalies > 100
        val_an = vcat(val_an, no_anomalies)
    else
        val_an = vcat(val_an[val_an .< no_anomalies], no_anomalies)
    end

    # calculate average validation AUC for given number of anomalies
    scores = data[:val_scores]
    labels =  data[:val_labels]
    results_val = []

    for no in val_an
        if no < no_anomalies
            # calculate the AUC score on validation data for given number of anomalies
            # average the AUC over seeds
            if all(isnan.(scores))
                if verb
                    @info "score is NaN"
                end
                return
            end
            
            AUC = []
            for s in 1:minimum([Int(no_anomalies), seeds])
                an_idx = findall(x -> x == 1, labels)
                # make sure its reproducible and no anomalies are repeated
                Random.seed!(s)
                chosen_idx = sample(an_idx, no, replace=false)

                sc = vcat(
                    scores[labels .== 0],
                    scores[chosen_idx]
                )

                lb = vcat(
                    labels[labels .== 0],
                    labels[chosen_idx]
                )

                sc = Base.vec(sc)
                roc = roccurve(lb, sc)
                auc = auc_trapezoidal(roc...)

                push!(AUC, auc)
            end

            push!(results_val, [no, mean(AUC)])
        end
    end

    # calculate the test scores for given model
    # test scores are calculated for full test dataset => calculated only once
    scores = data[:tst_scores]
    labels =  data[:tst_labels]
    results_test = []
    if all(isnan.(scores))
        if verb
            @info "score is NaN"
        end
        return
    end
    scores = Base.vec(scores)
    roc = roccurve(labels, scores)
    test_auc = auc_trapezoidal(roc...)

    # create a dictionary for the future dataframe
	@unpack dataset, seed, modelname = data
	d = @dict dataset seed modelname
	measures_val = Symbol.(["no_anomalous", "val_AUC"])

    df = DataFrame(hcat(results_val...)', measures_val)
    insertcols!(df, :test_AUC => test_auc)

    # add the model parameters
	pars = data[:parameters]
	df_params = repeat(DataFrame(hcat(values(pars)...), vcat(keys(pars)...)), size(df, 1))

    # create the resulting DataFrame (row)
	hcat(df, df_params)
end


function results_dataframe_at_val(target::Array{String,1})
	j = 1
	df = []
	for i in 1:length(target)
		r = results_dataframe_row_at_val(target[i])
		if !isnothing(r)
			df = results_dataframe_row_at_val(target[i])
			break
		end
		j += 1
	end

    for i in j:length(target)
		r = results_dataframe_row_at_val(target[i])
		if !isnothing(r)
        	df = vcat(df, r, cols=:union)
		end
    end
    unique(df)
end
function results_dataframe_at_val(target::Array{String,1}, seeds)
	j = 1
	df = []
	for i in 1:length(target)
		r = results_dataframe_row_at_val_over_seeds(target[i]; seeds = seeds)
		if !isnothing(r)
			df = results_dataframe_row_at_val_over_seeds(target[i]; seeds = seeds)
			break
		end
		j += 1
	end

    for i in j:length(target)
		r = results_dataframe_row_at_val_over_seeds(target[i]; seeds = seeds)
		if !isnothing(r)
        	df = vcat(df, r, cols=:union)
		end
    end
    unique(df)
end

"""
    results_dataframe_at_val(target::String)

Collects all result files from the folder specified and returns a DataFrame
with calculated scores and parameters of the model (and data) based on number
of anomalous samples in validation data.
"""
function results_dataframe_at_val(target::String)
    files = GroupAD.Evaluation.collect_scores(target)
    results_dataframe_at_val(files)
end
function results_dataframe_at_val(target::String, seeds)
    files = GroupAD.Evaluation.collect_scores(target)
    results_dataframe_at_val(files, seeds)
end

function combined_dataframe_at_val(modelname, dataset)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset)
    data = results_dataframe_at_val(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
    params = merge(point[:parameters], (no_anomalous = nothing,))

    g = groupby(data, [keys(params)...])
    un = unique(map(x -> size(x), g))
    if length(un) != 1
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing groups with less than 6 seeds."
        g = g[findall(x -> size(x,1) > 5, g)]
    end

    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]
    cdf = combine(g, map(x -> x => mean, metricsnames))
end

"""
    find_best_model_at_val(modelname, dataset; metric=:val_AUC)

Finds the best model based on hyperparameters selection with fixed
number of anomalous samples in validation set. Returns the best model
for each `no_anomalous = [1,2,5,10,20,50]`.
"""
function find_best_model_at_val(modelname, dataset; metric=:val_AUC)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset)
    data = results_dataframe_at_val(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
    params = merge(point[:parameters], (no_anomalous = nothing,))

    g = groupby(data, [keys(params)...])
    un = unique(map(x -> size(x), g))
    if length(un) != 1
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing groups with less than 6 seeds."
        g = g[findall(x -> size(x,1) > 5, g)]
    end

    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]
    cdf = combine(g, map(x -> x => mean, metricsnames))
    sort!(cdf, :val_AUC_mean, rev=true)

    g2 = groupby(cdf, :no_anomalous)
    best_models = vcat(map(x -> DataFrame(x[1,:]), g2)...)
end

function find_best_model_at_val_over_seeds(modelname, dataset; metric=:val_AUC)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset)
    data = results_dataframe_at_val_over_seeds(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
    params = merge(point[:parameters], (no_anomalous = nothing,))

    g = groupby(data, [keys(params)...])
    un = unique(map(x -> size(x), g))
    if length(un) != 1
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing groups with less than 6 seeds."
        g = g[findall(x -> size(x,1) > 5, g)]
    end

    metricsnames = [:val_AUC, :val_AUPRC]
    cdf = combine(g, map(x -> x => mean, metricsnames))
    sort!(cdf, :val_AUC_mean, rev=true)

    g2 = groupby(cdf, :no_anomalous)
    best_models = vcat(map(x -> DataFrame(x[1,:]), g2)...)
end

"""
    best_model_files(best_models)

Given the dataframe of best models, returns the file names of these models.
"""
function best_model_files(best_models)
    pr = best_models[:, Not([:no_anomalous, :val_AUC_mean, :val_AUPRC_mean, :test_AUC_mean, :test_AUPRC_mean])]
    params = map(x -> Dict(names(x) .=> values(x)), eachrow(pr))
    files = map(x -> savename(x, "bson", digits=5), params)
    no_anomalous = best_models[:,:no_anomalous]
    return (files, no_anomalous)
end

function best_model_files_over_seeds(best_models)
    pr = best_models[:, Not([:no_anomalous, :val_AUC_mean, :val_AUPRC_mean])]
    params = map(x -> Dict(names(x) .=> values(x)), eachrow(pr))
    files = map(x -> savename(x, "bson", digits=5), params)
    no_anomalous = best_models[:,:no_anomalous]
    return (files, no_anomalous)
end

"""
    best_paths(modelname, dataset, files, no_anomalous)

Given the files of best models, returns the full path to the chosen models.
"""
function best_paths(modelname, dataset, files, no_anomalous)
    paths = []
    for f in files
        seed_paths = String[]
        for seed in 1:10
            path = joinpath(datadir("experiments", "contamination-0.0", modelname, dataset, "seed=$seed"), f)
            push!(seed_paths, path)
        end
        push!(paths, seed_paths)
    end
    return (paths, no_anomalous)
end

function combine_at_val(results, no_anomalous)
    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]

    combined_df = []
    for result in results
        cdf = combine(result, map(x -> x => mean, metricsnames))
        push!(combined_df, cdf)
    end
    hcat(vcat(combined_df...), DataFrame(:no_anomalous => no_anomalous))
end

"""
    evaluate_at_val(modelname::String, dataset)
    evaluate_at_val(modelnames::Array{String,1}, dataset)

Finds the best model based on hyperparameter selection via validation
dataset with given number of anomalies in validation set. Returns a results
dataframe calculated for each selected model on full validation set
averaged over seeds.
"""
function evaluate_at_val(modelname::String, dataset)
    best_models = find_best_model_at_val(modelname, dataset);
    files, no_anomalous = best_model_files(best_models);
    target, no_anomalous = best_paths(modelname, dataset, files, no_anomalous);
    results = GroupAD.Evaluation.results_dataframe.(target);

    combined_df = combine_at_val(results, no_anomalous)
end
function evaluate_at_val(modelnames::Array{String,1}, dataset)
    comb = DataFrame[]
    for modelname in modelnames
        best_models = find_best_model_at_val(modelname, dataset);
        files, no_anomalous = best_model_files(best_models);
        target, no_anomalous = best_paths(modelname, dataset, files, no_anomalous);
        results = GroupAD.Evaluation.results_dataframe.(target);

        combined_df = combine_at_val(results, no_anomalous)
        push!(comb, combined_df)
    end
    return comb
end

function evaluate_at_val_over_seeds(modelname::String, dataset)
    best_models = find_best_model_at_val_over_seeds(modelname, dataset)
    files, no_anomalous = best_model_files_over_seeds(best_models)
    target, no_anomalous = best_paths(modelname, dataset, files, no_anomalous)
    results = GroupAD.Evaluation.results_dataframe.(target)

    combined_df = combine_at_val(results, no_anomalous)
end
function evaluate_at_val_over_seeds(modelnames::Array{String,1}, dataset)
    comb = DataFrame[]
    for modelname in modelnames
        best_models = find_best_model_at_val_over_seeds(modelname, dataset);
        files, no_anomalous = best_model_files_over_seeds(best_models);
        target, no_anomalous = best_paths(modelname, dataset, files, no_anomalous);
        results = GroupAD.Evaluation.results_dataframe.(target);

        combined_df = combine_at_val(results, no_anomalous)
        push!(comb, combined_df)
    end
    return comb
end


function plot_at_val(df, modelname, dataset; savef = false, kwargs...)
    d = sort(df, :no_anomalous)
    p = plot(d[:, :no_anomalous], d[:, :val_AUC_mean], marker=:square, label="$modelname val", legend=:outerright; kwargs...)
    p = plot!(d[:, :no_anomalous], d[:, :test_AUC_mean], marker=:circle, label="$modelname test", legend=:outerright; kwargs...)
    if savef
        savefig(plotsdir("at_val_$(modelname)_$(dataset).png"))
    end
    return p
end
function plot_at_val(df, modelname, dataset, p; savef = false, only_test = false, kwargs...)
    d = sort(df, :no_anomalous)
    if !only_test
        p = plot!(d[:, :no_anomalous], d[:, :val_AUC_mean], marker=:square, label="$modelname val", legend=:outerright; kwargs...)
    end
    p = plot!(d[:, :no_anomalous], d[:, :test_AUC_mean], marker=:circle, label="$modelname test", legend=:outerright; kwargs...)
    if savef
        savefig(plotsdir("at_val_$(modelname)_$(dataset).png"))
    end
    return p
end

function plot_at_val(dfs::Array{DataFrame,1}, modelnames, dataset; kwargs...)
    p = plot(xlabel="# anomalies in validation", ylabel="AUC", legend=:outerright)
    for (i,df) in enumerate(dfs)
        p = plot_at_val(df, modelnames[i], dataset, p; color=i, kwargs...)
    end
    return p
end