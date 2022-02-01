using Random
using StatsBase
using BSON
using Plots
ENV["GKSwstype"] = "100"

"""
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
"""

function results_dataframe_row_at_val_over_seeds(target::String; verb = false, seeds = 50)
    data = load(target)
    no_anomalies = Int(sum(data[:val_labels]))
    val_an = [0, 1, 2, 5, 10, 20, 50, 100]
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
        # calculate the AUC score on validation data for given number of anomalies
        # average the AUC over seeds

        # check that scores are valid
        if all(isnan.(scores))
            if verb
                @info "score is NaN"
            end
            return
        end

        if no == 0
            # clean validation
            # only returns the mean score on normal data in validation
            clean_scores = scores[labels .== 0]
            # here the smaller the score the better (negative log-likelihood)
            # but the larger AUC the better
            AUC = -mean(clean_scores)
        else
            # for number of anomalies, samples indexes and gets AUC for validation data
            # with limited number of anomalies
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
        end
        # push the result to results vector
        push!(results_val, [no, mean(AUC)])
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
    insertcols!(
        df,
        :test_AUC => test_auc,
        :dataset => d[:dataset],
        :model => d[:modelname],
        :seed => d[:seed]
        )

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

function find_best_models_at_val(folder::String, seeds::Int)
    d = results_dataframe_at_val(folder, seeds)
    point = load(GroupAD.Evaluation.collect_scores(joinpath(folder, "seed=1"))[1])
    params = point[:parameters]

    gdf = groupby(d, :no_anomalous)
    gdf_combined = DataFrame[]
    for data in gdf
        g = groupby(data, [keys(params)...])
        un = unique(map(x -> size(x), g))
        if length(un) != 1
            idx = findall(x -> size(x,1) > 5, g)
            @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
            Removing $(length(g) - length(idx)) groups out of $(length(g)) with less than 6 seeds."
            g = g[idx]
        end

        metricsnames = [:val_AUC, :test_AUC]
        cdf = combine(g, map(x -> x => mean, metricsnames))
        sort!(cdf, :val_AUC_mean, rev=true)
        insertcols!(cdf, :no_anomalous => g[1][1,:no_anomalous])
        push!(gdf_combined, cdf)
    end

    _best_models = map(x -> DataFrame(x[1,:]), gdf_combined)
    best_models = vcat(_best_models...)
end

function plot_at_val(df::DataFrame)
    p = plot(df[:, :no_anomalous], df[:, :val_AUC_mean], marker=:square, label="val")
    p = plot!(df[:, :no_anomalous], df[:, :test_AUC_mean], marker=:circle, label="test", legend=:outerright)
end
function plot_at_val_test(df::DataFrame, label::String)
    p = plot(df[:, :no_anomalous], df[:, :test_AUC_mean], marker=:circle, label=label, legend=:outerright)
end
function plot_at_val_test!(df::DataFrame, label::String, p)
    p = plot!(df[:, :no_anomalous], df[:, :test_AUC_mean], marker=:circle, label=label, legend=:outerright)
end

# test
folder = datadir("experiments", "contamination-0.0", "vae_basic", "BrownCreeper")
best_models = find_best_models_at_val(folder, 20)
plot_at_val_test(best_models, "VAEagg")
savefig(plotsdir("validation", "BrownCreeper_vaeagg_test.png"))

results_at_validation = Dict()
results_at_validation = load(datadir("dataframes", "at_validation", "mill_validation.bson"))

for d in mill_datasets
    @info "Starting computation for $d dataset."
    p = plot()
    dres = Dict()
    for modelname in modelnames
        folder = datadir("experiments", "contamination-0.0", modelname, d)
        best_models = find_best_models_at_val(folder, 50)
        @info "Best $modelname found."
        push!(dres, modelname => best_models)
        p = plot_at_val_test!(best_models, modelname, p)
    end
    p
    #savefig(plotsdir("validation", "all", "$(d).png"))
    wsave(plotsdir("validation", "all", "$(d).png"), p)
    push!(results_at_validation, d => dres)
    wsave(datadir("dataframes", "at_validation", "mill_validation.bson"), results_at_validation)
end

results_at_validation0 = Dict()
results_at_validation0 = load(datadir("dataframes", "at_validation", "mill_validation0.bson"))

for d in mill_datasets[2:end]
    @info "Starting computation for $d dataset."
    p = plot()
    dres = Dict()
    for modelname in modelnames
        folder = datadir("experiments", "contamination-0.0", modelname, d)
        best_models = find_best_models_at_val(folder, 50)
        @info "Best $modelname found."
        push!(dres, modelname => best_models)
        p = plot_at_val_test!(best_models, modelname, p)
    end
    p
    #savefig(plotsdir("validation", "all", "$(d).png"))
    wsave(plotsdir("validation", "all_0", "$(d).png"), p)
    push!(results_at_validation0, d => dres)
    wsave(datadir("dataframes", "at_validation", "mill_validation0.bson"), results_at_validation0)
end