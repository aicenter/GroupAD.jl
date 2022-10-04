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
using DistributionsAD, Flux, GroupAD
using GroupAD.GenerativeModels
using ConditionalDists
using ProgressMeter

"""
    compute_stats(row::DataFrameRow)

Calculates validation and test AUC/AUPRC values from DataFrameRow.
If scores are NaN, returns zeros.
"""
function compute_stats(row::DataFrameRow)
    if typeof(row[:tst_scores]) <: BitVector
        return (0.0,0.0,0.0,0.0)
    end
	scores_labels = [(row[:val_scores], row[:val_labels]), (row[:tst_scores], row[:tst_labels])]
	setnames = ["validation", "test"]

	results = []
	for (scores, labels) in scores_labels
		if all(isnan.(scores))
			# @info "score is NaN"
			# return (NaN, NaN, NaN, NaN)
            return (0.0,0.0,0.0,0.0)
		end
		scores = vec(scores) .|> Float32
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

    # decide on metric
    metric == :val_AUC ? test_metric = :test_AUC : test_metric = :test_AUPRC

    for g in gdf
		# find the maximum value based on `metric`
        vauc, ix = findmax(g[:, metric])
        tauc = g[ix, test_metric]

        # export parameters
        p = g.parameters[ix]
        pdf = DataFrame(keys(p) .=> values(p))

        push!(res, [g.dataset[1] values(g.parameters[ix])... round(vauc, digits=3) round(tauc, digits=3)])       
    end

	# create a resulting dataframe
	df = DataFrame(vcat(res...), [:dataset, p_names..., metric, test_metric])
    sort(df, :dataset)
end

mill_datasets_wo_Web = [
    "BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2",
    "Mutagenesis1", "Mutagenesis2", "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein",
    "Tiger", "UCSBBreastCancer", "WinterWren"
]

"""
    collect_mill(model::String, mill_datasets=mill_datasets)

Collects the results from all folders of MIL datasets using multi-threading.
"""
function collect_mill(model::String, mill_datasets=mill_datasets)
    len = length(mill_datasets)
    dfs = repeat([DataFrame()], len)
    Threads.@threads for i in 1:len
        # _df = collect_results(datadir("experiments", "contamination-0.0", model, "MIL", mill_datasets[i]), subfolders=true)
        # _df = collect_results(datadir("experiments", "contamination-0.0_old_data", model, mill_datasets[i]), subfolders=true)
        _df = collect_results(datadir("experiments", "contamination-0.0", "MIL", model, mill_datasets[i]), subfolders=true, rexclude=[r"model_.*"])
        dfs[i] = _df
    end
    return vcat(dfs...)
end

"""
    collect_mill(model::String, mill_datasets=mill_datasets)

Collects the results from LHCO using multi-threading.

*Note: It is recommended to use the same number of threads as the number of seeds.*
"""
function collect_lhco(model::String, dataset="events_anomalydetection_v2.h5")
    dir = readdir(datadir("experiments", "contamination-0.0", "LHCO", model, dataset), join=true)
    len = length(dir)
    dfs = repeat([DataFrame()], len)
    Threads.@threads for i in 1:len
        _df = collect_results(dir[i], subfolders=true, rexclude=[r"model_.*"])
        dfs[i] = _df
    end
    return vcat(dfs...)
end

function collect_mvtec(model::String, datasets=mvtec_datasets)
    len = length(mvtec_datasets)
    dfs = repeat([DataFrame()], len)
    Threads.@threads for i in 1:len
        _df = collect_results(datadir("experiments", "contamination-0.0", "mv_tec", model, datasets[i]), subfolders=true, rexclude=[r"model_.*"])
        dfs[i] = _df
    end
    return vcat(dfs...)
end

"""
    calculate_results(model::String; dataset::String="MIL", metric::Symbol=:val_AUC, show=false, tf=tf_unicode, filter_fun=nothing, max_seed=10)

Collects results for given model, filters only models with completed at least `max_seed` runs over the seeds.
Returns a grouped dataframe, where groups are dataset results aggregated over seeds.

Uses parallel processes for collecting results and calculating scores.
"""
function calculate_results(model::String; dataset::String="MIL", metric::Symbol=:val_AUC, show=false, tf=tf_unicode, filter_fun=nothing, max_seed=10)
    # load results collection
    if dataset == "MIL"
        df = collect_mill(model)
    elseif dataset == "LHCO"
        df = collect_lhco(model)
    elseif dataset == "mvtec"
        df = collect_mvtec(model)
    end
    @info "Data loaded."
    # filter out model files (for vae, statistician...) - not needed with the newest DrWatson's collect_results rexclude
    # df = filter(:path => x -> !occursin("model", x), _df)
    # other filtering, if needed
    if !isnothing(filter_fun)
        df = filter_fun(df)
    end

    # preallocate and calculate metrics
    n = nrow(df)
    metrics = zeros(n, 4)
    p = Progress(n, 1)
    Threads.@threads for i in 1:n
        row = df[i,:]
        metrics[i, :] .= compute_stats(row)
        next!(p)
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
    b = map(i -> nrow(g[i]) >= max_seed, 1:k)
    g = g[b]

    # get mean values of metrics over seeds
    cdf = combine(g, ["val_AUC", "val_AUPRC", "test_AUC", "test_AUPRC"] .=> mean, renamecols=false)

    # group by dataset
    g2 = groupby(cdf, :dataset)
    return g2
end

"""
    mill_model_results(model::String; metric::Symbol=:val_AUC, show=false, tf=tf_unicode)

Collects all results for MIL datasets from given model. Calculates the validation and test
AUC and AUPRC metrics, finds the best model based on results on validation metric `metric`
for each MIL dataset. Returns a DataFrame with metrics, parameters and dataset name.

If `show=true`, prints a PrettyTable with the specified `tf` formatting.
"""
function mill_model_results(model::String; metric::Symbol=:val_AUC, show=false, tf=tf_unicode, filter_fun=nothing)
    # load results and create a grouped dataframe
    g2 = calculate_results(model, dataset="MIL", metric=metric, show=show, tf=tf, filter_fun=filter_fun)
    # find the best model based on metric (validation AUC)
    R = findmaxs(g2, metric)

    # reorder columns
    c = ncol(R)
    R2 = R[:, vcat([1,c-1,c], setdiff(1:c, [1,c,c-1]))]

    # create a pretty table
    if show
        pretty_table(R2, nosubheader=true, tf = tf)
    end

    return R2, g2
end

function lhco_model_results(model::String; metric::Symbol=:val_AUC, show=false, tf=tf_unicode, filter_fun=nothing, max_seed=5)
    g2 = calculate_results(model, dataset="LHCO", metric=metric, show=show, tf=tf, filter_fun=filter_fun, max_seed=max_seed)
    R = findmaxs(g2, metric)

    # reorder columns
    c = ncol(R)
    R2 = R[:, vcat([1,c-1,c], setdiff(1:c, [1,c,c-1]))]

    # create a pretty table
    if show
        pretty_table(R2, nosubheader=true, tf = tf)
    end
    R2, g2[1]
end

function mvtec_model_results(model::String; metric::Symbol=:val_AUC, show=false, tf=tf_unicode, filter_fun=nothing, max_seed=5)
    # load results and create a grouped dataframe
    g2 = calculate_results(model, dataset="mvtec", metric=metric, show=show, tf=tf, filter_fun=filter_fun, max_seed=max_seed)
    # find the best model based on metric (validation AUC)
    R = findmaxs(g2, metric)

    # reorder columns
    c = ncol(R)
    R2 = R[:, vcat([1,c-1,c], setdiff(1:c, [1,c,c-1]))]

    # create a pretty table
    if show
        pretty_table(R2, nosubheader=true, tf = tf)
    end

    return R2, g2
end

function results_all_models(dataset::String; models = ["knn_basic", "vae_basic", "vae_instance", "statistician", "PoolModel"],
                            metric::Symbol=:val_AUC, show=false, tf=tf_unicode, filter_fun=nothing, max_seed=5)
    PT = []

    for model in models
        g = calculate_results(model, dataset=dataset, metric=metric, show=show, tf=tf, filter_fun=filter_fun, max_seed=max_seed)
        R = findmaxs(g, metric)
        c = ncol(R)
        R2 = R[:, vcat([1,c-1,c], setdiff(1:c, [1,c,c-1]))]
        p = hcat(DataFrame(:modelname => model), R2)
        push!(PT, p)
    end

    map(x -> pretty_table(x, nosubheader=true, tf=tf), PT)
end