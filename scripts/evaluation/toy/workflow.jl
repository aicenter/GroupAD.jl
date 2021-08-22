"""
    find_best_model(modelname, dataset; metric=:val_AUC)

Returns the row of a results dataframe with the best result based on chosen
metric.
"""
function find_best_model(modelname, dataset; metric=:val_AUC)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset)
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
end

"""
    find_best_model_scores(modelname::String, dataset::String, scenario::Int; metric=:val_AUC)

Returns the rows of a results dataframe with the best result based on chosen
metric for all scores calculated for the model.
"""
function find_best_model_scores(modelname::String, dataset::String, scenario::Int; groupsymbol=:score, metric=:val_AUC)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset, "scenario=$scenario")
    data = GroupAD.Evaluation.results_dataframe(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
    params = point[:parameters]

    g_score = groupby(data, groupsymbol)
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


function combined_dataframe(modelname, dataset)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset)
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
end