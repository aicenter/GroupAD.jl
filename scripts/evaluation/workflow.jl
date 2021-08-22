"""
    find_best_model(modelname, dataset; metric=:val_AUC)

Returns the row of a results dataframe with the best result based on chosen
metric.
"""
function find_best_model(modelname, dataset; metric=:val_AUC)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset)
    data = GroupAD.Evaluation.results_dataframe(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
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

function combined_dataframe(modelname, dataset)
    folder = datadir("experiments", "contamination-0.0", modelname, dataset)
    data = GroupAD.Evaluation.results_dataframe(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
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