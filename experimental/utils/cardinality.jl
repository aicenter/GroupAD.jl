using Plots, StatsPlots

"""
    cardinality_dist(data)

Returns Poisson and LogNormal cardinality distributions
fitted on data (MLE).
"""
function cardinality_dist(data)
    nvec = size.(data,2)
    poisson = fit_mle(Poisson,nvec)
    lognormal = fit_mle(LogNormal,nvec)
    return poisson, lognormal
end

"""
    cardinality_hist(normal,data,labels;label="")

Fits normal data with Poisson and LogNormal cardinality distribution.
Returns the cardinality plot of validation or test data with the fitted
cardinality distributions.
"""
function cardinality_hist(normal,data,labels;label="")
    # fit Poisson and LogNormal distributions to number of samples
    nvec = size.(normal,2)
    poisson = fit_mle(Poisson,nvec)
    lognormal = fit_mle(LogNormal,nvec)

    data_normal = data[labels .== 0]
    data_anomalous = data[labels .== 1]
    nvec_normal = size.(data_normal,2)
    nvec_anomalous = size.(data_anomalous,2)

    # plot the distributions with data histogram
    bw = maximum(nvec)÷2

    histogram(nvec_normal, normalized=true, opacity=0.3, label="$label normal",
                lw=1,color=:green,linecolor=:green,nbins=bw)
    histogram!(nvec_anomalous, normalized=true, opacity=0.3,label="$label anomalous",
                lw=1,color=:red,ilnecolor=:red,nbins=bw)

    plot!(poisson,lw=2,color=:blue,label="Poisson")
    p = plot!(lognormal,lw=2,color=:black,label="LogNormal")
    return poisson, lognormal, p
end

"""
    score_report(model, data, labels, fun [, L])

Returns a dataframe of binary_eval_report results from EvalMetrics.jl
for a chosen score function. If L is specified, returns the scores averaged
over L samples.
"""
function score_report(model, data, labels, fun)
    scores = [fun(model, bag) for bag in data];
    scores_p = [fun(model, bag, poisson) for bag in data];
    scores_ln = [fun(model, bag, lognormal) for bag in data];
    report = vcat([DataFrame(binary_eval_report(labels, scores)) for scores in [scores, scores_p, scores_ln]]...)
    print(hcat(["sum", "Poisson", "LogNormal"],report))
    return (scores, scores_p, scores_ln)
end
function score_report(model, data, labels, fun, L)
    scores = [mean([fun(model, bag) for _ in 1:L]) for bag in data]
    scores_p = [mean([fun(model, bag, poisson) for _ in 1:L]) for bag in data]
    scores_ln = [mean([fun(model, bag, lognormal) for _ in 1:L]) for bag in data]
    report = vcat([DataFrame(binary_eval_report(labels, scores)) for scores in [scores, scores_p, scores_ln]]...)
    print(hcat(["sum", "Poisson", "LogNormal"],report))
    return (scores, scores_p, scores_ln)
end