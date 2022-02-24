# test how the SVM decision scores work

using DrWatson
@quickactivate

using GroupAD
using GroupAD.Models: SMMModel, SMM

model = SMMModel("Chamfer", "none", 1f0, 1f0, 0.5)
data = GroupAD.load_data("toy", 300, 300; scenario=1, seed=2053)
data = GroupAD.load_data("toy", 300, 300; scenario=2, seed=2053)
data = GroupAD.load_data("toy", 300, 300; scenario=3, seed=2053)

using StatsBase
m = StatsBase.fit!(model, data)

targets = data[2][2]
scores = score(m, data[1][1], data[2][1])
p = GroupAD.Models.predictions(m, data[1][1], data[2][1])

using EvalMetrics
binary_eval_report(targets, p .|> Float64)
binary_eval_report(targets, scores)

using UnicodePlots
scatterplot(scores, marker=".", color=targets .+ 2 .|> Int, width=200, height=20)