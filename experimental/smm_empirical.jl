using DrWatson
@quickactivate

using GroupAD
using GroupAD: load_data
using GroupAD.Models: unpack_mill
using GroupAD.Models: Chamfer, MMD
# include("experimental/smm_src.jl")
using LIBSVM
using StatsBase
using EvalMetrics

using Plots, StatsPlots
ENV["GKSwstype"] = "100"
gr(label="",color=:jet)

# data = load_data("toy", 120, 120; scenario=2, seed=2053)
data = load_data("BrownCreeper", seed=2053)
# data = load_data("events_anomalydetection_v2.h5")

train, val, test = data
Xtrain, ytrain = unpack_mill(train)
Xval, yval = unpack_mill(val)
Xtest, ytest = unpack_mill(test)
x1 = Xtrain[1]
x2 = Xtrain[2]

using Distances
using Distances: UnionMetric
import Distances: result_type

function smm_bandwidth(data)
    m = median(pairwise(SqEuclidean(), data))
    return 1/(2m)
end

function group_kernel(γ, X::T, Y::T) where T <: AbstractMatrix
    mean(exp.(-pairwise(SqEuclidean(), X, Y) .* γ))
end

struct GroupGaussian <: UnionMetric
    γ::Number
end
(dist::GroupGaussian)(X, Y) = group_kernel(dist.γ, X, Y)
result_type(dist::GroupGaussian, x, y) = Float32

using ProgressMeter
using Distributions

function bandwidth_auc(dataset::String)
    data = load_data(dataset, seed=2053)
    train, val, test = data
    Xtrain, ytrain = unpack_mill(train)
    Xval, yval = unpack_mill(val)

    r1 = sort(rand(Uniform(0,1), 500))
    r2 = sort(rand(Uniform(1,30), 500))
    bd = smm_bandwidth(hcat(Xtrain...))

    r = sort(vcat(r1, r2, bd))
    AUC = []
    @showprogress for γ in r
        M = pairwise(GroupGaussian(γ), Xtrain)
        model = svmtrain(M, kernel=Kernel.Precomputed; svmtype=OneClassSVM)
        Mval = pairwise(GroupGaussian(γ), Xtrain, Xval)
        _, dec = svmpredict(model, Mval)
        auc = binary_eval_report(yval, -dec[1,:])["au_roccurve"]
        push!(AUC, auc)
    end
    return r, AUC, bd
end

# dataset = "Fox"

for dataset in mill_datasets[13:end]
    r, AUC, bd = bandwidth_auc(dataset)
    i = findmax(AUC)[2]
    bdi = findall(x -> x == bd, r)[1]
    bdauc = AUC[bdi]
    rmax, aucmax = r[i], AUC[i]
    p1 = plot(
            r[1:500], AUC[1:500],
            title="""
                dataset = $dataset
                max AUC         = $(round(aucmax, digits=3)) (at bandwidth = $(round(rmax,digits=3)))
                calculated AUC  = $(round(bdauc, digits=3)) (bandwidth = $(round(bd, sigdigits=3)))
                """,
            titlefontsize=10,
            label=""
    )
    p2 = plot(r[501:end], AUC[501:end], label="")
    p = plot(p1, p2, layout=(2, 1),size=(800,500))
    wsave("plots/bandwidth/$dataset.png", p)
end

# for dataset in mill_datasets
#     data = load_data(dataset, seed=2053)
#     train, val, test = data
#     Xtrain, ytrain = unpack_mill(train)
#     γ = smm_bandwidth(hcat(Xtrain...))

#     println("$dataset: $(round(γ, sigdigits=3))")
# end


γ = smm_bandwidth(hcat(Xtrain...))
M = pairwise(GroupGaussian(γ), Xtrain)
model = svmtrain(M, kernel=Kernel.Precomputed; svmtype=OneClassSVM)
Mval = pairwise(GroupGaussian(γ), Xtrain, Xval)
_, dec = svmpredict(model, Mval)

binary_eval_report(yval, -dec[1,:])

scatter(-dec[1,:], zcolor=yval .|> Int)
# savefig = Plots.savefig
savefig("plot.png")


# calculate cardinalities
ctrain = length.(Xtrain)
cval = length.(Xval)
ctest = length.(Xtest)

c = 1/median(pairwise(Cityblock(), ctrain))
kernel_ctrain = exp.(- c * pairwise(Cityblock(), ctrain))

M = pairwise(GroupGaussian(γ), Xtrain)
TK = M .* kernel_ctrain
model_c = svmtrain(TK, kernel=Kernel.Precomputed; svmtype=OneClassSVM)

kernel_cval = exp.(.- c .* pairwise(Cityblock(), ctrain, cval))
Mval = pairwise(GroupGaussian(γ), Xtrain, Xval)
VK = Mval .* kernel_cval

_, dec = svmpredict(model_c, VK)
binary_eval_report(yval, -dec[1,:])
scatter(-dec[1,:], zcolor=yval .|> Int)
savefig("plot.png")

mod = svmtrain(kernel_ctrain, kernel=Kernel.Precomputed; svmtype=OneClassSVM)
_, dec = svmpredict(mod, kernel_cval)
binary_eval_report(yval, dec[1,:])

cs = sort(rand(Uniform(0,0.2), 1000))

AUC = Float32[]
M = pairwise(GroupGaussian(γ), Xtrain)
Mval = pairwise(GroupGaussian(γ), Xtrain, Xval)

Ctrain = pairwise(Cityblock(), ctrain)
Cval = pairwise(Cityblock(), ctrain, cval)

@showprogress for c in cs
    kernel_ctrain = exp.(- c * Ctrain)
    
    TK = M .* kernel_ctrain
    model_c = svmtrain(TK, kernel=Kernel.Precomputed; svmtype=OneClassSVM)

    kernel_cval = exp.(.- c .* Cval)
    
    VK = Mval .* kernel_cval

    _, dec = svmpredict(model_c, VK)
    auc = binary_eval_report(yval, -dec[1,:])["au_roccurve"]
    push!(AUC, auc)
end

plot(cs, AUC)
aucmax, imax = findmax(AUC)
cmax = cs[imax]
scatter!([cmax], [aucmax], markersize=5,label="maximum",color=:green)

c = 1/median(pairwise(Cityblock(), ctrain))*2
    kernel_ctrain = exp.(- c * Ctrain)
    TK = M .* kernel_ctrain
    model_c = svmtrain(TK, kernel=Kernel.Precomputed; svmtype=OneClassSVM)
    kernel_cval = exp.(.- c .* Cval)
    VK = Mval .* kernel_cval
    _, dec = svmpredict(model_c, VK)
    auc = binary_eval_report(yval, -dec[1,:])["au_roccurve"]

scatter!([c], [auc], markersize=5,label="ideal*2",color=:red)

savefig("plot.png")
