using DrWatson
@quickactivate

using GroupAD
using GroupAD: load_data
using GroupAD.Models: unpack_mill

using LIBSVM
using StatsBase
using EvalMetrics

function group_kernel(γ, X::T, Y::T) where T <: AbstractMatrix
    mean(exp.(-pairwise(SqEuclidean(), X, Y) .* γ))
end

using Distances
using Distances: UnionMetric
import Distances: result_type

struct GroupGaussian <: UnionMetric
    γ::Number
end
(dist::GroupGaussian)(X, Y) = group_kernel(dist.γ, X, Y)
result_type(dist::GroupGaussian, x, y) = Float32

using ProgressMeter
using Distributions

mutable struct SMM
	γ::Float32
	c::Float32
	nu::Float32
	model::Union{LIBSVM.SVM, Nothing}
end

# SMM(;γ, nu) = SMM(γ, 0f0, nu, nothing)
SMM(;γ, c=0f0, nu) = SMM(γ, c, nu, nothing)

"""
    StatsBase.fit!(m::SMM, data::Tuple)

Function to fit SMM model.
"""
function StatsBase.fit!(m::SMM, data::Tuple)
    Xtrain, _ = unpack_mill(data[1])
    M = pairwise(GroupGaussian(m.γ), Xtrain)

    if m.c != 0
        ctrain = length.(Xtrain)
        kernel_ctrain = exp.(- m.c * pairwise(Cityblock(), ctrain))
        TK = M .* kernel_ctrain
        m.model = svmtrain(TK, kernel=Kernel.Precomputed; svmtype=OneClassSVM)
    else
        m.model = svmtrain(M, kernel=Kernel.Precomputed; svmtype=OneClassSVM)
    end
    m
end


# anomaly score functions

function score(m::SMM, Xtrain, Xtest)
    M = pairwise(GroupGaussian(m.γ), Xtrain, Xtest)

    if m.c != 0
        ctrain = length.(Xtrain)
        ctest = length.(Xtest)
        C = exp.(- m.c * pairwise(Cityblock(), ctrain, ctest))
        K = M .* C
        _, dec = svmpredict(m.model, K)
    else
        _, dec = svmpredict(m.model, M)
    end
    
    return .- dec[1,:]
end