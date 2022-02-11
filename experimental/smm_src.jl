using LIBSVM
using GroupAD

using GroupAD: load_data
using GroupAD.Models: unpack_mill
using IPMeasures
using Flux3D: chamfer_distance
using EvalMetrics
using Statistics

# Define Chamfer and MMD as metrics to easily calculate
# pairwise distances
using Distances
using Distances: UnionMetric
import Distances: result_type
using IPMeasures: pairwisel2, AbstractKernel

struct Chamfer <: UnionMetric end
(dist::Chamfer)(x, y) = chamfer_distance(x, y)
result_type(dist::Chamfer, x, y) = Float32

(dist::MMD)(x, y) = (dist)(x, y)
result_type(dist::MMD, x, y) = Float32
MMD(k::AbstractKernel) = MMD(k, pairwisel2)

struct SMMModel
    distance::String
    kernel::String
    γ::Float32
    h::Float32
    nu::Float64
end

struct SMM
    distance
    h::Float32
    nu::Float64
    model
end

function SMM(dist::T, Xtrain::AbstractVector, h::Float32; nu::Float64 = 0.5) where T<:Union{MMD, Chamfer}
    DM = pairwise(dist, Xtrain)
    kernel_train = exp.(.- h .* DM)

    model = svmtrain(kernel_train, kernel=Kernel.Precomputed; svmtype=OneClassSVM, nu=nu)

    return SMM(dist, h, nu, model)
end
function SMM(dist::T, Xtrain::AbstractVector; nu::Float64 = 0.5) where T<:Union{MMD, Chamfer}
    DM = pairwise(dist, Xtrain)
    h = 1/median(DM)
    kernel_train = exp.(.- h .* DM)

    model = svmtrain(kernel_train, kernel=Kernel.Precomputed; svmtype=OneClassSVM, nu=nu)

    return SMM(dist, h, nu, model)
end

function predict(m::SMM, Xtrain, Xtest)
    DM = pairwise(m.distance, Xtrain, Xtest)
    kernel = exp.(.- m.h .* DM)
    pred, dec = svmpredict(m.model, kernel)
    return pred, dec
end

function Base.show(io::IO, m::SMM)
    nm = """
    OC-SMM with $(typeof(m.distance)) distance:
    (h = $(m.h), nu = $(m.nu))
    """
	print(io, nm)
end



"""
using Plots
ENV["GKSwstype"] = "100"

    scatter2(X, x=1, y=2; kwargs...)

Plots a 2D scatterplot from a matrix of type (n_features, n_samples).
Uses first 2 feature rows by default but different rows can be chosen
with parameters `x, y`.

function scatter2(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter(X[x,:],X[y,:]; label="", kwargs...)
end
function scatter2!(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter!(X[x,:],X[y,:]; label="", kwargs...)
end
"""

using UnicodePlots
scatterun2(X, x=1, y=2; color=:1) = scatterplot(X[x, :], X[y, :], marker=:circle, color=color)
scatterun2!(X, x=1, y=2; color=:1) = scatterplot!(X[x, :], X[y, :], marker=:circle, color=color)