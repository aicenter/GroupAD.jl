using LIBSVM
using IPMeasures
using Flux3D: chamfer_distance
using Statistics
using Distributions

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

# unfitted SMM model - just a container for parameters
struct SMMModel
    distance::String
    kernel::String
    γ::Float32
    h::Float32
    nu::Float64
end

# fitted SMM model
struct SMM
    distance
    h::Float32
    nu::Float64
    model
end

function Base.show(io::IO, m::SMM)
    nm = """
    OC-SMM with $(typeof(m.distance)) distance:
    (h = $(m.h), nu = $(m.nu))
    """
	print(io, nm)
end


"""
SMM(dist::T, Xtrain::AbstractVector [, h::Float32]; nu::Float64 = 0.5) where T<:Union{MMD, Chamfer}

Fits a SMM model with either MMD or Chamfer distance on training data. Uses pairwise
calculation of distances to calculate the kernel matrix and then LIBSVM.SVM (one-class version).

If `h` is not specified, calculates the 'optimal' value as `h = 1/median(DM)` where `DM` is the
calculated distance matrix. 
"""
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


"""
    StatsBase.fit!(m::SMMModel, data::Tuple)

Function to fit SMM model.
"""
function StatsBase.fit!(m::SMMModel, data::Tuple)

    Xtrain, _ = unpack_mill(data[1])

    if m.kernel == "none"
        distance = Chamfer()
    elseif m.kernel == "Gaussian"
        distance = MMD(GaussianKernel(m.γ))
    elseif m.kernel == "IMQ"
        distance = MMD(IMQKernel(m.γ))
    end

    model = SMM(distance, Xtrain, m.h; nu = m.nu)
end

# anomaly score functions

function StatsBase.score(m::SMM, Xtrain, Xtest)
    DM = pairwise(m.distance, Xtrain, Xtest)
    kernel = exp.(.- m.h .* DM)
    _, dec = svmpredict(m.model, kernel)
    return dec[1,:]
end
function predictions(m::SMM, Xtrain, Xtest)
    DM = pairwise(m.distance, Xtrain, Xtest)
    kernel = exp.(.- m.h .* DM)
    pred, _ = svmpredict(m.model, kernel)
    return pred
end