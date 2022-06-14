using LIBSVM
using IPMeasures
using Flux3D: chamfer_distance
using Statistics
using Distributions
using Distances

# unfitted SMMC model - just a container for parameters
struct SMMCModel
    distance::String
    kernel::String
    γ::Float32
    h::Float32
    c::Float32
    nu::Float64
end

# fitted SMMCModel model
struct SMMC
    distance
    h::Float32
    c::Float32
    nu::Float64
    model
end

function Base.show(io::IO, m::SMMC)
    nm = """
    OC-SMMC with $(typeof(m.distance)) distance:
    (h = $(m.h), c = $(m.c), nu = $(m.nu))
    """
	print(io, nm)
end


"""
    SMMC(dist::T, Xtrain::AbstractVector [, h::Float32]; nu::Float64 = 0.5) where T<:Union{MMD, Chamfer}

Fits a SMM cardinality model with either MMD or Chamfer distance on training data. Uses pairwise
calculation of distances to calculate the kernel matrix and then LIBSVM.SVM (one-class version).

If `h` is not specified, calculates the 'optimal' value as `h = 1/median(DM)` where `DM` is the
calculated distance matrix. 
"""
function SMMC(dist::T, Xtrain::AbstractVector, h::Float32, c::Float32; nu::Float64 = 0.5) where T<:Union{MMD, Chamfer}
    DM = pairwise(dist, Xtrain)
    CM = pairwise(TotalVariation(), length.(Xtrain))

    kernel_data = exp.(.- h .* DM)
    kernel_card = exp.(.- c .* CM)
    kernel_train = kernel_data .* kernel_card

    model = svmtrain(kernel_train, kernel=Kernel.Precomputed; svmtype=OneClassSVM, nu=nu)

    return SMMC(dist, h, c, nu, model)
end
function SMMC(dist::T, Xtrain::AbstractVector; nu::Float64 = 0.5) where T<:Union{MMD, Chamfer}
    DM = pairwise(dist, Xtrain)
    h = 1/median(DM)
    CM = pairwise(TotalVariation(), length.(Xtrain))
    c = 1/median(CM)

    kernel_data = exp.(.- h .* DM)
    kernel_card = exp.(.- c .* CM)
    kernel_train = kernel_data .* kernel_card

    model = svmtrain(kernel_train, kernel=Kernel.Precomputed; svmtype=OneClassSVM, nu=nu)

    return SMMC(dist, h, c, nu, model)
end


"""
    StatsBase.fit!(m::SMMCModel, data::Tuple)

Function to fit SMMC model.
"""
function StatsBase.fit!(m::SMMCModel, data::Tuple)

    Xtrain, _ = unpack_mill(data[1])

    if m.kernel == "none"
        distance = Chamfer()
    elseif m.kernel == "Gaussian"
        distance = MMD(GaussianKernel(m.γ))
    elseif m.kernel == "IMQ"
        distance = MMD(IMQKernel(m.γ))
    end

    model = SMMC(distance, Xtrain, m.h, m.c; nu = m.nu)
end

# anomaly score functions

function StatsBase.score(m::SMMC, Xtrain, Xtest)
    DM = pairwise(m.distance, Xtrain, Xtest)
    CM = pairwise(TotalVariation(), length.(Xtrain), length.(Xtest))

    kernel_data = exp.(.- m.h .* DM)
    kernel_card = exp.(.- m.c .* CM)
    kernel = kernel_data .* kernel_card
    
    _, dec = svmpredict(m.model, kernel)
    # the decision boundary needs to be swapped
    return .- dec[1,:]
end