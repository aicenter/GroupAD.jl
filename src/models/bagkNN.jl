using StatsBase
using IPMeasures

struct BagkNNModel
    distance::String
    kernel::String
    γ::Float32
    k::Int
end

struct BagkNN
    distance
    k::Int
    Xtrain
end

function Base.show(io::IO, m::BagkNN)
    nm = """
    Bag $(m.k)-NN model with $(typeof(m.distance)) distance.
    """
	print(io, nm)
end

"""
    StatsBase.fit!(m::BagkNN, data::Tuple)

Function to create BagkNN model.
"""
function StatsBase.fit!(m::BagkNNModel, data::Tuple)

    Xtrain, _ = unpack_mill(data[1])

    if m.kernel == "none"
        distance = Chamfer()
    elseif m.kernel == "Gaussian"
        distance = MMD(GaussianKernel(m.γ))
    elseif m.kernel == "IMQ"
        distance = MMD(IMQKernel(m.γ))
    end

    model = BagkNN(distance, m.k, Xtrain)
end

function distance_matrix(m::BagkNN, Xtest)
    M = pairwise(m.distance, Xtest, m.Xtrain)
    sort!(M, dims=2)
end

function StatsBase.score(m::BagkNN, dm::AbstractMatrix, v::V) where {V<:Val{:kappa}}
    return dm[:, m.k]
end
function StatsBase.score(m::BagkNN, dm::AbstractMatrix, v::V) where {V<:Val{:gamma}}
    return mean(dm[:, 1:m.k], dims=2)[:]
end
StatsBase.score(m::BagkNN, dm::AbstractMatrix, v::Symbol) = StatsBase.score(m, dm, Val(v))
StatsBase.score(m::BagkNN, dm::AbstractMatrix, v::String) = StatsBase.score(m, dm, Val(Symbol(v)))

# test it
# m = BagkNNModel("MMD", "Gaussian", 0.1f0, 10)
# m = BagkNNModel("Chamfer", "none", 0.1f0, 10)
# model = fit!(m, data)

# dm = distance_matrix(model, unpack_mill(data[2])[1])
# score(model, dm, :gamma)