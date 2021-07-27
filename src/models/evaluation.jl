### Provides evaluation functions which do not depend on the used model (VAE or Statistician)
using IPMeasures

"""
    calculate_logU(lh)

Calculates the estimated logU constant from instance likelihoods of training data.
"""
calculate_logU(lh) = mean(vcat(lh...))

"""
    chamfer_score(x, y)

Returns the Chamfer distance from Flux3D.jl
"""
chamfer_score(X,Y) = map((x, y) -> chamfer_distance(x, y), X, Y)

"""
    mmd_score(x, y)

Returns the MMD score.
"""
mmd_score(X, Y, kernel, bw) = map((x, y) -> mmd(kernel(bw), x, y), X, Y)

"""
    mmd_bandwidth(x)

Calculates the median distance of instances to estimate the bandwidth for
Gaussian and IMQ kernels in MMD. `x` should be training data.

If data is too large, only takes a 5000 instances as a subsample and
calculates the bandwidth only from the subset.
"""
function mmd_bandwidth(x)
    X = hcat(x...)
    sz = size(X, 2)
    # downsample if the number of instances is large
    n = 5000
    if sz > n
        # calculate the bandwidth for the kernel
        return bw = median(pairwise(Euclidean(), X[:,sample(1:sz,n)], dims=2))
    else
        return bw = median(pairwise(Euclidean(), X, dims=2))
    end
end


"""
    rec_score_from_likelihood(lh, fun::Function)

Returns reconstruction score from instance likelihoods
for individual bags. Default mode is sum. Used functions
should be (might be):
- sum (makes sense because of probability rules)
- mean
- maximum (might be better for those datasets where a bag is anomalous if at least
           one instance in the bag is anomalous)
"""
rec_score_from_likelihood(lh, sizes; fun=sum::Function) = map(x -> fun(x), lh)

"""
    rec_score_from_likelihood(lh, sizes, pc<:Distribution)

Returns reconstruction score from instance likelihoods
for individual bags using sum of instance likelihoods.
Corrected with the cardinality distribution.
"""
function rec_score_from_likelihood(lh, sizes, pc::T) where T <: Distribution
    s = sum.(lh)
    c = -logpdf(pc, sizes)
    return s .+ c
end

"""
    rec_score_from_likelihood(lh, sizes, logU::AbstractFloat)

Returns reconstruction score from instance likelihoods
for individual bags using sum of instance likelihoods.
Corrected with the logU constant.
"""
function rec_score_from_likelihood(lh, sizes, logU::AbstractFloat)
    s = sum.(lh)
    nU = logU * sizes
    return s .- nU
end

"""
    rec_score_from_likelihood(lh, sizes, pc<:Distribution, logU::AbstractFloat)

Returns reconstruction score from instance likelihoods
for individual bags using sum of instance likelihoods.
Corrected with both cardinality distribution and the logU constant.
"""
function rec_score_from_likelihood(lh, sizes, pc::T, logU::AbstractFloat) where T <: Distribution
    s = sum.(lh)
    c = -logpdf(pc, sizes)
    nU = logU * sizes
    return s .+ c .- nU
end