struct SplitLayer
    μ::Flux.Dense
    σ::Flux.Dense
end

Flux.@functor SplitLayer

function SplitLayer(in::Int, out::Vector{Int}, acts::Vector)
	SplitLayer(
		Flux.Dense(in, out[1], acts[1]),
		Flux.Dense(in, out[2], acts[2])
	)
end

function (m::SplitLayer)(x)
	return (m.μ(x), m.σ(x))
end

function Base.rand(m::ConditionalMvNormal, x::AbstractVecOrMat{T}) where T<:Real
    μ, Σ = m.mapping(x)
    z = μ .+ Σ .* randn(Float32, size(μ)...)
    return z
end

function logpdf(m::ConditionalMvNormal, x::AbstractArray, z::AbstractArray)
    # Z = -K/2 ( log (2*pi) + 2*log(det{Σ}) 
    # - ∑ᵏ(μ - x) * 1/sigma * (μ - x)
    μ, Σ = m.mapping(z)
    dim, bs = size(μ)
    exponent = Flux.sum(-0.5 .* (μ - x).^2 .* (1 ./ Σ), dims=1)
    scale_const = - (dim/2) *log(2*pi) .- 0.5 .* log.(prod(Σ, dims=1))
    return scale_const .+ exponent
end

function kl_divergence(p::BMN, q::TuMvNormal)
    (μ₁, σ₁) = Flux.mean(p), Flux.var(p)
    (μ₂, σ₂) = Flux.mean(q), Flux.var(q)
    _kld_gaussian1(μ₁, σ₁, μ₂, σ₂)
end

function kl_divergence(p::BMN, q::BMN)
    (μ₁, σ₁) = Flux.mean(p), Flux.var(p)
    (μ₂, σ₂) = Flux.mean(q), Flux.var(q)
    _kld_gaussian1(μ₁, σ₁, μ₂, σ₂)
end


function _kld_gaussian1(μ1::AbstractArray, σ1::AbstractArray, μ2::AbstractArray, σ2::AbstractArray)
    k  = size(μ1, 1)
    m1 = sum(σ1 ./ σ2, dims=1)
    m2 = sum((μ2 .- μ1).^2 ./ σ2, dims=1)
    m3 = sum(log.(σ2 ./ σ1), dims=1)
    (m1 .+ m2 .+ m3 .- k) ./ 2
end
