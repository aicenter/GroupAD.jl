struct VAE{P<:ContinuousMultivariateDistribution,E<:ConditionalMvNormal,D<:ConditionalMvNormal} <: AbstractVAE
    prior::P
    encoder::E
    decoder::D
end

Flux.@functor VAE

function Flux.trainable(m::VAE)
    (encoder=m.encoder, decoder=m.decoder)
end

function VAE(zlength::Int, enc::ConditionalMvNormal, dec::ConditionalMvNormal)
    W = first(Flux.params(enc))
    μ = fill!(similar(W, zlength), 0)
    σ = fill!(similar(W, zlength), 1)
    prior = DistributionsAD.TuringMvNormal(μ, σ)
    VAE(prior, enc, dec)
end

function elbo(m::AbstractVAE, x::AbstractArray; β=1)
    # sample latent
    z = rand(m.encoder, x) #custom function in utils.jl

    # reconstruction error
    llh = mean(logpdf(m.decoder, x, z))

    # KLD with `condition`ed encoder
    kld = mean(kl_divergence(condition(m.encoder, x), m.prior))

    llh - β*kld
end





