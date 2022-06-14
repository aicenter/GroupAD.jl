using Flux
using Zygote
using DistributionsAD
using ConditionalDists: ContinuousMultivariateDistribution, ConditionalMvNormal, condition
#include("utils.jl")
include("GenerativeModels.jl")
#include("vae.jl")

T = Float32
xlen = 4
zlen = 2
batch = 20
data = randn(T, 4, batch)/100 .+ hcat(ones(T,xlen,Int(batch/2)), -ones(T,xlen,Int(batch/2)))

enc = Chain(Dense(xlen, xlen, relu),
            Dense(xlen, xlen, relu),
            Dense(xlen, xlen, relu),
            Main.GenerativeModels.SplitLayer(xlen, [zlen,zlen], [identity,softplus]))
enc_dist = ConditionalMvNormal(enc)

dec = Chain(Dense(zlen, xlen, relu),
            Dense(xlen, xlen, relu),
            Dense(xlen, xlen, relu),
            Main.GenerativeModels.SplitLayer(xlen, [xlen,1], [identity,softplus]))
dec_dist = ConditionalMvNormal(dec)

model = Main.GenerativeModels.VAE(zlen, enc_dist, dec_dist)

tmp = Zygote.pullback(Flux.params(enc)) do
    Flux.mean(enc(data[1])[1])
end

Main.GenerativeModels.kl_divergence(condition(model.encoder, data), model.prior)

tmp = Zygote.pullback(Flux.params(model.encoder)) do
    Flux.mean(Main.GenerativeModels.kl_divergence(condition(model.encoder, data), model.prior))
end