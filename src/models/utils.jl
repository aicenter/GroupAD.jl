import GenerativeModels: VAE

"""
	function build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; activation::String="relu", lastlayer::String="")

Creates a chain with `nlayers` layers of `hdim` neurons with transfer function `activation`.
input and output dimension is `idim` / `odim`
If lastlayer is no specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense.
It is also possible to specify dimensions in a vector.

```juliadoctest
julia> build_mlp(4, 11, 1, 3, activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp([4, 11, 11, 1], activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp(4, 11, 1, 3, activation="relu", lastlayer="tanh")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, tanh))
```
"""
build_mlp(ks::Vector{Int}, fs::Vector) = Flux.Chain(map(i -> Dense(i[2],i[3],i[1]), zip(fs,ks[1:end-1],ks[2:end]))...)

build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; kwargs...) =
	build_mlp(vcat(idim, fill(hdim, nlayers-1)..., odim); kwargs...)

function build_mlp(ks::Vector{Int}; activation::String = "relu", lastlayer::String = "")
	activation = (activation == "linear") ? "identity" : activation
	fs = Array{Any}(fill(eval(:($(Symbol(activation)))), length(ks) - 1))
	if !isempty(lastlayer)
		fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
	end
	build_mlp(ks, fs)
end

"""
	reconstruct(model::VAE, x)

Data reconstruction.
"""
reconstruct(model::VAE, x) = mean(model.decoder, rand(model.encoder, x))

"""
	generate(model::VAE, N::Int[, outdim])

Data generation. Support output dimension if the output needs to be reshaped, e.g. in convnets.
"""
generate(model::VAE, N::Int) = mean(model.decoder, rand(model.prior, N))
generate(model::VAE, N::Int, outdim) = reshape(generate(model, N), outdim..., :)

"""
	encode_mean(model::VAE, x)

Produce data encodings.
"""
encode_mean(model::VAE, x) = mean(model.encoder, x)

"""
	reconstruction_score(model::VAE, x[, L=1])

Anomaly score based on the reconstruction probability of the data.
"""
function reconstruction_score(model::VAE, x) 
	p = condition(model.decoder, rand(model.encoder, x))
	-logpdf(p, x)
end
#reconstruction_score(model::VAE, x, L::Int) = 
#	mean([reconstruction_score(model, x) for _ in 1:L])
function reconstruction_score(model::VAE, x, L::Int)
	println("asdasd $L")
	mean([reconstruction_score(model, x) for _ in 1:L])
end
"""
	reconstruction_score_mean(model::VAE, x)

Anomaly score based on the reconstruction probability of the data. Uses mean of encoding.
"""
function reconstruction_score_mean(model::VAE, x) 
	p = condition(model.decoder, mean(model.encoder, x))
	-logpdf(p, x)
end
"""
	latent_score(model::VAE, x[, L=1]) 

Anomaly score based on the similarity of the encoded data and the prior.
"""
function latent_score(model::VAE, x) 
	z = rand(model.encoder, x)
	-logpdf(model.prior, z)
end
latent_score(model::VAE, x, L::Int) = 
	mean([latent_score(model, x) for _ in 1:L])

"""
	latent_score_mean(model::VAE, x) 

Anomaly score based on the similarity of the encoded data and the prior. Uses mean of encoding.
"""
function latent_score_mean(model::VAE, x) 
	z = mean(model.encoder, x)
	-logpdf(model.prior, z)
end

"""
	batched_score(model,score,x,L,batchsize)

Batched score for large datasets.
"""
batched_score(model,score,x,batchsize,args...) = 
	vcat(map(y-> Base.invokelatest(score, model, y, args...), Flux.Data.DataLoader(x, batchsize=batchsize))...)


# JacoDeco score
# see https://arxiv.org/abs/1905.11890
"""
	jacobian(f, x)

Jacobian of f given due to x.
"""
function jacobian(f, x)
	y = f(x)
	n = length(y)
	m = length(x)
	T = eltype(y)
	j = Array{T, 2}(undef, n, m)
	for i in 1:n
		j[i, :] .= gradient(x -> f(x)[i], x)[1]
	end
	return j
end

"""
	lJacoD(m,x)

Jacobian decomposition JJ(m,x).
"""
function lJacoD(m,x)
	JJ = zeros(eltype(x),size(x,ndims(x)));
	zg = mean(m.encoder,x);
	for i=1:size(x,ndims(x))
		jj,J = jacobian(y->mean(m.decoder,reshape(y,:,1))[:],zg[:,i]);
		(U,S,V) = svd(J);
		JJ[i]= sum(2*log.(S));
	end
	JJ
end

"""
	lpx(m,x)

p(x|g(x))
"""
lpx(m,x) = logpdf(m.decoder,x,mean(m.encoder,x))

"""
	lpz(m,x)

p(z|e(x))
"""
lpz(m,x) = logpdf(m.prior,mean(m.encoder,x)) # 

"""
	lp_orthD(m,x)

JacoDeco score: p(x|g(x)) + p(z|e(x)) - JJ(m,x)
"""
jacodeco(m,x) = (lpx(m,x) .+ lpz(m,x) .- lJacoD(m,x));
