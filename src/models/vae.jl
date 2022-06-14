using Flux
using ConditionalDists
using GroupAD.GenerativeModels
import  GroupAD.GenerativeModels: VAE
using ValueHistories
using MLDataPattern: RandomBatches
using Distributions
using DistributionsAD
using StatsBase
using Random

"""
	safe_softplus(x::T)

Safe version of softplus.	
"""
safe_softplus(x::T) where T  = softplus(x) + T(0.000001)

"""
	init_vamp(pseudoinput_mean,k::Int)

Initializes the VAMP prior from a mean vector and number of components.
"""
function init_vamp(pseudoinput_mean, k::Int)
	T = eltype(pseudoinput_mean)
	pseudoinputs = T(1) .* randn(T, size(pseudoinput_mean)[1:end-1]..., k) .+ pseudoinput_mean
	VAMP(pseudoinputs)
end

"""
	vae_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers::Int=3, 
		init_seed=nothing, prior="normal", pseudoinput_mean=nothing, k=1, kwargs...)

Constructs a classical variational autoencoder.

# Arguments
	- `idim::Int`: input dimension.
	- `zdim::Int`: latent space dimension.
	- `activation::String="relu"`: activation function.
	- `hdim::Int=128`: size of hidden dimension.
	- `nlayers::Int=3`: number of decoder/encoder layers, must be >= 3. 
	- `init_seed=nothing`: seed to initialize weights.
	- `prior="normal"`: one of ["normal", "vamp"].
	- `pseudoinput_mean=nothing`: mean of data used to initialize the VAMP prior.
	- `k::Int=1`: number of VAMP components. 
	- `var="scalar"`: decoder covariance computation, one of ["scalar", "diag"].
"""
function vae_constructor(;idim::Int=1, zdim::Int=1, activation="relu", hdim=128, nlayers::Int=3, 
	init_seed=nothing, prior="normal", pseudoinput_mean=nothing, k=1, var="scalar", kwargs...)
	(nlayers < 3) ? error("Less than 3 layers are not supported") : nothing
	
	# if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing
	
	# construct the model
	# encoder - diagonal covariance
	encoder_map = Chain(
		build_mlp(idim, hdim, hdim, nlayers-1, activation=activation)...,
		ConditionalDists.SplitLayer(hdim, [zdim, zdim], [identity, safe_softplus])
		)
	encoder = ConditionalMvNormal(encoder_map)
	
	# decoder - we will optimize only a shared scalar variance for all dimensions
	if var=="scalar"
		decoder_map = Chain(
			build_mlp(zdim, hdim, hdim, nlayers-1, activation=activation)...,
			ConditionalDists.SplitLayer(hdim, [idim, 1], [identity, safe_softplus])
			)
	else
		decoder_map = Chain(
				build_mlp(zdim, hdim, hdim, nlayers-1, activation=activation)...,
				ConditionalDists.SplitLayer(hdim, [idim, idim], [identity, safe_softplus])
				)
	end
	decoder = ConditionalMvNormal(decoder_map)

	# prior
	if prior == "normal"
		prior_arg = zdim
	elseif prior == "vamp"
		(pseudoinput_mean === nothing) ? error("if `prior=vamp`, supply pseudoinput array") : nothing
		prior_arg = init_vamp(pseudoinput_mean, k)
	end

	# reset seed
	(init_seed !== nothing) ? Random.seed!() : nothing

	# get the vanilla VAE
	model = VAE(prior_arg, encoder, decoder)
end

"""
	StatsBase.fit!(model::VAE, data::Tuple, loss::Function; max_train_time=82800, lr=0.001, 
		batchsize=64, patience=30, check_interval::Int=10, kwargs...)
"""
function StatsBase.fit!(model::VAE, data::Tuple, loss::Function; max_iters=10000, max_train_time=82800, 
	lr=0.001, batchsize=64, patience=30, check_interval::Int=10, kwargs...)
	history = MVHistory()
	opt = ADAM(lr)

	tr_model = deepcopy(model)
	ps = Flux.params(tr_model)
	_patience = patience

	tr_x = data[1][1]
	# i know this could be done generally for all sizes but it is very ugly afaik
	if ndims(tr_x) == 2
		val_x = data[2][1][:,data[2][2] .== 0]
	elseif ndims(tr_x) == 4
		val_x = data[2][1][:,:,:,data[2][2] .== 0]
	else
		error("not implemented for other than 2D and 4D data")
	end
	val_N = size(val_x,ndims(val_x))

	# on large datasets, batching loss is faster
	best_val_loss = Inf
	i = 1
	start_time = time() # end the training loop after 23hrs
	for batch in RandomBatches(tr_x, batchsize)
		# batch loss
		batch_loss = 0f0
		gs = gradient(() -> begin 
			batch_loss = loss(tr_model,batch)
		end, ps)
	 	Flux.update!(opt, ps, gs)

		push!(history, :training_loss, i, batch_loss)
		if mod(i, check_interval) == 0
			
			# validation/early stopping
			val_loss = (val_N > 5000) ? loss(tr_model, val_x, 256) : loss(tr_model, val_x)
			
			@info "$i - loss: $(batch_loss) (batch) | $(val_loss) (validation)"
				
			if isnan(val_loss) || isnan(batch_loss)
				error("Encountered invalid values in loss function.")
			end

			push!(history, :validation_likelihood, i, val_loss)
			
			if val_loss < best_val_loss
				best_val_loss = val_loss
				_patience = patience

				# this should save the model at least once
				# when the validation loss is decreasing 
				model = deepcopy(tr_model)
			else # else stop if the model has not improved for `patience` iterations
				_patience -= 1
				if _patience == 0
					@info "Stopped training after $(i) iterations."
					break
				end
			end
		end
		if (time() - start_time > max_train_time) | (i > max_iters) # stop early if time is running out
			model = deepcopy(tr_model)
			@info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours."
			break
		end
		i += 1
	end
	# again, this is not optimal, the model should be passed by reference and only the reference should be edited
	(history=history, iterations=i, model=model, npars=sum(map(p->length(p), Flux.params(model))))
end

# likelihood functions
"""
	likelihood(model::VAE, bag)

Calculates the instance likelihoods for the VAE model.
"""
function likelihood(model::VAE, bag)
	p = condition(model.decoder, rand(model.encoder, bag))
	-logpdf(p, bag)
end
function likelihood(model::VAE, bag, L::Int)
    l = hcat([likelihood(model, bag) for _ in 1:L]...)
    return mean(l, dims=2)
end

function mean_likelihood(model::VAE, bag)
	p = condition(model.decoder, mean(model.encoder, bag))
	-logpdf(p, bag)
end

# anomaly score functions
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
	reconstruction_score(model::VAE, x::AbstractArray{T,2}[, L=1])
	reconstruction_score(model::VAE, x::Mill.BagNode, agf::Function[, L=1])

Anomaly score based on the reconstruction probability of the data. Support an aggregation function
in case x is a BagNode.
"""
function reconstruction_score(model::VAE, x::AbstractArray{T,2}) where T 
	p = condition(model.decoder, rand(model.encoder, x))
	-logpdf(p, x)
end
function reconstruction_score(model::VAE, x::AbstractArray{T,2}, L::Int) where T
	mean([reconstruction_score(model, x) for _ in 1:L])
end
function reconstruction_score(model::VAE, x::Mill.BagNode, agf::Function, args...)
	# aggregate x - bags to vectors
	_x = aggregate(x, agf)
	return reconstruction_score(model, _x, args...)
end

"""
	reconstruction_score_mean(model::VAE, x)
	reconstruction_score_mean(model::VAE, x::Mill.BagNode, agf)

Anomaly score based on the reconstruction probability of the data. Uses mean of encoding. 
Support an aggregation function in case x is a BagNode.
"""
function reconstruction_score_mean(model::VAE, x::AbstractArray{T,2}) where T 
	p = condition(model.decoder, mean(model.encoder, x))
	-logpdf(p, x)
end
function reconstruction_score_mean(model::VAE, x::Mill.BagNode, agf::Function)
	# aggregate x - bags to vectors
	_x = aggregate(x, agf)
	reconstruction_score_mean(model, _x)
end

"""
	latent_score(model::VAE, x[, L=1]) 
	latent_score(model::VAE, x::Mill.BagNode, agf::Function[, L=1])

Anomaly score based on the similarity of the encoded data and the prior.
"""
function latent_score(model::VAE, x::AbstractArray{T,2}) where T 
	z = rand(model.encoder, x)
	-logpdf(model.prior, z)
end
latent_score(model::VAE, x::AbstractArray{T,2}, L::Int) where T = 
	mean([latent_score(model, x) for _ in 1:L])
function latent_score(model::VAE, x::Mill.BagNode, agf::Function, args...)
	# aggregate x - bags to vectors
	_x = aggregate(x, agf)
	latent_score(model, _x, args...)
end

"""
	latent_score_mean(model::VAE, x) 
	latent_score_mean(model::VAE, x::Mill.BagNode, agf::Function)

Anomaly score based on the similarity of the encoded data and the prior. Uses mean of encoding.
"""
function latent_score_mean(model::VAE, x::AbstractArray{T,2}) where T 
	z = mean(model.encoder, x)
	-logpdf(model.prior, z)
end
function latent_score_mean(model::VAE, x::Mill.BagNode, agf::Function)
	# aggregate x - bags to vectors
	_x = aggregate(x, agf)
	latent_score_mean(model, _x)
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
