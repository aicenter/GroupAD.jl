using Flux
using ConditionalDists
using GroupAD.GenerativeModels
import GroupAD.GenerativeModels: NeuralStatistician, elbo
using ValueHistories
using MLDataPattern: RandomBatches
using Distributions
using DistributionsAD
using StatsBase
using Random

"""
    statistician_constructor(;idim::Int,hdim::Int,vdim::Int,cdim::Int,zdim::Int,
        nlayers::Int=3,activation::String="relu",init_seed=nothing)

Constructs basic NeuralStatistician model.

# Arguments
    - `idim::Int`: input dimension
    - `hdim::Int`: size of hidden dimension
    - `vdim::Int`: feature vector dimension
    - `cdim::Int`: context dimension
    - `zdim::Int`: dimension on latent over instances
    - `nlayers::Int=3`: number of layers in model networks, must be >= 3
    - `activation::String="relu"`: activation function
    - `init_seed=nothing`: seed to initialize weights
"""
function statistician_constructor(;idim::Int,hdim::Int,vdim::Int,cdim::Int,zdim::Int,
    var::String="scalar", nlayers::Int=3,activation="relu",init_seed=nothing, kwargs...)

    (nlayers < 3) ? error("Less than 3 layers are not supported") : nothing

    # if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing

    # construct the model
    # instance encoder
	instance_enc = Chain(
		build_mlp(idim, hdim, hdim, nlayers - 1, activation=activation)...,
		Dense(hdim, vdim)
	)
	
    # context encoder q(c|D)
	enc_c = Chain(
		build_mlp(vdim, hdim, hdim, nlayers - 1, activation=activation)...,
		SplitLayer(hdim, [cdim,cdim], [identity,safe_softplus])
		)
	enc_c_dist = ConditionalMvNormal(enc_c)

    # conditional p(z|c)
	cond_z = Chain(
		build_mlp(cdim, hdim, hdim, nlayers - 1, activation=activation)...,
		SplitLayer(hdim, [zdim,zdim], [identity,safe_softplus])
		)
	cond_z_dist = ConditionalMvNormal(cond_z)

    # latent instance encoder q(z|c,x)
	enc_z = Chain(
		build_mlp(cdim + vdim, hdim, hdim, nlayers - 1, activation=activation)...,
		SplitLayer(hdim, [zdim,zdim], [identity,safe_softplus])
		)
	enc_z_dist = ConditionalMvNormal(enc_z)

    # decoder
	if var == "scalar"
		dec = Chain(
			build_mlp(zdim,hdim,hdim,nlayers-1,activation=activation)...,
			SplitLayer(hdim, [idim,1], [identity,softplus])
			)
		dec_dist = ConditionalMvNormal(dec)
	else
		dec = Chain(
			build_mlp(zdim,hdim,hdim,nlayers-1,activation=activation)...,
			SplitLayer(hdim, [idim,idim], [identity,softplus])
			)
		dec_dist = ConditionalMvNormal(dec)
	end

    # reset seed
	(init_seed !== nothing) ? Random.seed!() : nothing

    # get NeuralStatistician model
	model = NeuralStatistician(instance_enc, cdim, enc_c_dist, cond_z_dist, enc_z_dist, dec_dist)
end

"""
    elbo1(m::NeuralStatistician,x::AbstractArray; β1=1.0, β2=1.0)
Neural Statistician log-likelihood lower bound.
For a Neural Statistician model, simply create a loss
function as
    
    `loss(x) = -elbo(model,x)`
where `model` is a NeuralStatistician type.
The β terms scale the KLDs:
* β1: KL[q(c|D) || p(c)]
* β2: KL[q(z|c,x) || p(z|c)]

This function uses a speed-up concatenation of `v` and `c` which
allocates less memory and should be more efficient.
"""
function elbo1(m::NeuralStatistician, x::AbstractArray;β1=1.0,β2=1.0)
    # instance network
    v = m.instance_encoder(x)
    p = mean(v, dims=2)

    # sample latent for context
    c = rand(m.encoder_c, p)
	C = reshape(repeat(c, size(v,2)),size(c,1),size(v,2))

    # sample latent for instances
    h = vcat(v,C)
    z = rand(m.encoder_z, h)
	
    # 3 terms - likelihood, kl1, kl2
    llh = mean(logpdf(m.decoder, x, z))
    kld1 = mean(kl_divergence(condition(m.encoder_c, v), m.prior_c))
    kld2 = mean(kl_divergence(condition(m.encoder_z, h), condition(m.conditional_z, c)))
    llh - β1 * kld1 - β2 * kld2
end

"""
	StatsBase.fit!(model::NeuralStatistician, data::Tuple, loss::Function; max_train_time=82800, lr=0.001, 
		batchsize=64, patience=30, check_interval::Int=10, kwargs...)

Function to fit NeuralStatistician model.
"""
function StatsBase.fit!(model::NeuralStatistician, data::Tuple, loss::Function;
	max_iters=10000, max_train_time=82800, lr=0.001, batchsize=64, patience=30,
	check_interval::Int=10, kwargs...)
	
	history = MVHistory()
	opt = ADAM(lr)

	tr_model = deepcopy(model)
	ps = Flux.params(tr_model)
	_patience = patience
    
	# prepare data for Neural Statistician
	tr_x, tr_l = unpack_mill(data[1])
	vx, vl = unpack_mill(data[2])
	val_x = vx[vl .== 0]

	best_val_loss = Inf
	i = 1
	start_time = time()

	lossf(x) = loss(tr_model, x)

	# infinite for loop via RandomBatches
	for batch in RandomBatches(tr_x, 10)
		"""
		Neural Statistician models doesn't support minibatches
		as basic VAE.

		batch_loss = 0f0
		gs = gradient(() -> begin 
			batch_loss = loss(tr_model,batch)
		end, ps)
	 	Flux.update!(opt, ps, gs)
		"""
		# classic training
		bag_batch = RandomBagBatches(tr_x,batchsize=batchsize,randomize=true)
		Flux.train!(lossf, ps, bag_batch, opt)
		# only batch training loss
		batch_loss = mean(lossf.(bag_batch))

    	push!(history, :training_loss, i, batch_loss)
		if mod(i, check_interval) == 0
			
			# validation/early stopping
			val_loss = mean(lossf.(val_x))
			
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
				# @info "Patience is: $_patience."
				if _patience == 0
					@info "Stopped training after $(i) iterations."
					break
				end
			end
		end
		if (time() - start_time > max_train_time) | (i > max_iters) # stop early if time is running out
			model = deepcopy(tr_model)
			@info "Stopped training after $(i) iterations, $((time() - start_time) / 3600) hours."
			break
		end
		i += 1
	end
	# again, this is not optimal, the model should be passed by reference and only the reference should be edited
	(history = history, iterations = i, model = model, npars = sum(map(p -> length(p), Flux.params(model))))
end

# anomaly score functions
"""
	reconstruct_input(model::NeuralStatistician, bag)

Data reconstruction for NeuralStatistician.
Data must be bags!
"""
function reconstruct_input(model::NeuralStatistician, bag)
	v = model.instance_encoder(bag)
	p = mean(v, dims=2)
    c = rand(model.encoder_c, p)
    C = reshape(repeat(c, size(v,2)),size(c,1),size(v,2))
    h = vcat(v,C)
	z = rand(model.encoder_z, h)
	mean(model.decoder, z)
end

"""
	likelihood(model::NeuralStatistician, bag, [L])

Calculates likelihood of a single bag. If L is provided,
returns the sampled likelihood (mean).
"""
function likelihood(model::NeuralStatistician, bag)
    v = model.instance_encoder(bag)
	p = mean(v,dims=2)
    c = rand(model.encoder_c, p)
    C = reshape(repeat(c, size(v,2)),size(c,1),size(v,2))
    h = vcat(v,C)
	z = rand(model.encoder_z, h)
    -logpdf(model.decoder, bag, z)
end
function likelihood(model::NeuralStatistician, bag, L::Int)
    l = hcat([likelihood(model, bag) for _ in 1:L]...)
    return mean(l, dims=2)
end

"""
	mean_likelihood(model::NeuralStatistician, bag)

Calculates the mean likelihood of a bag.
"""
function mean_likelihood(model::NeuralStatistician, bag)
    v = model.instance_encoder(bag)
    p = mean(v,dims=2)
	c = rand(model.encoder_c, p) # should there be mean?
    C = reshape(repeat(c, size(v,2)),size(c,1),size(v,2))
    h = vcat(v,C)
    z = mean(model.encoder_z, h)
    -logpdf(model.decoder, bag, z)
end



##################
### Deprecated ###
##################

"""
	encode_context(model::NeuralStatistician, bag)

Produce bag mean encoding to context.
"""
function encode_mean(model::NeuralStatistician, bag)
	v = model.instance_encoder(bag)
	p = mean(v, dims=2)
	c = mean(model.encoder_c, p)
end

"""
	reconstruction_score(model::NeuralStatistician, ...)

Returns anomaly score based on reconstruction probability
for a whole bag.
"""
function reconstruction_score(model::NeuralStatistician, bag::AbstractArray)
	llh = likelihood(model, bag)
    sum(llh)
end
function reconstruction_score(model::NeuralStatistician, bag::AbstractArray, L::Int)
	mean([reconstruction_score(model,bag) for _ in 1:L])
end
function reconstruction_score(model::NeuralStatistician, x::Mill.BagNode, args...)
	[reconstruction_score(model, x[i].data.data, args...) for i in 1:length(x)]
end

"""
	reconstruction_score_mean(model::NeuralStatistician, bag)

Returns anomaly score based on reconstruction probability
when using mean encoding for both context and instance
latent representation.
"""
function reconstruction_score_mean(model::NeuralStatistician, bag::AbstractArray)
	v = model.instance_encoder(bag)
	p = mean(v,dims=2)
	c = mean(model.encoder_c, p)
	h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
	z = mean(model.encoder_z, h)
    llh = -logpdf(model.decoder, bag, z)
	sum(llh)
end
function reconstruction_score_mean(model::NeuralStatistician, bag::AbstractArray, L::Int)
	mean([reconstruction_score_mean(model,bag) for _ in 1:L])
end
function reconstruction_score_mean(model::NeuralStatistician, x::Mill.BagNode, args...)
	[reconstruction_score_mean(model, x[i].data.data, args...) for i in 1:length(x)]
end

"""
	latent_score_mean(model::NeuralStatistician, bag)

Returns anomaly score based on the similarity of the
encoded data and the prior. Uses mean encoding of context.
"""
function latent_score_mean(model::NeuralStatistician, bag)
	v = model.instance_encoder(bag)
	p = mean(v,dims=2)
	c = mean(model.encoder_c, p)
	-logpdf(model.prior_c, c)
end

"""
	batched_score(model,score,x,L,batchsize)

Batched score for large datasets. Prepared for bags.
"""
function batched_bag_score(model, score, data, batchsize, args...)
	l = length(data)
	ids = vcat(1:batchsize:l,l+1)
	vcat(map(i->Base.invokelatest(score, model,data[ids[i]:ids[i+1]-1], args...),1:length(ids)-1)...)
end