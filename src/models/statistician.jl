using Flux
using ConditionalDists
using GenerativeModels
import GenerativeModels:NeuralStatistician
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
    nlayers::Int=3,activation="relu",init_seed=nothing, kwargs...)

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
	dec = Chain(
		build_mlp(zdim, hdim, hdim, nlayers - 1, activation=activation)...,
		SplitLayer(hdim, [idim,1], [identity,safe_softplus])
		)
	dec_dist = ConditionalMvNormal(dec)

    # reset seed
	(init_seed !== nothing) ? Random.seed!() : nothing

    # get NeuralStatistician model
	model = NeuralStatistician(instance_enc, cdim, enc_c_dist, cond_z_dist, enc_z_dist, dec_dist)
end

"""
	unpack_mill(dt)

Takes Tuple of BagNodes and bag labels and returns
both in a format that is fit for Flux.train!
"""
function unpack_mill(dt)
    bag_labels = dt[2]
	bag_data = [dt[1][i].data.data for i in 1:length(bag_labels)]
    return bag_data, bag_labels
end


"""
    RandomBagBatches(data;batchsize::Int=32,randomize=false)

Creates random batch for bag data which are an array of
arrays.
"""
function RandomBagBatches(data;batchsize::Int=32,randomize=false)
    l = length(data)
	if batchsize > l
		return data
	end
    idx = sample(1:l-batchsize)
    if randomize
        return shuffle(data)[idx:idx+batchsize-1]
    else
        return data[idx:idx+batchsize-1]
    end
end


"""
	StatsBase.fit!(model::NeuralStatistician, data::Tuple, loss::Function; max_train_time=82800, lr=0.001, 
		batchsize=64, patience=30, check_interval::Int=10, kwargs...)
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
	@info "Number of training bags: $(length(data[1][1]))"
	tr_x, tr_l = unpack_mill(data[1])
	vx, vl = unpack_mill(data[2])
	val_x = vx[vl .== 0]

	best_val_loss = Inf
	i = 1
	start_time = time()

	lossf(x) = loss(tr_model, x)

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
		bag_batch = RandomBagBatch(tr_x,batchsize=batchsize,randomize=true)
		Flux.train!(lossf, ps, bag_batch, opt)
		train_loss = mean([lossf(x) for x in tr_x])

    		push!(history, :training_loss, i, train_loss)
		if mod(i, check_interval) == 0
			
			# validation/early stopping
			val_loss = mean([lossf(x) for x in val_x])
			
			@info "$i - loss: $(train_loss) (batch) | $(val_loss) (validation)"

			if isnan(val_loss) || isnan(train_loss)
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
	reconstruct(model::NeuralStatistician, bag)

Data reconstruction for NeuralStatistician.
Data must be bags!
"""
function reconstruct(model::NeuralStatistician, bag)
	v = model.instance_encoder(bag)
	p = mean(v, dims=2)
	c = rand(model.encoder_c, p)
	h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
	z = rand(model.encoder_z, h)
	mean(model.decoder, z)
end

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
	likelihood(model::NeuralStatistician, bag)

Calculates likelihood of a single bag.
"""
function likelihood(model::NeuralStatistician, bag)
    v = model.instance_encoder(bag)
	p = mean(v,dims=2)
	c = rand(model.encoder_c, p)
	h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
	z = rand(model.encoder_z, h)
    llh = -logpdf(model.decoder, bag, z)
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