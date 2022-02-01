using Flux
using ConditionalDists, Distributions
using MLDataPattern: RandomBatches
using StatsBase
using Random
using Mill

"""
PoolModel is a generative model which reconstructs and generates
output from a single vector summary of the input set.

PoolModel has 7 components:
- prepool_net
- postpool_net
- poolf
- prior
- encoder
- generator
- decoder

Pre-pool and post-pool nets are neural networks. The former is responsible for the
transformation of all instances, the latter only transforms the one-vector summary.
The summary is created with a pooling function which has to be permutation invariant.
Possible functions include: mean, sum, maximum, etc.
"""
struct PoolModel{pre <: Chain, post <: Chain, fun <: Function, p <: ContinuousMultivariateDistribution, e <: ConditionalMvNormal, g <: ConditionalMvNormal, d <: Chain}
    prepool_net::pre
    postpool_net::post
    poolf::fun
    prior::p
    encoder::e
    generator::g
    decoder::d
end

Flux.@functor PoolModel

function Flux.trainable(m::PoolModel)
    (prepool_net = m.prepool_net, postpool_net = m.postpool_net, encoder = m.encoder, generator = m.generator, decoder = m.decoder)
end

function PoolModel(pre, post, fun, gen, dec, enc::ConditionalMvNormal, plength::Int)
    W = first(Flux.params(enc))
    μ = fill!(similar(W, plength), 0)
    σ = fill!(similar(W, plength), 1)
    prior = DistributionsAD.TuringMvNormal(μ, σ)
    PoolModel(pre, post, fun, prior, enc, gen, dec)
end

function Base.show(io::IO, pm::PoolModel)
    nm = "PoolModel($(pm.poolf))"
	print(io, nm)
end


"""
    pm_constructor(;idim, hdim, predim, postdim, edim, activation="swish", nlayers=3, var="scalar", fun=sum_stat)

Constructs a PoolModel. Some input dimensions are automatically calculated based on the chosen
pooling function.

Dimensions:
- idim: input dimension
- hdim: hidden dimension in all networks
- predim: the input dimension of pooling function
- postdim: the output dimension of post-pool network and input dimension of encoder and generator
- edim: output dimension of encoder and generator, input dimension to decoder
"""
function pm_constructor(;idim, hdim, predim, postdim, edim, activation="swish", nlayers=3, var="scalar",
    poolf=sum_stat, init_seed=nothing, kwargs...)

    fun = eval(:($(Symbol(poolf))))

    # if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing

    # pre-pool network
    pre = Chain(
        build_mlp(idim,hdim,hdim,nlayers-1,activation=activation)...,
        Dense(hdim,predim)
    )
    # dimension after pooling
    pooldim = length(fun(randn(predim)))
    # post-pool network
    post = Chain(
        build_mlp(pooldim,hdim,hdim,nlayers-1,activation=activation)...,
        Dense(hdim,postdim)
    )
    
    if var == "scalar"
    # encoder
        enc = Chain(
            build_mlp(postdim,hdim,hdim,nlayers-1,activation=activation)...,
            SplitLayer(hdim,[edim,1])
        )
        enc_dist = ConditionalMvNormal(enc)

        gen = Chain(
            build_mlp(postdim,hdim,hdim,nlayers-1,activation=activation)...,
            SplitLayer(hdim,[edim,1])
        )
        gen_dist = ConditionalMvNormal(gen)
    else
        enc = Chain(
            build_mlp(postdim,hdim,hdim,nlayers-1,activation=activation)...,
            SplitLayer(hdim,[edim,edim])
        )
        enc_dist = ConditionalMvNormal(enc)

        gen = Chain(
            build_mlp(postdim,hdim,hdim,nlayers-1,activation=activation)...,
            SplitLayer(hdim,[edim,edim])
        )
        gen_dist = ConditionalMvNormal(gen)
    end

    dec = Chain(
        build_mlp(edim,hdim,hdim,nlayers-1,activation=activation)...,
        Dense(hdim,idim)
    )

    pm = PoolModel(pre, post, fun, gen_dist, dec, enc_dist, edim)
    return pm
end

#################################
### Special pooling functions ###
#################################

bag_mean(x) = mean(x, dims=2)
bag_maximum(x) = maximum(x, dims=2)

"""
    mean_max(x)

Concatenates mean and maximum.
"""
function mean_max(x)
    m1 = mean(x, dims=2)
    m2 = maximum(x, dims=2)
    return vcat(m1,m2)
end

"""
    mean_max_card(x)

Concatenates mean, maximum and set cardinality.
"""
function mean_max_card(x)
    m1 = mean(x, dims=2)
    m2 = maximum(x, dims=2)
    return vcat(m1,m2,size(x,2))
end

"""
    sum_stat(x)

Calculates a summary vector as a concatenation of mean, maximum, minimum, and var pooling.
"""
function sum_stat(x)
    m1 = mean(x, dims=2)
    m2 = maximum(x, dims=2)
    m3 = minimum(x, dims=2)
    m4 = var(x, dims=2)
    if any(isnan.(m4))
        m4 = zeros(length(m1))
    end
    return vcat(m1,m2,m3,m4)
end

function sum_stat_card(x)
    m1 = mean(x, dims=2)
    m2 = maximum(x, dims=2)
    m3 = minimum(x, dims=2)
    m4 = var(x, dims=2)
    if any(isnan.(m4))
        m4 = zeros(length(m1))
    end
    return vcat(m1,m2,m3,m4,size(x,2))
end

"""
    pm_loss(m::PoolModel, x)

Loss function for the PoolModel. Based on Chamfer distance.
"""
function pm_loss(m::PoolModel, x)
    # pre-pool network transformation of X
    v = m.prepool_net(x)
    # pooling
    p = m.poolf(v)
    # post-pool
    p_post = m.postpool_net(p)
    z = hcat([rand(m.generator, p_post) for i in 1:size(x, 2)]...)
    dz = m.decoder(z)

    return chamfer_distance(x, dz)
end

"""
StatsBase.fit!(model::MGMM, data::Tuple, loss::Function; max_train_time=82800, lr=0.001, 
    batchsize=64, patience=30, check_interval::Int=10, kwargs...)

Function to fit MGMM model.
"""
function StatsBase.fit!(model::PoolModel, data::Tuple, loss::Function;
	max_iters=10000, max_train_time=82800, lr=0.001, batchsize=64, patience=30,
	check_interval::Int=10, kwargs...)

    history = MVHistory()
    opt = ADAM(lr)

    tr_model = deepcopy(model)
    ps = Flux.params(tr_model)
    _patience = patience

    # prepare data for bag model
    tr_x, tr_l = unpack_mill(data[1])
    vx, vl = unpack_mill(data[2])
    val_x = vx[vl .== 0]

    best_val_loss = Inf
    i = 1
    start_time = time()

    lossf(x) = loss(tr_model, x)

    # infinite for loop via RandomBatches
	for batch in RandomBatches(tr_x, 10)
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

######################################
### Score functions and evaluation ###
######################################

"""
    reconstruct(m::PoolModel, x)

Reconstructs the input bag.
"""
function reconstruct(m::PoolModel, x)
    v = m.prepool_net(x)
    p = m.poolf(v)
    p_post = m.postpool_net(p)
    z = hcat([rand(m.generator, p_post) for i in 1:size(x, 2)]...)
    m.decoder(z)
end

"""
    pool_encoding(m::PoolModel, x; post=true)

Returns the one-vector summary encoding for a bag.
If `post=true`, takes the bag through pre-pool network,
pooling function and post-pool network. If `post=false`,
skips the post-pool network transformation.
"""
function pool_encoding(m::PoolModel, x; post=true)
    v = m.prepool_net(x)
    p = m.poolf(v)
    if post
        return m.postpool_net(p)
    else
        return p
    end
end