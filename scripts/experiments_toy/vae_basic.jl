using DrWatson
@quickactivate
using ArgParse
using GroupAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using GenerativeModels
using Distributions

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
        arg_type = Int
        help = "seed"
        default = 1
    "dataset"
        default = "toy"
        arg_type = String
        help = "dataset"
	"scenario"
        default = 1
        arg_type = Int
        help = "scenario of toy dataset"
   "contamination"
        default = 0.0
        arg_type = Float64
        help = "training data contamination rate"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, scenario, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "vae_basic"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	par_vec = ([1,2,3], 2 .^(2:4), ["scalar", "diagonal"], 10f0 .^(-4:-3), 2 .^ (5:7), ["relu", "swish", "tanh"], 3:4, 1:Int(1e8),
		["mean", "maximum", "median"])
	argnames = (:zdim, :hdim, :var, :lr, :batchsize, :activation, :nlayers, :init_seed, :aggregation)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	# ensure that zdim < hdim
	while parameters.zdim >= parameters.hdim
		parameters = merge(parameters, (zdim = sample(par_vec[1])[1],))
	end
	return parameters
end

"""
	loss(model::GenerativeModels.VAE, x[, batchsize])

Negative ELBO for training of a VAE model.
"""
loss(model::GenerativeModels.VAE, x) = -elbo(model, x)
# version of loss for large datasets
loss(model::GenerativeModels.VAE, x, batchsize::Int) = 
	mean(map(y->loss(model,y), Flux.Data.DataLoader(x, batchsize=batchsize)))


function aggregate_toy(data, agf::Function)
    m = mean.(data, dims=2)
    hcat(m...)
end

aggregate_toy(data::Tuple, agf::Function) = Tuple(map(d->(aggregate_toy(d[1], agf), d[2]), data))

function reconstruction_score(model::VAE, x::AbstractArray, agf::Function, args...)
	# aggregate x - bags to vectors
	_x = aggregate_toy(x, agf)
	return GroupAD.Models.reconstruction_score(model, _x, args...)
end

function reconstruction_score_mean(model::VAE, x::AbstractArray, agf::Function)
	# aggregate x - bags to vectors
	_x = aggregate_toy(x, agf)
	GroupAD.Models.reconstruction_score_mean(model, _x)
end

"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GroupAD.Models.vae_constructor(;idim=2, parameters...)

	# aggregate bags into vectors
	# first convert the aggregation string to a function
	agf = getfield(StatsBase, Symbol(parameters.aggregation))
	data = aggregate_toy(data, agf)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss; max_train_time=10*3600/max_seed, 
			patience=200, check_interval=10, parameters...)
	catch e
		# return an empty array if fit fails so nothing is computed
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		history = info.history,
		npars = info.npars,
		model = info.model
		)

	# now return the infor to be saved and an array of tuples (anomaly score function, hyperparatemers)
	L=100
	training_info, [
		(x -> reconstruction_score(info.model,x,agf), 
			merge(parameters, (score = "reconstruction",))),
		(x -> reconstruction_score_mean(info.model,x,agf), 
			merge(parameters, (score = "reconstruction-mean",))),
		(x -> reconstruction_score(info.model,x,agf,L), 
			merge(parameters, (score = "reconstruction-sampled", L=L)))		
	]
end

"""
	edit_params(data, parameters)
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters, scenario)
	merge(parameters, (scenario = scenario, ))
end

@info "Starting experimental loop."

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	GroupAD.toy_experimental_loop(
		sample_params, 
		fit, 
		edit_params, 
		max_seed, 
		scenario, 
		modelname, 
		dataset,
		datadir("experiments/contamination-$(contamination)")
		)
end