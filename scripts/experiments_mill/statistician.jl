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
        default = "Fox"
        arg_type = String
        help = "dataset"
   "contamination"
        default = 0.0
        arg_type = Float64
        help = "training data contamination rate"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "statistician"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
For NeuralStatistician, latent dimensions cdim and zdim should be smaller
or equal to hidden dimension:
- `cdim` <= `hdim`
- `vdim` <= `hdim`
- `zdim` <= `hdim`
"""
function sample_params()
	par_vec = (2 .^(4:9), 2 .^(2:7), 2 .^(2:7), 2 .^(2:7), ["scalar", "diagonal"], 10f0 .^(-4:-3), 3:4, 2 .^(5:7), ["relu", "swish", "tanh"], 1:Int(1e8))
	argnames = (:hdim, :vdim, :cdim, :zdim, :var, :lr, :nlayers, :batchsize, :activation, :init_seed)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	
	# ensure that vdim, zdim, cdim <= hdim
	while parameters.vdim >= parameters.hdim
		parameters = merge(parameters, (vdim = sample(par_vec[2]),))
	end
	while parameters.cdim >= parameters.hdim
		parameters = merge(parameters, (cdim = sample(par_vec[3]),))
	end
	while parameters.zdim >= parameters.hdim
		parameters = merge(parameters, (zdim = sample(par_vec[4]),))
	end
	return parameters
end

"""
	loss(model::GenerativeModels.NeuralStatistician, x)

Negative ELBO for training of a Neural Statistician model.
"""
loss(model::GenerativeModels.NeuralStatistician,x) = -GroupAD.Models.elbo1(model, x)

(m::KLDivergence)(p::ConditionalDists.BMN, q::ConditionalDists.BMN) = IPMeasures._kld_gaussian(p,q)

"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GroupAD.Models.statistician_constructor(;idim=size(data[1][1],1), parameters...)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss; max_train_time=82800/max_seed, 
			patience=100, check_interval=1, parameters...)
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

	# now return the info to be saved and an array of tuples (anomaly score function, hyperparatemers)
	L=100
	return training_info, [
		(x -> GroupAD.Models.likelihood(info.model,x), 
			merge(parameters, (score = "reconstruction",))),
		(x -> GroupAD.Models.mean_likelihood(info.model,x), 
			merge(parameters, (score = "reconstruction-mean",))),
		(x -> GroupAD.Models.likelihood(info.model,x,L), 
			merge(parameters, (score = "reconstruction-sampled", L=L))),
		(x -> GroupAD.Models.reconstruct_input(info.model, x),
			merge(parameters, (score = "reconstructed_input",)))
	]
end

"""
	edit_params(data, parameters)
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters)
	parameters
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	GroupAD.basic_experimental_loop(
		sample_params, 
		fit, 
		edit_params, 
		max_seed, 
		modelname, 
		dataset, 
		contamination, 
		datadir("experiments/contamination-$(contamination)")
		)
end
