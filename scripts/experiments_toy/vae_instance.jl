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
	"type"
        default = 1
        arg_type = Int
        help = "type of toy dataset"
   "contamination"
        default = 0.0
        arg_type = Float64
        help = "training data contamination rate"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, type, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "vae_instance"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	par_vec = (2 .^(1:3), 2 .^(2:4), ["scalar", "diagonal"], 10f0 .^(-4:-3), 2 .^ (5:7), ["relu", "swish", "tanh"], 3:4, 1:Int(1e8))
	argnames = (:zdim, :hdim, :var, :lr, :batchsize, :activation, :nlayers, :init_seed)
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

"""
    train_val_data(data::Tuple)

Make the data trainable via the basic vae model and its fit function.
"""
function train_val_data(data::Tuple)
    train, val, test = data

    train_new = (hcat(train[1]...), train[2])
    V = hcat(val[1]...)
    nv_all = size(V,2)
    nv = size(hcat(val[1][val[2] .== 0]...),2)
    val_new = (V, vcat(zeros(Float32,nv),ones(Float32,nv_all-nv)))
    return (train_new, val_new)
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
	# get only the data needed and unpack them from bags to intances
    instance_data = train_val_data(data)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, instance_data, loss; max_train_time=10*3600/max_seed, 
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

	# now return the info to be saved and an array of tuples (anomaly score function, hyperparatemers)
	L=100
	training_info, [
		(x -> GroupAD.Models.likelihood(info.model,x), 
			merge(parameters, (score = "reconstruction",))),
		(x -> GroupAD.Models.mean_likelihood(info.model,x), 
			merge(parameters, (score = "reconstruction-mean",))),
		(x -> GroupAD.Models.likelihood(info.model,x,L), 
			merge(parameters, (score = "reconstruction-sampled", L=L))),
		(x -> GroupAD.Models.reconstruct(info.model,x), 
			merge(parameters, (score = "reconstructed_input",)))
	]
end

"""
	edit_params(data, parameters)
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters, class, method)
	merge(parameters, (scenario = type, ))
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	GroupAD.toy_experimental_loop(
		sample_params, 
		fit, 
		edit_params, 
		max_seed, 
		type, 
		modelname, 
		dataset,
		datadir("experiments/contamination-$(contamination)")
		)
end
