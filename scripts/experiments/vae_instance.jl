using DrWatson
@quickactivate
using ArgParse
using GroupAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using GenerativeModels

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
modelname = "vae_instance"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	par_vec = (2 .^(3:8), 2 .^(4:9), 10f0 .^(-4:-3), 2 .^ (5:7), ["relu", "swish", "tanh"], 3:4, 1:Int(1e8))
	argnames = (:zdim, :hdim, :lr, :batchsize, :activation, :nlayers, :init_seed)
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

    train_new = (train[1].data.data, train[2])
    val_inst = GroupAD.reindex(val[1],val[2] .== 0).data.data
    nv = size(val_inst,2)
    nv_all = size(val[1].data.data,2)
    val_new = (val[1].data.data, vcat(zeros(Float32,nv),ones(Float32,nv_all-nv)))
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
	model = GroupAD.Models.vae_constructor(;idim=size(data[1][1],1), parameters...)
	# get only the data needed and unpack them from bags to intances
    instance_data = train_val_data(data)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, instance_data, loss; max_train_time=82800/max_seed, 
			patience=100, check_interval=10, parameters...)
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
		(x -> GroupAD.Models.reconstruction_score_bag(info.model,x,mean), 
			merge(parameters, (score = "reconstruction_mean",))),
        (x -> GroupAD.Models.reconstruction_score_bag(info.model,x,sum), 
			merge(parameters, (score = "reconstruction_sum",))),
        (x -> GroupAD.Models.reconstruction_score_bag_mean(info.model,x,mean), 
			merge(parameters, (score = "reconstruction-mean_mean",))),
        (x -> GroupAD.Models.reconstruction_score_bag_mean(info.model,x,sum), 
			merge(parameters, (score = "reconstruction-mean_sum",))),
	]
end

"""
	edit_params(data, parameters)

This function edits the sampled parameters based on nature of data - e.g. dimensions etc. Default
behaviour is doing nothing - then used `GroupAD.edit_params`.
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
		datadir("experiments/contamination-$(contamination)"),
		)
end
