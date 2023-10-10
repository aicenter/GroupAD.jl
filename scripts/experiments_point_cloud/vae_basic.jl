using DrWatson
@quickactivate
using ArgParse
using GroupAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using GroupAD.GenerativeModels

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
        arg_type = Int
        help = "seed"
        default = 1
    "dataset"
        default = "MNIST"
        arg_type = String
        help = "dataset"
	"anomaly_classes"
		arg_type = Int
		default = 1
		help = "number of anomaly classes"
	"method"
		default = "leave-one-out"
		arg_type = String
		help = "method for data creation -> \"leave-one-out\" or \"leave-one-in\" "
   "contamination"
        default = 0.0
        arg_type = Float64
        help = "training data contamination rate"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes, method, contamination = parsed_args


#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "vae_basic"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	# MNIST has idim = 2 -> fewer possibilities for sampling
	# zdim: 1, 2, 4, 8, 12, 16
	# hdim: 4, 8, 16, 32, 64
	par_vec = ([1, 2, 4, 8, 12, 16], 2 .^(2:6),  ["scalar", "diagonal"], 10f0 .^(-4:-3), 2 .^ (5:7), ["relu", "swish", "tanh"], 3:4, 1:Int(1e8),
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

	# aggregate bags into vectors
	# first convert the aggregation string to a function
	agf = getfield(StatsBase, Symbol(parameters.aggregation))
	data = GroupAD.Models.aggregate(data, agf)

	# fit train data
	# max. train time: 24 hours, over 10 CPU cores -> 2.4 hours of training for each model
	# the full traning time should be 48 hours to ensure all scores are calculated
	# training time is decreased automatically for less cores!
	try
		cores = Threads.nthreads()
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss; max_train_time=23*3600*cores/max_seed/anomaly_classes, 
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
	batchsize=512
	training_info, [
		(x -> GroupAD.Models.reconstruction_score(info.model,x,agf), 
			merge(parameters, (score = "reconstruction",))),
		(x -> GroupAD.Models.reconstruction_score_mean(info.model,x,agf), 
			merge(parameters, (score = "reconstruction-mean",))),
		(x -> GroupAD.Models.reconstruction_score(info.model,x,agf,L), 
			merge(parameters, (score = "reconstruction-sampled", L=L)))		
	]
end

"""
	edit_params(data, parameters)

This function edits the sampled parameters based on nature of data - e.g. dimensions etc. Default
behaviour is doing nothing - then used `GroupAD.edit_params`.

Note: Since MNIST has idim = 3, the only possible zdim smaller than idim is 2.
Therefore this function is suspended.

function edit_params(data, parameters)
	idim = size(data[1][1].data.data,1)
	# put the largest possible zdim where zdim < idim, the model tends to converge poorly if the latent dim is larger than idim
	if parameters.zdim >= idim
		zdims = 2 .^(1:8)
		zdim_new = zdims[zdims .< idim][end]
		parameters = merge(parameters, (zdim=zdim_new,))
	end
	parameters
end
"""
function edit_params(data, parameters, class, method)
	merge(parameters, (method = method, class = class, ))
end 

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	if dataset == "MNIST"
		GroupAD.point_cloud_experimental_loop(
			sample_params, 
			fit, 
			edit_params, 
			max_seed, 
			modelname, 
			dataset, 
			contamination, 
			datadir("experiments/contamination-$(contamination)/MNIST"),
			anomaly_classes,
			method
		)
	elseif dataset == "modelnet"
		GroupAD.point_cloud_experimental_loop(
			sample_params, 
			fit, 
			edit_params, 
			max_seed, 
			modelname, 
			dataset, 
			contamination, 
			datadir("experiments/contamination-$(contamination)/modelnet"),
			anomaly_classes,
			method
		)
	end
end
