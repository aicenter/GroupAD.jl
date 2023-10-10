using DrWatson
@quickactivate
using ArgParse
using GroupAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using Distributions
using ValueHistories
using MLDataPattern: RandomBatches
using Random

# dataset = "modelnet"
# method = "chair"
# data = GroupAD.load_data(dataset, method=method)

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
		default = "leave-one-in"
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
modelname = "hmil_classifier"
# sample parameters, should return a Dict of model kwargs 

# fix seed to always choose the same hyperparameters
function sample_params()
    mdim = sample([8,16,32,64,128,256])
    activation = sample(["sigmoid", "tanh", "relu", "swish"])
    aggregation = sample(["SegmentedMeanMax", "SegmentedMax", "SegmentedMean"])
    nlayers = sample(1:3)
    return (mdim=mdim, activation=activation, aggregation=aggregation, nlayers=nlayers)
end

loss(model, x, y) = Flux.crossentropy(model(x), y)
# loss(model, x, y) = Flux.logitcrossentropy(model(x), y)

"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters, seed)
	# construct model - constructor should only accept kwargs
	# model = GroupAD.Models.hmil_constructor(;idim=size(data[1][1],1), parameters...)
	model = GroupAD.Models.hmil_constructor(data[1][1]; parameters...)

	# fit train data
	# max. train time: 24 hours
	try
		global _info, fit_t, _, _, _ = @timed GroupAD.Models.fit_hmil!(model, data, loss; max_train_time=22*3600/max_seed/3, 
			patience=200, check_interval=5, seed=seed, parameters...)
		global info = _info[1]
		global new_data = (_info[2], _info[3], data[3])
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
	# the score functions themselves are inside experimental loop
	return training_info, [
		(x -> GroupAD.Models.score_hmil(info.model, x),
			merge(parameters, (score = "normal_prob",))),
		(x -> GroupAD.Models.get_label_hmil(info.model, x),
			merge(parameters, (score = "get_label",)))
	], new_data
end

"""
	edit_params(data, parameters)
	
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters, class, method)
	merge(parameters, (method = method, class = class, ))
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	if dataset == "MNIST"
		GroupAD.Models.hmil_pc_loop(
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
		GroupAD.Models.hmil_pc_loop(
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