using DrWatson
@quickactivate
using ArgParse
using GroupAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Random
using ValueHistories
#generative MIL
using GenerativeMIL
using Flux, Flux3D
using Zygote
using CUDA
using GenerativeMIL: transform_batch
using GenerativeMIL.Models: check, loss, unpack_mill
using MLDataPattern

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
        arg_type = Int
        help = "seed, if seed =< 0 it is considered concrete single seed to train with"
        default = 1
    "dataset"
        default = "MNIST"
        arg_type = String
        help = "dataset"
	"anomaly_classes"
		arg_type = Int
		default = 10
		help = "number of anomaly classes, if =< 0 then considered as single anomaly class to train with"
	"method"
		default = "leave-one-in"
		arg_type = String
		help = "method for data creation -> \"leave-one-out\" or \"leave-one-in\" "
    "contamination"
        default = 0.0
        arg_type = Float64
        help = "training data contamination rate"
	"random_seed"
		default = 0
		arg_type = Int
		help = "random seed for sample_params function (to be able to train multile seeds in parallel)"
end
parsed_args = parse_args(ARGS, s)
@unpack max_seed, dataset, anomaly_classes, method, contamination, random_seed = parsed_args


####################################################
# simple preparation "hacking" before sampling etc #
####################################################
if random_seed != 0
	Random.seed!(random_seed)
end
# option to train specific seed and specific anomaly class | useful for possible parallelization
max_seed = (max_seed <= 0) ? [abs(max_seed)] : [1:max_seed...]
anomaly_classes = (anomaly_classes <= 0) ? [abs(anomaly_classes)] : [1:anomaly_classes...]

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "foldingnet_vae"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params(seed=nothing)
	(seed!==nothing) ? Random.seed!(seed) : nothing

	model_par_vec = (
		[3, 16],				# :n_neighbors -> number of neighbors for knn local seach
		2 .^(4:6),				# :edim -> number of neurons in encoder 
		[2 .^(3:7)..., 512],	# :zdim -> latent dimension (512 is added because of orignal paper)
		["relu", "swish"],		# :activation -> activation function for whole network
		2 .^(5:7),				# :ddim -> number of neurons in decoder (foldings)
		[3],				    # :pdim -> dimension of "prior" shpere which is later folded	
	)

	training_par_vec = (
		2 .^ (6:8), 		# :batchsize -> size of one training batch
		10f0 .^(-4:-3),		# :lr -> learning rate
		[1f0],				# :beta -> final β scaling factor for KL divergence
		[300], 			# :epochs -> n of iid iterations (depends on bs and datasize) proportional to n of :epochs 
		1:Int(1e8), 		# :init_seed -> init seed for random samling for experiment instace 
	)
	model_argnames = (:n_neighbors, :edim, :zdim, :activation, :ddim, :pdim)
	training_argnames = (:batchsize, :lr, :beta, :epochs, :init_seed )

	model_params = (;zip(model_argnames, map(x->sample(x, 1)[1], model_par_vec))...)
	training_params = (;zip(training_argnames, map(x->sample(x, 1)[1], training_par_vec))...)

	#reset seed
	(seed!==nothing) ? Random.seed!() : nothing

	return merge(model_params, training_params)
end


sample_params_() = (random_seed != 0) ? sample_params(random_seed) : sample_params()

"""
	fit(data, parameters)
This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	n_dim = size(data[1][1], 1)
	n_samples = maximum([maximum([size(data[j][1][i].data)[2] for i = 1:length(data[j][1])]) for j=1:3])
	min_n_samples = minimum([minimum([size(data[j][1][i].data)[2] for i = 1:length(data[j][1])]) for j=1:3])
	# n_samples for model is set to max number of instances in bag in dataset
	if parameters[:n_neighbors] > min_n_samples
		update_params = (n_neighbors=min_n_samples,)
		parameters = merge(parameters, update_params)
	end
	# construct model - constructor should only accept kwargs
	model = GenerativeMIL.Models.foldingnet_constructor_from_named_tuple( 
        ;idim=n_dim, local_cov=true, skip=true, parameters...
    ) 
	# fit train data
	# max. train time: 24 hours, over 10 CPU cores -> 2.4 hours of training for each model
	# the full traning time should be 48 hours to ensure all scores are calculated
	# training time is decreased automatically for less cores!
	try 
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss; max_train_time=(24-0.5)*3600/length(max_seed)/length(anomaly_classes), 
			patience=50, check_interval=20, parameters...)
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
	return training_info, [
		(x -> Flux3D.chamfer_distance(model(x), x), 
		merge(parameters, (score = "input",)))
	]
	#((x, x_mask) -> GenerativeMIL.Models.reconstruct(info.model, x, x_mask), merge(parameters, (score = "reconstructed_input",)))
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
	GroupAD.point_cloud_experimental_loop_gpu(
		sample_params_, 
		fit, 
		edit_params, 
		max_seed, 
		modelname, 
		dataset, 
		contamination, 
		datadir("experiments/c-$(contamination)"),
		anomaly_classes,
        method
		)
end