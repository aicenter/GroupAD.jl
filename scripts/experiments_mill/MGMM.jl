using DrWatson
@quickactivate
using ArgParse
using GroupAD
using GroupAD.Models: MGMM, unpack_mill, MGMM_constructor
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
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
modelname = "MGMM"

"""
	sample_params()

Returns sampled parameters. MGMM has only 2 parameters:
- T: number of topics
- K: number of Gaussian clusters.
"""
function sample_params()
    par_vec = (2:10, 2:10, 1:Int(1e8))
    argnames = (:K, :T, :init_seed)
    parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
    return parameters
end

"""
	loss(m::MGMM, x)

Returns the negative log-likelihood of a bag as a sum of instance log-likelihoods.
"""
function loss(m::MGMM, x)
    MM = GroupAD.Models.toMixtureModel(m)
    -sum(logpdf(MM, x))
end

"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	tr_x, _ = unpack_mill(data[1])
    model = MGMM_constructor(tr_x; parameters...)

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
	return training_info, [
		(x -> GroupAD.Models.topic_score(info.model,x), 
			merge(parameters, (score = "topic",))),
        (x -> GroupAD.Models.point_score(info.model,x), 
			merge(parameters, (score = "point",))),
        (x -> GroupAD.Models.MGMM_score(info.model,x), 
			merge(parameters, (score = "topic+point",)))
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
		datadir("experiments/contamination-$(contamination)/MIL")
		)
end