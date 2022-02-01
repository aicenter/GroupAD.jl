using DrWatson
@quickactivate
using ArgParse
using GroupAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Statistics
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
modelname = "knn_basic"

"""
	sample_params()
Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	par_vec = (1:50,["mean", "maximum", "median"])
	argnames = (:k,:aggregation)
	return (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
end

function aggregate_toy(data, agf::Function)
    m = mean.(data, dims=2)
    hcat(m...)
end

aggregate_toy(data::Tuple, agf::Function) = Tuple(map(d->(aggregate_toy(d[1], agf), d[2]), data))


"""
	fit(data, parameters)
This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GroupAD.Models.knn_constructor(;v=:kappa, parameters...)

	# aggregate bags into vectors
	# first convert the aggregation string to a function
	agf = getfield(StatsBase, Symbol(parameters.aggregation))
	data = aggregate_toy(data, agf)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		# return an empty array if fit fails so nothing is computed
		return (fit_t = NaN,), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		model = nothing
		)

	# now return the different scoring functions
	function knn_predict(model, x, v::Symbol)
		_x = aggregate_toy(x, agf)
		try 
			return predict(model, _x, v)
		catch e
			if isa(e, ArgumentError) # this happens in the case when k > number of points
				return NaN # or nothing?
			else
				rethrow(e)
			end
		end
	end
	training_info, [(x -> knn_predict(model, x, v), merge(parameters, (distance = v,))) for v in [:gamma, :kappa, :delta]]
end
"""
	edit_params(data, parameters)

This function edits the sampled parameters based on nature of data - e.g. dimensions etc. Default
behaviour is doing nothing.
""" 
function edit_params(data, parameters, scenario)
	merge(parameters, (scenario = scenario, ))
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
		scenario, 
		modelname, 
		dataset,
		datadir("experiments/contamination-$(contamination)")
		)
end
