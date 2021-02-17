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
using CSV, DataFrames
include(srcdir("mnist_data.jl"))

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
modelname = "nex_statistician"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
For NeuralStatistician model there are two conditions:
- `cdim` < `hdim`
- `zdim` < `cdim`
"""
function sample_params()
	df = CSV.read(datadir("table.csv"),DataFrame)
    argnames = (:hdim, :vdim, :cdim, :zdim, :lr, :nlayers, :activation, :init_seed)
    idx = sample(1:32)
    par = df[idx,:] |> Array
    parameters = (;zip(argnames, vcat(par,sample(1:Int(1e8))))...)
	return parameters
end


"""
	loss(model::GenerativeModels.NeuralStatistician, x)

Negative ELBO for training of a Neural Statistician model.
"""
loss(model::GenerativeModels.NeuralStatistician,x) = -elbo(model, x)

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

	# aggregate bags into vectors
	# first convert the aggregation string to a function
	# agf = getfield(StatsBase, Symbol(parameters.aggregation))
	# data = GroupAD.Models.aggregate(data, agf)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss; max_train_time=82800/max_seed, 
			patience=10, check_interval=1, parameters...)
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
        (x -> GroupAD.Models.reconstruction_score_mean(info.model,x), 
			merge(parameters, (score = "reconstruction-mean",))),
		]
end

"""
	edit_params(data, parameters)
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters)
	inter = [p for p in pairs(parameters) if p[1] != :init_seed]
	(; inter...)
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
