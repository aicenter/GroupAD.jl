using DrWatson
@quickactivate
using ArgParse
using GroupAD
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
modelname = "SMM"
# sample parameters, should return a Dict of model kwargs
function sample_params()
    distance = sample(["MMD", "Chamfer"])
    kernel = sample(["Gaussian", "IMQ"])
    γ = rand()
    rand() > 0.5 ? h = rand(Uniform(0,3)) : h = rand(Uniform(3,100))
    nu = sample(0.2:0.1:0.9)

    if distance == "Chamfer"
        kernel = "none"
        γ = 0f0
    end
    parameters = (
        distance = distance,
        kernel = kernel,
        γ = Float32(γ),
        h = Float32(h),
        nu = nu
    )
    return parameters
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
	model = GroupAD.Models.SMMModel(parameters...)
    @info "Models created."

	# fit train data
	model, fit_t, _, _, _ = @timed StatsBase.fit!(model, data)
    @info "Model fitted."

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		model = nothing,
        history=  nothing
		)

    train_data, _ = GroupAD.Models.unpack_mill(data[1])
	# now return the info to be saved and an array of tuples (anomaly score function, hyperparatemers)
	return training_info, [
		(x -> GroupAD.Models.score(model, train_data, x), merge(parameters, (score = "score", ))),
	]
end

"""
	edit_params(data, parameters)
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters)
	Xtrain, _ = GroupAD.Models.unpack_mill(data[1])

    # calculate ideal bandwidth
    if parameters.distance == "MMD"
        M = pairwise(GroupAD.Models.PEuclidean(), Xtrain)
        m = 1/median(M)
        γnew = sample(0.6m:0.1m:1.4m)
        parameters = merge(parameters, (γ = γnew,))
    end

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