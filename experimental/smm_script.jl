function sample_params()
    distance = sample(["MMD", "Chamfer"])
    kernel = sample(["Gaussian", "IMQ"])
    rand() > 0.5 ? γ = rand(Uniform(0,3)) : γ = rand(Uniform(3,100))
    rand() > 0.5 ? h = rand(Uniform(0,3)) : h = rand(Uniform(3,100))
    nu = sample(0.1:0.1:0.9)

    if distance == "Chamfer"
        kernel = "none"
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

function StatsBase.score(m::SMM, Xtrain, Xtest)
    DM = pairwise(m.distance, Xtrain, Xtest)
    kernel = exp.(.- m.h .* DM)
    _, dec = svmpredict(m.model, kernel)
    return pred, dec[1,:]
end
function predictions(m::SMM, Xtrain, Xtest)
    DM = pairwise(m.distance, Xtrain, Xtest)
    kernel = exp.(.- m.h .* DM)
    _, dec = svmpredict(m.model, kernel)
    return pred
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
	model = SMMModel(parameters...)

	# fit train data
	model, fit_t, _, _, _ = @timed StatsBase.fit!(model, data)

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		model = model
		)

	# now return the info to be saved and an array of tuples (anomaly score function, hyperparatemers)
	return training_info, [
		(x -> score(model,x), merge(parameters, (score = "score", ))),
        (x -> predictions(model,x), merge(parameters, (score = "predict", )))
	]
end


"""
    StatsBase.fit!(m::SMMModel, data::Tuple)

Function to fit SMM model.
"""
function StatsBase.fit!(m::SMMModel, data::Tuple)
	
	train, _, _ = data
    Xtrain, ytrain = unpack_mill(train)

    if m.kernel == "none"
        distance = Chamfer()
    elseif m.kernel == "Gaussian"
        distance = MMD(GaussianKernel(m.γ))
    elseif m.kernel == "IMQ"
        distance = MMD(IMQKernel(m.γ))
    end

    model = SMM(distance, Xtrain, m.h; nu = m.nu)
end