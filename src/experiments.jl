Base.size(x::Mill.BagNode,args...) = size(x.data.data, args...)

"""
	experiment(score_fun, parameters, data, savepath; save_entries...)

Eval score function on test/val/train data and save.
"""
function experiment(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	tr_data, val_data, tst_data = data

	# extract scores
	tr_scores, tr_eval_t, _, _, _ = @timed score_fun(tr_data[1])
	val_scores, val_eval_t, _, _, _ = @timed score_fun(val_data[1])
	tst_scores, tst_eval_t, _, _, _ = @timed score_fun(tst_data[1])

	# now save the stuff
	savef = joinpath(savepath, savename(parameters, "bson", digits=5))
	result = (
		parameters = parameters,
		tr_scores = tr_scores,
		tr_labels = tr_data[2], 
		tr_eval_t = tr_eval_t,
		val_scores = val_scores,
		val_labels = val_data[2], 
		val_eval_t = val_eval_t,
		tst_scores = tst_scores,
		tst_labels = tst_data[2], 
		tst_eval_t = tst_eval_t
		)
	result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
	if save_result
		tagsave(savef, result, safe = true)
		verb ? (@info "Results saved to $savef") : nothing
	end
	result
end

"""

Evaluate likelihoods and save all the possible corrections.
Saves 6 scores in total:
- sum of likelihoods
- sum + Poisson fit
- sum + LogNormal fit
- sum + logU
- sum + Poisson fit + logU
- sum + LogNormal fit + logU
"""
function experiment_bag(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	tr_data, val_data, tst_data = data

	# unpack data for easier manipulation
	train, _ = unpack_mill(tr_data)
	val, _ = unpack_mill(val_data)
	test, _ = unpack_mill(tst_data)

	# fit MLE cardinality distribution to data
	bag_sizes = size.(train,1)
	poisson, poisson_eval_t = @timed fit_mle(Poisson, bag_sizes)
	lognormal, lognormal_eval_t = @timed fit_mle(LogNormal, bag_sizes)

	# calculate likelihoods
	tr_lh, tr_eval_t, _, _, _ = @timed score_fun(train)
	val_lh, val_eval_t, _, _, _ = @timed score_fun(val)
	tst_lh, tst_eval_t, _, _, _ = @timed score_fun(test)

	logU, logU_eval_t, _, _, _ = @timed calculate_logU(tr_lh)

	# possible scores as anonymous functions
	scores = [
		x -> rec_score_from_likelihood(x),
		x -> rec_score_from_likelihood(x, poisson),
		x -> rec_score_from_likelihood(x, lognormal),
		x -> rec_score_from_likelihood(x, logU),
		x -> rec_score_from_likelihood(x, poisson, logU),
		x -> rec_score_from_likelihood(x, lognormal, logU)
	]

	# helpful vectors
	eval_times = [0, poisson_eval_t, lognormal_eval_t, logU_eval_t, poisson_eval_t + logU_eval_t, lognormal_eval_t + logU_eval_t]
	reconstruction_names = [
		"sum",
		"poisson",
		"lognormal",
		"logU",
		"poisson+logU",
		"lognormal+logU"
	]

	# calculate all the possible scores and save them
	for (i, final_score) in enumerate(scores)
		tr_scores = final_score(tr_lh)
		val_scores = final_score(val_lh)
		tst_scores = final_score(tst_lh)

		# now save the stuff
		savef = joinpath(savepath, savename(merge(parameters, (corr = reconstruction_names[i],)), "bson", digits=5))
		result = (
			parameters = parameters,
			tr_scores = tr_scores,
			tr_labels = tr_data[2], 
			tr_eval_t = tr_eval_t + eval_times[i],
			val_scores = val_scores,
			val_labels = val_data[2], 
			val_eval_t = val_eval_t + eval_times[i],
			tst_scores = tst_scores,
			tst_labels = tst_data[2], 
			tst_eval_t = tst_eval_t + eval_times[i]
			)
		result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
		if save_result
			tagsave(savef, result, safe = true)
			verb ? (@info "Results saved to $savef") : nothing
		end
	end
end

"""
	edit_params(data, parameters)
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters)
	parameters
end

"""
	check_params(savepath, parameters)

Returns `true` if the model with given parameters wasn't already trained and saved. 
"""
function check_params(savepath, parameters)
	if ~isdir(savepath)
		return true
	end
	# filter out duplicates created by tagsave
	fs = filter(x->!(occursin("_#", x)), readdir(savepath))
	# filter out model files
	fs = filter(x->!(startswith(x, "model")), fs)
	# if the first argument name contains a "_", than the savename is parsed wrongly
	saved_params = map(x -> DrWatson.parse_savename("_"*x)[2], fs)
	# now filter out saved models where parameter names are different or missing
	pkeys = collect(keys(parameters))
	filter!(ps->intersect(pkeys, Symbol.(collect(keys(ps))))==pkeys, saved_params)
	for params in saved_params
		all(map(k->params[String(k)] == parameters[k], pkeys)) ? (return false) : nothing
	end
	true
end

"""
	basic_experimental_loop(sample_params_f, fit_f, edit_params_f, 
		max_seed, modelname, dataset, contamination, savepath)

This function takes a function that samples parameters, a fit function and a function that edits the sampled
parameters and other parameters. Then it loads data, samples hyperparameters, calls the fit function
that is supposed to construct and fit a model and finally evaluates the returned score functions on 
the loaded data.
"""
function basic_experimental_loop(sample_params_f, fit_f, edit_params_f, 
		max_seed, modelname, dataset, contamination, savepath)
	# set a maximum for parameter sampling retries
	# this is here because you might sample the same parameters of an already trained model
	# in that case this loop runs again, for a total of 10 tries
	try_counter = 0
	max_tries = 10*max_seed
	while try_counter < max_tries
		# sample the random hyperparameters
	    parameters = sample_params_f()

	    # with these hyperparameters, train and evaluate the model on different train/val/tst splits
	    for seed in 1:max_seed
	    	# define where data is going to be saved
			_savepath = joinpath(savepath, "$(modelname)/$(dataset)/seed=$(seed)")
			mkpath(_savepath)

			# get data
			@time data = load_data(dataset, seed=seed, contamination=contamination)
			@info "Data loaded..."

			# edit parameters
			edited_parameters = edit_params_f(data, parameters)

			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			@info "Train/validation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[3][1], 2))"
			@info "Number of features: $(size(data[1][1], 1))"

			# check if a combination of parameters and seed alread exists
			if check_params(_savepath, edited_parameters)
				# fit
				training_info, results = fit_f(data, edited_parameters)

				# save the model separately			
				if training_info.model != nothing
					modelf = joinpath(_savepath, savename("model", edited_parameters, "bson", digits=5))
					tagsave(
						modelf, 
						Dict("model"=>training_info.model,
							"fit_t"=>training_info.fit_t,
							"history"=>training_info.history,
							"parameters"=>edited_parameters
							), 
						safe = true)
					(@info "Model saved to $modelf")

					training_info = merge(training_info, (model = nothing,history=nothing))
				end

				# here define what additional info should be saved together with parameters, scores, labels and predict times
				save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset))

				# now loop over all anomaly score funs
				for result in results
					if modelname in ["vae_instance", "statistician"]
						experiment_bag(result..., data, _savepath; save_entries...)
					else
						experiment(result..., data, _savepath; save_entries...)
					end
				end
				try_counter = max_tries + 1
			else
				@info "Model already present, trying new hyperparameters..."
				try_counter += 1
			end
		end
	end
	(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
end