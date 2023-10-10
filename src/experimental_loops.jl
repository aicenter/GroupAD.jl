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

###################################################
### MIL, point cloud and toy experimental loops ###
###################################################

"""
	basic_experimental_loop(sample_params_f, fit_f, edit_params_f, 
		max_seed, modelname, dataset, contamination, savepath)

This function takes a function that samples parameters, a fit function and a function that edits the sampled
parameters and other parameters. Then it loads data, samples hyperparameters, calls the fit function
that is supposed to construct and fit a model and finally evaluates the returned score functions on 
the loaded data.

Runs on MIL data.
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
			@time data = GroupAD.load_data(dataset, seed=seed, contamination=contamination)
			@info "Data loaded..."

			# edit parameters
			# bagkNN and SMM need to calculate parameter from the data
			# the parameter needs to stay the same for all seeds
			if modelname in ["bag_knn","SMM", "SMMC"]
				if seed == 1
					parameters = edit_params_f(data, parameters)
					edited_parameters = parameters
				else
					edited_parameters = parameters
				end
			else
				# other models can edit parameters for each seed, because the change is deterministic
				edited_parameters = edit_params_f(data, parameters)
			end


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
				@time for result in results
					if modelname in ["vae_instance", "vae_instance_chamfer", "statistician", "statistician_chamfer", "PoolModel"]
						experiment_bag(result..., data, _savepath; save_entries...)
					elseif modelname in ["SMM", "SMMC"]
						experiment(result..., GroupAD.Models.unpack_mill.(data), _savepath; save_entries...)
					elseif modelname == "bag_knn"
						experiment_bagknn(result..., GroupAD.Models.unpack_mill.(data), _savepath; save_entries...)
					else # vae_basic, knn_basic
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

"""
	point_cloud_experimental_loop(sample_params_f, fit_f, edit_params_f, 
		max_seed, modelname, dataset, contamination, savepath, anomaly_classes, method)

This function takes a function that samples parameters, a fit function and a function that edits the sampled
parameters and other parameters. Then it loads data, samples hyperparameters, calls the fit function
that is supposed to construct and fit a model and finally evaluates the returned score functions on 
the loaded data.

This function works for point cloud datasets. Differentiation between leave-one-in and leave-one-out
setting is done via parameter `method`.

The calculation is run in parallel over seeds. It is to be determined how many cores will
be used and what the max training time will be. Have in mind that statistician is slower to learn
and might need smaller check time (prob. 1 or 2 might be best) and it will take time to calculate
all the scores for larger model (with more neurons).
"""
function point_cloud_experimental_loop(sample_params_f, fit_f, edit_params_f, 
	max_seed, modelname, dataset, contamination, savepath, anomaly_classes, method)
	# set a maximum for parameter sampling retries
	# this is here because you might sample the same parameters of an already trained model
	# in that case this loop runs again, for a total of 10 tries
	try_counter = 0
	max_tries = 10*max_seed
	while try_counter < max_tries
		# sample the random hyperparameters
		parameters = sample_params_f()

		# run over all classes with the same hyperparameters
		# use more CPU cores for calculation
		@info "Starting parallel process on $(Threads.nthreads()) cores (over $max_seed seeds)."
		Threads.@threads for seed in 1:max_seed
			# with these hyperparameters, train and evaluate the model on different train/val/tst splits
			# load data for either "MNIST_in" or "MNIST_out" and set the setting
			# prepared for other point cloud datasets such as ModelNet10
			for class in 1:anomaly_classes
				data = load_data(dataset, anomaly_class_ind=class, seed=seed, method=method, contamination=contamination)
				if method == "leave-one-in"
					data = GroupAD.leave_one_in(data; seed=seed)
				elseif method == "leave-one-out"
					data = GroupAD.leave_one_out(data; seed=seed)
				elseif dataset == "modelnet"
					nothing
				else
					error("This model can only run on point cloud datasets!")
				end
				
				# define where data is going to be saved
				# _savepath = joinpath(savepath, "$(modelname)/$(dataset)/$(method)/class_index=$(class)/seed=$(seed)")
				if dataset == "modelnet"
					_savepath = joinpath(savepath, "$(modelname)/$(method)/seed=$(seed)")
				else
					_savepath = joinpath(savepath, "$(modelname)/$(method)/class_index=$(class)/seed=$(seed)")
				end
				mkpath(_savepath)
				
				# edit parameters
				edited_parameters = edit_params_f(data, parameters, class, method)
				
				@info "Trying to fit $modelname on $(dataset) in $method setting.\nModel parameters: $(edited_parameters)..."
				@info "Train/validation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[3][1], 2))"
				@info "Number of features: $(size(data[1][1], 1))"

				# check if a combination of parameters and seed alread exists
				if check_params(_savepath, edited_parameters)
					@info "Params check done. Trying to fit."
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

						training_info = merge(training_info, (model = nothing, history=nothing))
					end

					# here define what additional info should be saved together with parameters, scores, labels and predict times
					save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset))

					# now loop over all anomaly score funs
					@time for result in results
						if modelname in ["vae_instance", "vae_instance_chamfer", "statistician", "statistician_chamfer", "PoolModel"]
							@info "Calculating results for bag models."
							experiment_bag(result..., data, _savepath; save_entries...)
						else
							@info "Calculating results for instance models."
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
	end
	(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
end

"""
	toy_experimental_loop_toy(sample_params_f, fit_f, edit_params_f, 
		max_seed, modelname, dataset, contamination, savepath)

Experimental loop for toy dataset.
"""
function toy_experimental_loop(sample_params_f, fit_f, edit_params_f, 
		max_seed, scenario, modelname, dataset, savepath)
	# set a maximum for parameter sampling retries
	# this is here because you might sample the same parameters of an already trained model
	# in that case this loop runs again, for a total of 10 tries
	@info "Inside loop."
	try_counter = 0
	max_tries = 10*max_seed
	while try_counter < max_tries
		# sample the random hyperparameters
	    parameters = sample_params_f()
		@info "Parameters sampled."
	    # with these hyperparameters, train and evaluate the model on different train/val/tst splits
	    for seed in 1:max_seed
	    	# define where data is going to be saved
			_savepath = joinpath(savepath, "$(modelname)/$(dataset)/scenario=$(scenario)/seed=$(seed)")
			mkpath(_savepath)

			# get data
			data = load_data(dataset, 120, 120; seed=seed, scenario=scenario)
			@info "Data created..."

			# edit parameters
			edited_parameters = edit_params_f(data, parameters, scenario)

			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			@info "Toy dataset scenario: $scenario."
			@info "Train/validation/test splits: $(length(data[1][1])) | $(length(data[2][1])) | $(length(data[3][1]))"
			@info "Number of features: $(size(data[1][1][1], 1))"

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
				@time for result in results
					if modelname in ["vae_instance", "statistician", "PoolModel"]
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