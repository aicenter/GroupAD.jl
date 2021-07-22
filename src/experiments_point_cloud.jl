"""
	basic_experimental_loop(sample_params_f, fit_f, edit_params_f, 
		max_seed, modelname, dataset, contamination, savepath, classes)

This function takes a function that samples parameters, a fit function and a function that edits the sampled
parameters and other parameters. Then it loads data, samples hyperparameters, calls the fit function
that is supposed to construct and fit a model and finally evaluates the returned score functions on 
the loaded data.

This function works for point cloud datasets. Differentiation between leave-one-in and leave-one-out
setting is done parameter `method`.
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
		for seed in 1:max_seed
	    # with these hyperparameters, train and evaluate the model on different train/val/tst splits
			for class in 1:anomaly_classes
				# load data for either "MNIST_in" or "MNIST_out" and set the setting
				# prepared for other point cloud datasets such as ModelNet10
				data = load_data(dataset, anomaly_class_ind=class, seed=seed, method=method, contamination=contamination)
				if method == "leave-one-in"
					data = GroupAD.leave_one_in(data; seed=seed)
				elseif method == "leave-one-out"
					data = GroupAD.leave_one_out(data; seed=seed)
				else
					error("This model can only run on point cloud datasets!")
				end
				
				# define where data is going to be saved
				_savepath = joinpath(savepath, "$(modelname)/$(dataset)/$(method)/class_index=$(class)/seed=$(seed)")
				mkpath(_savepath)
				
				# edit parameters
				edited_parameters = edit_params_f(data, parameters)
				
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
	end
	(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
end