using Mill
using Mill: nobs
using StatsBase
using DrWatson
using Flux
using Random
using GroupAD: check_params, experiment

"""
    hmil_constructor(Xtrain, mdim, activation, aggregation, nlayers; seed = nothing)

Constructs a classifier as a model composed of Mill model and simple Chain of Dense layers.
The output dimension is fixed to be 2, `mdim` is the hidden dimension in both Mill model
the Chain model.
"""
function hmil_constructor(Xtrain; mdim::Int=16, activation::String="relu", aggregation::String="SegmentedMeanMax", nlayers::Int=2, odim::Int=2, seed = nothing, kwargs...)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    activation = eval(Symbol(activation))
    aggregation = BagCount ∘ eval(Symbol(aggregation))

    # mill model
    m = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, mdim, activation),
        d -> aggregation(d)
    )

    # create the net after Mill model
    if nlayers == 1
        net = Dense(mdim, odim)
    elseif nlayers == 2
        net = Chain(Dense(mdim, mdim, activation), Dense(mdim, odim))
    elseif nlayers == 3
        net = Chain(Dense(mdim, mdim, activation), Dense(mdim, mdim, activation), Dense(mdim, odim))
    end

    # connect the full model
    full_model = Chain(m, net, softmax)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    # try that the model works
    try
        full_model(Xtrain[1])
    catch e
        error("Model wrong, error: $e")
    end

    return full_model
end

function minibatch(data, labels; batchsize)
    n = nobs(data)
    ix = sample(1:n, batchsize)
    return data[ix], labels[:, ix]
end

function classification_data(data, na, seed)
	tr_x, tr_l = data[1]
    val_x, val_l = data[2]

    @info "There are $(sum(val_l)) anomalies in validation labels."

	# add specified number of anomalous samples to train data
	Random.seed!(seed)

	if na == 0
		# both train and validation data only contain normal samples
		train_data = tr_x
		train_labels = tr_l # Flux.onehotbatch(tr_l, [0,1])
		val_data = GroupAD.reindex(val_x, val_l .== 0)
		# val_labels = Flux.onehotbatch(val_l[val_l .== 0], [0,1])
		val_labels = val_l[val_l .== 0]
		@info "There are $(sum(val_labels)) anomalies in validation data."
    elseif na == 100
		# case for all anomalies in validation

		# get all validation
        an_data = GroupAD.reindex(val_x, val_l .== 1)
        an_labels = val_l[val_l .== 1]
		l_an = length(an_labels)

		# sample indexes with fixed seed without replacement
		ix = sample(1:l_an, l_an, replace=false)

		# get new train and validation datasets
		# train is the first half
        train_data = cat(tr_x, GroupAD.reindex(an_data, ix[1:l_an ÷ 2]))
        # train_labels = Flux.onehotbatch(vcat(tr_l, an_labels[1:l_an ÷ 2]), [0,1])
		train_labels = vcat(tr_l, an_labels[1:l_an ÷ 2])
		@info "Adding $(sum(train_labels)) (half) to train data."

		# validation is the second half
		val_data = cat(
			GroupAD.reindex(val_x, val_l .== 0), 			# normal validation data
			GroupAD.reindex(an_data, ix[l_an ÷ 2 + 1:end])	# second half of validation anomalies
		)
		val_labels = vcat(
			val_l[val_l .== 0],					# normal validation labels
			an_labels[ix[l_an ÷ 2 + 1:end]]		# anomalous validation labels
		)
        @info "$(sum(val_labels)) left in validation data."
	else
		if na > sum(val_l .== 1)
			# if there are not enough anomalies, error and not calculate the model
			error("Not enough anomalies to add.")
		else
			# get only data and labels with anomalies
            an_data = GroupAD.reindex(val_x, val_l .== 1)
            an_labels = val_l[val_l .== 1]
			l_an = length(an_labels)

			# sample new indexes based on a given seed
			ix = sample(1:l_an, na, replace=false)
			ix_train = ix[1:na÷2]		# first half to train
			ix_val = ix[na÷2+1:end]		# second half to validation
			
			train_data = cat(tr_x, GroupAD.reindex(an_data, ix_train))
			# train_labels = Flux.onehotbatch(vcat(tr_l, an_labels[ix_train]), [0,1])
			train_labels = vcat(tr_l, an_labels[ix_train])
            @info "Adding $(sum(an_labels[ix_train])) to train data."

			val_data = cat(
                GroupAD.reindex(val_x, val_l .== 0),
                GroupAD.reindex(an_data, ix_val)
            )
			val_labels = vcat(an_labels[ix_val], val_l[val_l .== 0])

            @info "Leaving $(sum(val_labels)) in validation data."
		end
	end

	# reset the seed
	Random.seed!()

	@show nobs(train_data), length(train_labels), sum(train_labels)
	@show nobs(val_data), length(val_labels), sum(val_labels)
	return (train_data, train_labels), (val_data, val_labels)
end

"""
	fit_hmil!(model::Chain, data::Tuple, loss::Function; max_train_time=82800, lr=0.001, 
		batchsize=64, patience=30, check_interval::Int=10, kwargs...)

Function to fit HMIL model.
"""
function fit_hmil!(model::Chain, data::Tuple, loss::Function;
	max_iters=10000, max_train_time=82800, lr=0.001, batchsize=64, patience=30, na=10,
	check_interval::Int=10, seed, kwargs...)

	history = MVHistory()
	opt = ADAM(lr)

	tr_model = deepcopy(model)
	ps = Flux.params(tr_model)
	_patience = patience

	best_val_loss = Inf
	i = 1
	start_time = time()

	# prepare data
	tr, vl = classification_data(data, na, seed)
	train_data, train_labels = tr[1], Flux.onehotbatch(tr[2], [0,1])
	val_data, val_labels = vl[1], Flux.onehotbatch(vl[2], [0,1])
    
	lossf(x, y) = loss(tr_model, x, y)

	# infinite for loop via RandomBatches
	for batch in RandomBatches(train_data, 1)
		# classic training
        batches = map(_ -> minibatch(train_data, train_labels, batchsize=batchsize), 1:10);
        Flux.train!(lossf, ps, batches, opt)
		
        # only batch training loss
        batch_loss = mean(x -> lossf(x...), batches)

    	push!(history, :training_loss, i, batch_loss)
		if mod(i, check_interval) == 0
			
			# validation/early stopping
			# val_loss = lossf(bag_batch) # mean(lossf.(val_x))
			val_loss = lossf(val_data, val_labels)
			
			@info "$i - loss: $(batch_loss) (batch) | $(val_loss) (validation)"

			if isnan(val_loss) || isnan(batch_loss)
				error("Encountered invalid values in loss function.")
			end

			push!(history, :validation_likelihood, i, val_loss)
			
			if val_loss < best_val_loss
				best_val_loss = val_loss
				_patience = patience

				# this should save the model at least once
				# when the validation loss is decreasing 
				model = deepcopy(tr_model)
			else # else stop if the model has not improved for `patience` iterations
				_patience -= 1
				# @info "Patience is: $_patience."
				if _patience == 0
					@info "Stopped training after $(i) iterations. Patience exceeded."
					break
				end
			end
		end
		if (time() - start_time > max_train_time) | (i > max_iters) # stop early if time is running out
			model = deepcopy(tr_model)
			@info "Stopped training after $(i) iterations, $((time() - start_time) / 3600) hours."
			break
		end
		i += 1
	end
	# again, this is not optimal, the model should be passed by reference and only the reference should be edited
	return [(history = history, iterations = i, model = model, npars = sum(map(p -> length(p), Flux.params(model)))), tr, vl]
end

function hmil_basic_loop(sample_params_f, fit_f, edit_params_f, 
	max_seed, modelname, dataset, contamination, savepath)
	
	# sample the random hyperparameters
	parameters = sample_params_f()

	for na in [100,10,20]
	# for na in 0
		# with these hyperparameters, train and evaluate the model on different train/val/tst splits
		for seed in 1:max_seed
			# define where data is going to be saved
			_savepath = joinpath(savepath, "$(modelname)/$(dataset)/na=$na/seed=$(seed)")
			mkpath(_savepath)

			# get data
			@time data = GroupAD.load_data(dataset, seed=seed, contamination=contamination)
			@info "Data loaded..."

			# edit parameters
			edited_parameters = edit_params_f(data, merge(parameters, (na=na, )))

			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			@info "Train/validation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[3][1], 2))"
			@info "Number of features: $(size(data[1][1], 1))"

			# fit
			training_info, results, data = fit_f(data, edited_parameters, seed)

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
				experiment(result..., data, _savepath; save_entries...)
			end
		end
	end
end

function hmil_pc_loop(sample_params_f, fit_f, edit_params_f, 
	max_seed, modelname, dataset, contamination, savepath, anomaly_classes, method)

	# sample the random hyperparameters
	parameters = sample_params_f()

	for na in [100,10,20]
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
				else
					error("This model can only run on point cloud datasets!")
				end
				
				# define where data is going to be saved
				# _savepath = joinpath(savepath, "$(modelname)/$(dataset)/$(method)/class_index=$(class)/seed=$(seed)")
				_savepath = joinpath(savepath, "$(modelname)/$(method)/class_index=$(class)/seed=$(seed)")
				mkpath(_savepath)
				
				# edit parameters
				edited_parameters = edit_params_f(data, merge(parameters, (na=na, )), class, method)
				
				@info "Trying to fit $modelname on $(dataset) in $method setting.\nModel parameters: $(edited_parameters)..."
				@info "Train/validation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[3][1], 2))"
				@info "Number of features: $(size(data[1][1], 1))"

				# fit
				training_info, results, data = fit_f(data, edited_parameters, seed)

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
					experiment(result..., data, _savepath; save_entries...)
				end
			end
		end
	end
end


function score_hmil(model, x)
    model(x)[2, :]
end

function get_label_hmil(model, x)
    Flux.onecold(model(x), [0,1])
end