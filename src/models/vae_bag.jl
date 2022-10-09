"""
	StatsBase.fit_bag!(model::VAE, data::Tuple, loss::Function; max_train_time=82800, lr=0.001, 
		batchsize=64, patience=30, check_interval::Int=10, kwargs...)

Modification of the fit function to incorporate training of instance VAE on bags,
with the usage of Chamfer distance etc.
"""
function StatsBase.fit_bag!(model::VAE, data::Tuple, loss::Function; max_iters=100000, max_train_time=82800, 
	lr=0.001, batchsize=64, patience=30, check_interval::Int=10, kwargs...)
	history = MVHistory()
	opt = ADAM(lr)

	tr_model = deepcopy(model)
	ps = Flux.params(tr_model)
	_patience = patience

	# prepare data for Neural Statistician
	tr_x, tr_l = unpack_mill(data[1])
	vx, vl = unpack_mill(data[2])
	val_x = vx[vl .== 0]

	# on large datasets, batching loss is faster
	best_val_loss = Inf
	i = 1
	start_time = time() # end the training loop after 23hrs

	lossf(x) = loss(tr_model, x)

	for batch in RandomBatches(tr_x, 1)
		# train with standard Flux.train! function
		bag_batch = RandomBagBatches(tr_x,batchsize=batchsize,randomize=true)
		Flux.train!(lossf, ps, [bag_batch], opt)
		batch_loss = lossf(bag_batch)

		push!(history, :training_loss, i, batch_loss)
		if mod(i, check_interval) == 0
			
			# validation/early stopping
			val_loss = lossf(bag_batch) # mean(lossf.(val_x))
			
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
				if _patience == 0
					@info "Stopped training after $(i) iterations."
					break
				end
			end
		end
		if (time() - start_time > max_train_time) | (i > max_iters) # stop early if time is running out
			model = deepcopy(tr_model)
			@info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours."
			break
		end
		i += 1
	end
	# again, this is not optimal, the model should be passed by reference and only the reference should be edited
	(history=history, iterations=i, model=model, npars=sum(map(p->length(p), Flux.params(model))))
end