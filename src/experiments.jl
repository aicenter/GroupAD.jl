#################################################################################
### Functions to calculate results and scores after models have been trained. ###
#################################################################################

Base.size(x::Mill.BagNode,args...) = size(x.data.data, args...)
using EvalMetrics

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
		# tagsave(savef, result, safe = true)
		safesave(savef, result)
		verb ? (@info "Results saved to $savef") : nothing
	end
	result
end

"""
	experiment_bagknn(score_fun, parameters, data, savepath; save_entries...)

Eval score functions on test/val/train data for BagkNN model and save.
"""
function experiment_bagknn(score_fun, model, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	tr_data, val_data, tst_data = data

	tr_dm, tr_dm_t, _, _, _ = @timed GroupAD.Models.distance_matrix(model, tr_data[1])
	val_dm, val_dm_t, _, _, _ = @timed GroupAD.Models.distance_matrix(model, val_data[1])
	tst_dm, tst_dm_t, _, _, _ = @timed GroupAD.Models.distance_matrix(model, tst_data[1])

	inds = ["kappa", "gamma"]

	# for scf in score_fun
	for i in 1:2
		# extract scores
		tr_scores, tr_eval_t, _, _, _ = @timed score_fun[i](tr_dm)
		val_scores, val_eval_t, _, _, _ = @timed score_fun[i](val_dm)
		tst_scores, tst_eval_t, _, _, _ = @timed score_fun[i](tst_dm)

		# now save the stuff
		savef = joinpath(savepath, savename(merge(parameters, (score = inds[i], )), "bson", digits=5))
		result = (
			parameters = merge(parameters, (score = inds[i], )),
			tr_scores = tr_scores,
			tr_labels = tr_data[2], 
			tr_eval_t = tr_eval_t + tr_dm_t,
			val_scores = val_scores,
			val_labels = val_data[2], 
			val_eval_t = val_eval_t + val_dm_t,
			tst_scores = tst_scores,
			tst_labels = tst_data[2], 
			tst_eval_t = tst_eval_t + tst_dm_t
			)
		result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
		if save_result
			# tagsave(savef, result, safe = true)
			safesave(savef, result)
			verb ? (@info "Results saved to $savef") : nothing
		end
	end
end

"""
	experiment_bag(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)

Bag data and models arle evaluated differently from instance models such as VAE acting on aggregated bags.
If model is either NeuralStatistician or VAE instance, the score functions are merged such that computation
of certain scores is only done once (such as calcuating the instance likelihoods).
"""
function experiment_bag(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	# calculate the scores from likelihoods
	if parameters[:score] in ["reconstruction", "reconstruction-mean", "reconstruction-sampled"]
		experiment_likelihoods(score_fun, parameters, data, savepath; verb=verb, save_result=save_result, save_entries...)

	# calculate the scores from reconstructed input
	elseif parameters[:score] == "reconstructed_input"
		experiment_reconstructed_input(score_fun, parameters, data, savepath; verb=verb, save_result=save_result, save_entries...)
	end
end

"""
	experiment_likelihoods(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)

Evaluate likelihoods and save all the possible corrections.
Saves 8 scores in total:
- sum of instance likelihoods
- mean of instance likelihoods
- maximum of instance likelihoods
- sum + Poisson fit
- sum + LogNormal fit
- sum + logU
- sum + Poisson fit + logU
- sum + LogNormal fit + logU.
"""
function experiment_likelihoods(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	tr_data, val_data, tst_data = data

	# unpack data for easier manipulation
	train, _ = GroupAD.Models.unpack_mill(tr_data)
	val, _ = GroupAD.Models.unpack_mill(val_data)
	test, _ = GroupAD.Models.unpack_mill(tst_data)

	# fit MLE cardinality distribution to data
	bag_sizes = size.(train,2)
	poisson, poisson_eval_t = @timed fit_mle(Poisson, bag_sizes)
	lognormal, lognormal_eval_t = @timed fit_mle(LogNormal, bag_sizes)

	# time safety
	# if sampled likelihood takes too much time, do not calculate it
	# if the calculation would take approximately more than an hour
	# (there are 2,4 hours of time for calculation of scores for each model)
	length(train) > 100 ? idx = 100 : idx = length(train)
	_, safe_time = @timed score_fun.(train[1:idx])
	length_all = length(train) + length(val) + length(test)
	if safe_time * length_all / 100 > 3600
		@info "Sampling would take too long, not calculating sampled likelihood."
		return nothing
	end

	# calculate likelihoods
	tr_lh, tr_lh_t, _, _, _ = @timed score_fun.(train)
	val_lh, val_lh_t, _, _, _ = @timed score_fun.(val)
	tst_lh, tst_lh_t, _, _, _ = @timed score_fun.(test)
	# and save the evaluation times
	likelihoods_time = [tr_lh_t, val_lh_t, tst_lh_t]

	# calculate logU and its evaluation time
	logU, logU_eval_t, _, _, _ = @timed GroupAD.Models.calculate_logU(tr_lh)

	# possible scores as anonymous functions
	scores = [
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, fun=sum),
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, fun=mean),
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, fun=maximum),
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, poisson),
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, lognormal),
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, logU),
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, poisson, logU),
		(x, s) -> GroupAD.Models.rec_score_from_likelihood(x, s, lognormal, logU)
	]

	# helpful vectors
	eval_times = [
		0, 0, 0,
		poisson_eval_t,
		lognormal_eval_t,
		logU_eval_t,
		poisson_eval_t + logU_eval_t,
		lognormal_eval_t + logU_eval_t]
		
	score_names = [
		"sum",
		"mean",
		"maximum,",
		"poisson",
		"lognormal",
		"logU",
		"poisson+logU",
		"lognormal+logU"
	]

	# calculate all the possible scores and save them
	for (i, final_score) in enumerate(scores)
		# calculate final scores from likelihoods
		tr_scores, tr_eval_t, _, _, _ = @timed final_score(tr_lh, size.(train, 2))
		val_scores, val_eval_t, _, _, _ = @timed final_score(val_lh, size.(val, 2))
		tst_scores, tst_eval_t, _, _, _ = @timed final_score(tst_lh, size.(test, 2))

		# now save the stuff
		savef = joinpath(savepath, savename(merge(parameters, (type = score_names[i],)), "bson", digits=5))
		result = (
			parameters = merge(parameters, (type = score_names[i],)),
			tr_scores = tr_scores,
			tr_labels = tr_data[2], 
			tr_eval_t = tr_eval_t + eval_times[i] + likelihoods_time[1], # sum all times
			val_scores = val_scores,
			val_labels = val_data[2], 
			val_eval_t = val_eval_t + eval_times[i] + likelihoods_time[2],
			tst_scores = tst_scores,
			tst_labels = tst_data[2], 
			tst_eval_t = tst_eval_t + eval_times[i] + likelihoods_time[3]
		)
		result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
		if save_result
			tagsave(savef, result, safe = true)
			verb ? (@info "Results saved to $savef") : nothing
		end
	end
end

"""
	experiment_reconstructed_input(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)

Iterates over all score functions which need reconstructed input.
Currently used:
- Chamfer distance (from Flux3D.jl)
- MMD with Gaussian kernel
- MMD with IMQ kernel

Note: For MMD, the ideal bandwidth needs to be calculated.
"""
function experiment_reconstructed_input(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	tr_data, val_data, tst_data = data

	# unpack data for easier manipulation
	train, _ = GroupAD.Models.unpack_mill(tr_data)
	val, _ = GroupAD.Models.unpack_mill(val_data)
	test, _ = GroupAD.Models.unpack_mill(tst_data)

	# create reconstructed bags
	tr_rec, tr_rec_t, _, _, _ = @timed score_fun.(train)
	val_rec, val_rec_t, _, _, _ = @timed score_fun.(val)
	tst_rec, tst_rec_t, _, _, _ = @timed score_fun.(test)

	# convenience vectors
	eval_times = [tr_rec_t, val_rec_t, tst_rec_t]
	score_names = ["chamfer", "MMD-GaussianKernel", "MMD-IMQKernel"]

	# calculate bandwidth for MMD and save the time
	bw, bw_t, _, _, _ = @timed GroupAD.Models.mmd_bandwidth(train)
	bw_time = [0, bw_t, bw_t]

	# create the scores
	scores = [
		(x,y) -> GroupAD.Models.chamfer_score(x, y),
		(x,y) -> GroupAD.Models.mmd_score(x, y, GroupAD.Models.GaussianKernel, bw),
		(x,y) -> GroupAD.Models.mmd_score(x, y, GroupAD.Models.IMQKernel, bw)
	]

	# calculate and save the scores
	for (i, final_score) in enumerate(scores)
		tr_scores, tr_eval_t, _, _, _ = @timed final_score(train, tr_rec)
		val_scores, val_eval_t, _, _, _ = @timed final_score(val, val_rec)
		tst_scores, tst_eval_t, _, _, _ = @timed final_score(test, tst_rec)

		# now save the stuff
		savef = joinpath(savepath, savename(merge(parameters, (type = score_names[i],)), "bson", digits=5))
		result = (
			parameters = merge(parameters, (type = score_names[i],)),
			tr_scores = tr_scores,
			tr_labels = tr_data[2], 
			tr_eval_t = tr_eval_t + eval_times[i] + bw_time[i],
			val_scores = val_scores,
			val_labels = val_data[2], 
			val_eval_t = val_eval_t + eval_times[i] + bw_time[i],
			tst_scores = tst_scores,
			tst_labels = tst_data[2], 
			tst_eval_t = tst_eval_t + eval_times[i] + bw_time[i]
			)
		result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
		if save_result
			tagsave(savef, result, safe = true)
			verb ? (@info "Results saved to $savef") : nothing
		end
	end
end