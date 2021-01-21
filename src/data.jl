"""
    get_mill_datapath()
Get the absolute path of MIProblems data. !!! Hardcoded path to ../MIProblems
TODO fix this with DataDeps.
"""
get_mill_datapath() = joinpath(dirname(dirname(@__FILE__)), "../../MIProblems")

function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

function y_on_instances(bagnode, y)
	# yout = reduce(cat, cat([y[p]*ones(length(bagnode.bags[p]))[:] for p=1:nobs(bagnode)]...,dims=1))
	yforbag = (p)->y[p]*ones(typeof(y[1]),length(bagnode.bags[p]))
	yout = mapreduce(yforbag, vcat, 1:length(bagnode.bags))
end


"""
	load_mill_data(dataset::String)
Loads basic MIProblems data. For a list of available datasets, check `GenerativeAD.Datasets.mill_datasets`.
"""
function load_mill_data(dataset::String)
	mdp = get_mill_datapath()
	x=readdlm("$mdp/$(dataset)/data.csv",'\t',Float32)
	bagids = readdlm("$mdp/$(dataset)/bagids.csv",'\t',Int)[:]
	y = readdlm("$mdp/$(dataset)/labels.csv",'\t',Int)
	
	# plit to 0/1 classes
	c0 = y.==0
	c1 = y.==1
	bags0 = seqids2bags(bagids[c0[:]])
	bags1 = seqids2bags(bagids[c1[:]])

	# normalize to standard normal
	x.=UCI.normalize(x)
	# return normal and anomalous bags
	(normal = BagNode(ArrayNode(x[:,c0[:]]), bags0), anomaly = BagNode(ArrayNode(x[:,c1[:]]), bags1),)
end

import Base.length
Base.length(B::BagNode)=length(B.bags.bags)

"""
    train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)

Split indices.
"""
function train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)
    (sum(ratios) == 1 && length(ratios) == 3) ? nothing :
    	error("ratios must be a vector of length 3 that sums up to 1")

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # set number of samples in individual subsets
    n = length(indices)
    ns = cumsum([x for x in floor.(Int, n .* ratios)])

    # scramble indices
    _indices = sample(indices, n, replace=false)

    # restart seed
    (seed == nothing) ? nothing : Random.seed!()

    # return the sets of indices
    _indices[1:ns[1]], _indices[ns[1]+1:ns[2]], _indices[ns[2]+1:ns[3]]
end

"""
	train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); seed=nothing,
	    	method="leave-one-out", contamination::Real=0.0)

Split data.
"""
function train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); 
	seed=nothing, method="leave-one-out", contamination::Real=0.0)

	# split the normal data, add some anomalies to the train set and divide
	# the rest between validation and test
	(0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
	any(method .== ["leave-one-out","leave-one-in"]) ? nothing : error("unknown method")
	nd = ndims(data_normal.data.data) # differentiate between 2D tabular and 4D image data

	# split normal indices
	indices = 1:length(data_normal.bags.bags)
	split_inds = train_val_test_inds(indices, ratios; seed=seed)

	# select anomalous indices
	indices_anomalous = 1:size(data_anomalous, nd)
	na_tr = floor(Int, length(split_inds[1])*contamination/(1-contamination))
	tr = na_tr/length(indices_anomalous) # training ratio
	vtr = (1 - tr)/2 # validation/test ratio
	split_inds_anomalous = train_val_test_inds(indices_anomalous, (tr, vtr, vtr); seed=seed)

	tr_n, val_n, tst_n = map(is -> data_normal[is], split_inds)
	tr_a, val_a, tst_a = map(is -> data_anomalous[is], split_inds_anomalous)

	# cat it together
	if method == "leave-one-in"
		tr_x = tr_n
		tr_y = zeros(Float32, size(tr_x, nd))
	else	
		tr_x = cat(tr_n, tr_a, dims = nd)
		tr_y = vcat(zeros(Float32, size(tr_n, nd)), ones(Float32, size(tr_a,nd)))
	end
	val_x = cat(val_n, val_a, dims = nd)
	tst_x = cat(tst_n, tst_a, dims = nd)

	# now create labels
	val_y = vcat(zeros(Float32, size(val_n, nd)), ones(Float32, size(val_a,nd)))
	tst_y = vcat(zeros(Float32, size(tst_n, nd)), ones(Float32, size(tst_a,nd)))

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end

"""
	load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
	method="leave-one-out", contamination::Real=0.0)

Returns 3 tuples of (data, labels) representing train/validation/test part. Arguments are the splitting
ratios for normal data, seed and training data contamination.

For a list of available datasets, check `GenerativeAD.Datasets.uci_datasets`, `GenerativeAD.Datasets.other_datasets`
and `GenerativeAD.Datasets.mldatasets`.
"""
function load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
	method="leave-one-out", contamination::Real=0.0, kwargs...)

	# extract data and labels
	data_normal, data_anomalous = load_mill_data(dataset; kwargs...)
	
	# now do the train/validation/test split
	if method=="leave-one-in"
		return train_val_test_split(data_anomalous, data_normal, ratios; seed=seed, method=method, contamination=ratios[1])
	else
		return train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
	end
end
