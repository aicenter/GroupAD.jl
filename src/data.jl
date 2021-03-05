function __init__()
	register(
		DataDep(
			"MIProblems",
			"""
			Dataset: MIProblems
			Authors: Collected by Tomáš Pevný
			Website: https://github.com/pevnak/MIProblems/
			
			Datasets that represent Multiple-Instance problems. 
			""",
			[
				"https://github.com/pevnak/MIProblems/archive/master.zip"
			],
			"9ab2153807d24143d4d0af0b6f4346e349611a4b85d5e31b06d56157b8eed990",
			post_fetch_method = unpack
		))
	register(
		DataDep(
			"EMBER",
			"""
			Dataset: EMBER 2018
			Website: https://github.com/elastic/ember
			
			EMBER dataset of PE data.
			""",
			[
				"https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
			],
			"b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812",
			post_fetch_method = unpack
		))
end

### MILL data ###
"""
    get_mill_datapath()

Get the absolute path of MIProblems data.
"""
get_mill_datapath() = joinpath(datadep"MIProblems", "MIProblems-master")

"""
	load_mill_data(dataset::String; normalize=true)

Loads basic MIProblems data. For a list of available datasets, do `readdir(GroupAD.get_mill_datapath())`.
"""
function load_mill_data(dataset::String; normalize=true)
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
	if normalize 
		x .= standardize(x)
	end
	
	# return normal and anomalous bags
	(normal = BagNode(ArrayNode(x[:,c0[:]]), bags0), anomaly = BagNode(ArrayNode(x[:,c1[:]]), bags1),)
end

### MNIST point-cloud ###
# unfortunately this is not available in a direct download format, so we need to do it awkwardly like this
"""
    get_mnist_point_cloud_datapath()

Get the absolute path of the MNIST point cloud dataset. Equals to `datadir("mnist_point_cloud")`.
"""
get_mnist_point_cloud_datapath() = datadir("mnist_point_cloud")

"""
	_process_raw_mnist_point_cloud()

One-time processing of MNIST point cloud data that saves them in .bson files.
"""
function _process_raw_mnist_point_cloud()
	dp = get_mnist_point_cloud_datapath()

	# check if the path exists
	if !ispath(dp) || length(readdir(dp)) == 0 || !all(map(x->x in readdir(dp), ["test.csv", "train.csv"]))
		mkpath(dp)
		error("MNIST point cloud data are not present. Unfortunately no automated download is available. Please download the `train.csv.zip` and `test.csv.zip` files manually from https://www.kaggle.com/cristiangarcia/pointcloudmnist2d and unzip them in `$(dp)`.")
	end
	
	@info "Processing raw MNIST point cloud data..."
	for fs in ["test", "train"]
		indata = readdlm(joinpath(dp, "$fs.csv"),',',Int32,header=true)[1]
		bag_labels = indata[:,1]
		labels = []
		bagids = []
		data = []
		for (i,row) in enumerate(eachrow(indata))
			# get x data and specify valid values
			x = row[2:3:end]
			valid_inds = x .!= -1
			x = reshape(x[valid_inds],1,:)
			
			# get y and intensity
			y = reshape(row[3:3:end][valid_inds],1,:)
			v = reshape(row[4:3:end][valid_inds],1,:)

			# now append to the lists
			push!(labels, repeat([row[1]], length(x)))
			push!(bagids, repeat([i], length(x)))
			push!(data, vcat(x,y,v))
		end
		outdata = Dict(
			:bag_labels => bag_labels,
			:labels => vcat(labels...),
			:bagids => vcat(bagids...),
			:data => hcat(data...)
			)
		bf = joinpath(dp, "$fs.bson")
		save(bf, outdata)
		@info "Succesfuly processed and saved $bf"
	end
	@info "Done."
end

"""
	load_mnist_point_cloud(;anomaly_class::Int=1 noise=true, normalize=true)

Load the MNIST point cloud data.
"""
function load_mnist_point_cloud(;anomaly_class::Int=1, noise=true, normalize=true)
	dp = get_mnist_point_cloud_datapath()

	# check if the data is there
	if !ispath(dp) || !all(map(x->x in readdir(dp), ["test.bson", "train.bson"]))
		_process_raw_mnist_point_cloud()
	end
	
	# load bson data and join them together
	test = load(joinpath(dp, "test.bson"))
	train = load(joinpath(dp, "train.bson"))
	bag_labels = vcat(train[:bag_labels], test[:bag_labels])
	labels = vcat(train[:labels], test[:labels])
	bagids = vcat(train[:bagids], test[:bagids])
	data = Float32.(hcat(train[:data], test[:data]))

	# add uniform noise to dequantize data
	if noise
		data = data .+ rand(size(data)...)
	end
	
	# split to 0/1 classes - instances
	obs_inds0 = labels .!= anomaly_class
	obs_inds1 = labels .== anomaly_class
	obs0 = seqids2bags(bagids[obs_inds0])
	obs1 = seqids2bags(bagids[obs_inds1])

	# split to 0/1 classes - bags
	bag_inds0 = bag_labels .!= anomaly_class
	bag_inds1 = bag_labels .== anomaly_class	
	l_normal = bag_labels[bag_inds0]
	l_anomaly = bag_labels[bag_inds1]

	# transform data
	if normalize
		data = standardize(data)
	end

	# return normal and anomalous bags (and their labels)
	(normal = BagNode(ArrayNode(data[:,obs_inds0]), obs0), anomaly = BagNode(ArrayNode(data[:,obs_inds1]), obs1), l_normal = l_normal, l_anomaly = l_anomaly)
end

### EMBER data
"""
    get_ember_datapath()

Get the absolute path of raw EMBER data.
"""
get_ember_datapath() = joinpath(datadep"EMBER", "ember2018")

"""
	process_ember()

Extract features from the EMBER raw PE data using the `ember` library. Requires Python3.
"""
function process_ember()
	dp = get_ember_datapath()
	bashf = abspath(joinpath(pathof(GroupAD), "../../scripts/ember_init/ember_init.sh"))
	cmd = `$bashf`
	run(cmd)
end

"""
	load_ember(;normalize=true)

Load the EMBER vectorized data.
"""
function load_ember(;normalize=true)
	dp = get_ember_datapath()

	# check if the data is there
	if !all(map(x->x in readdir(dp), ["X_test.dat", "X_train.dat", "y_test.dat", "y_train.dat", "metadata.csv"]))
		_process_ember()
	end
	


"""
	seqids2bags(bagids)

"""
function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

"""
	y_on_instances(bagnode, y)

"""
function y_on_instances(bagnode, y)
	# yout = reduce(cat, cat([y[p]*ones(length(bagnode.bags[p]))[:] for p=1:nobs(bagnode)]...,dims=1))
	yforbag = (p)->y[p]*ones(typeof(y[1]),length(bagnode.bags[p]))
	yout = mapreduce(yforbag, vcat, 1:length(bagnode.bags))
end

"""
    standardize(Y)

Scales down a 2 dimensional array so it has approx. standard normal distribution. 
Instance = column. 
"""
function standardize(Y::Array{T,2}) where T<:Real
    M, N = size(Y)
    mu = Statistics.mean(Y,dims=2);
    sigma = Statistics.var(Y,dims=2);

    # if there are NaN present, then sigma is zero for a given column -> 
    # the scaled down column is also zero
    # but we treat this more economically by setting the denominator for a given column to one
    # also, we deal with numerical zeroes
    den = sigma
    den[abs.(den) .<= 1e-15] .= 1.0
    den[den .== 0.0] .= 1.0
    den = repeat(sqrt.(den), 1, N)
    nom = Y - repeat(mu, 1, N)
    nom[abs.(nom) .<= 1e-8] .= 0.0
    Y = nom./den
    return Y
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
	contamination::Real=0.0)

Split data.
"""
function train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); 
	seed=nothing, contamination::Real=0.0)

	# split the normal data, add some anomalies to the train set and divide
	# the rest between validation and test
	(0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
	nd = ndims(data_normal.data.data) # differentiate between 2D tabular and 4D image data

	# split normal indices
	indices = 1:length(data_normal.bags.bags)
	split_inds = train_val_test_inds(indices, ratios; seed=seed)

	# select anomalous indices
	indices_anomalous = 1:length(data_anomalous.bags.bags)
	na_tr = floor(Int, length(split_inds[1])*contamination/(1-contamination))
	tr = na_tr/length(indices_anomalous) # training ratio
	vtr = (1 - tr)/2 # validation/test ratio
	split_inds_anomalous = train_val_test_inds(indices_anomalous, (tr, vtr, vtr); seed=seed)

	tr_n, val_n, tst_n = map(is -> data_normal[is], split_inds)
	tr_a, val_a, tst_a = map(is -> data_anomalous[is], split_inds_anomalous)

	# cat it together
	tr_x = cat(tr_n, tr_a, dims = nd)
	val_x = cat(val_n, val_a, dims = nd)
	tst_x = cat(tst_n, tst_a, dims = nd)

	# now create labels
	tr_y = vcat(zeros(Float32, length(tr_n)), ones(Float32, length(tr_a)))
	val_y = vcat(zeros(Float32, length(val_n)), ones(Float32, length(val_a)))
	tst_y = vcat(zeros(Float32, length(tst_n)), ones(Float32, length(tst_a)))

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end

"""
	load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
		contamination::Real=0.0, normalize=true, anomaly_class=1)

Returns 3 tuples of (data, labels) representing train/validation/test part. Arguments are the splitting ratios for normal data, seed and training data contamination.

Apart from `mnist_point_cloud` a.k.a. `MNIST`, the available datasets can be obtained by `readdir(GroupAD.get_mill_datapath())`.
"""
function load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
	contamination::Real=0.0, kwargs...)

	# extract data and labels
	if dataset in ["MNIST", "mnist_point_cloud"]
		data_normal, data_anomalous, bag_labels_normal, bag_labels_anomalous = load_mnist_point_cloud(;kwargs...)
	else
		data_normal, data_anomalous = load_mill_data(dataset; kwargs...)
	end
	
	# now do the train/validation/test split
	return train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
end
