# MILL data
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
	register(
		DataDep(
			"LHCO2020",
			"""
			Dataset: LHCO2020
			Dataset website: https://zenodo.org/record/6466204#.YqCRLXZBzIU
			Official website: https://lhco2020.github.io/homepage/
			
			LHC Olympics 2020 dataset for high-energy particle physics anomaly detection.
			""",
			[
				"https://zenodo.org/record/6466204/files/events_anomalydetection_v2.h5?download=1"
			],
			"5da8fda2ca78edf8fa5acc0af92fcb6a2d464ba7bf0fd75e3944a169b88cecc6",
			post_fetch_method = identity
		))
	# register(
	# 	DataDep(
	# 		"ModelNet10",
	# 		"""
	# 		Dataset: ModelNet10
	# 		Dataset website: https://github.com/AnTao97/PointCloudDatasets
	# 		Official website: http://modelnet.cs.princeton.edu/
			
	# 		Princeton ModelNet project of 3D point cloud objects. Using Point Cloud
	# 		dataset version.
	# 		""",
	# 		[
	# 			"https://www.dropbox.com/s/d5tnwg2legbd6rh/modelnet10_hdf5_2048.zip?dl=1"
	# 		],
	# 		"3dd357dfa1ea4bd858ffb2ff9032368f4dbe131a265fdf126a38bbf97075a8e3",
	# 		post_fetch_method = unpack
	# 	))
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
	obs_inds0 = vec(y.==0)
	obs_inds1 = vec(y.==1)
	bags0 = seqids2bags(bagids[obs_inds0])
	bags1 = seqids2bags(bagids[obs_inds1])

	# normalize to standard normal
	if normalize 
		x .= standardize(x)
	end
	
	# return normal and anomalous bags
	(normal = BagNode(ArrayNode(x[:,obs_inds0]), bags0), anomaly = BagNode(ArrayNode(x[:,obs_inds1]), bags1),)
end

### ModelNet10 point cloud dataset ###
# get_modelnet_datapath() = joinpath(datadep"ModelNet10")

### LHCO2020 R&D dataset ###
get_lhco_datapath() = joinpath(datadep"LHCO2020")

"""
    load_lhco(dataset = "events_anomalydetection_v2.h5")

This function loads the LHCO2020 dataset (the R&D version for now)
and processes it to get a Mill.jl datasets of normal and anomalous
samples.

The initialization script with data must be run already!
"""
function load_lhco(dataset = "events_anomalydetection_v2.h5")
	if "RD.h5" ∉ readdir(joinpath(get_lhco_datapath()))
		@warn(
			"""
			The data is not yet preprocessed. We will run the initialization scripts `lhco_init.jl` now.
			The script will fail if you are not using Python module and not have pandas package installed.
			"""
		)

		include(srcdir("lhco_init.jl"))
	end

	file = h5open(joinpath(get_lhco_datapath(), "RD.h5"), "r")
	normal = read(file["R&D"], "normal")
	anomaly = read(file["R&D"], "anomaly")

	d = BSON.load(joinpath(get_lhco_datapath(), "lhco_bagids.bson"))

    return (
        normal = BagNode(ArrayNode(normal), d[:bagids0]),
        anomaly = BagNode(ArrayNode(anomaly), d[:bagids1])
    )
end

### MNIST point-cloud ###
# unfortunately this is not available in a direct download format, so we need to do it awkwardly like this
"""
    get_mnist_point_cloud_datapath()

Get the absolute path of the MNIST point cloud dataset. Equals to `datadir("mnist_point_cloud")`.
"""
get_mnist_point_cloud_datapath() = datadir("mnist_point_cloud")

"""
	process_raw_mnist_point_cloud()

One-time processing of MNIST point cloud data that saves them in .bson files.
"""
function process_raw_mnist_point_cloud()
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
	load_mnist_point_cloud(;anomaly_class_ind::Int=1 noise=true, normalize=true)

Load the MNIST point cloud data. Anomaly class is chosen as
`anomaly_class = sort(unique(bag_labels))[anomaly_class_ind]`.
"""
function load_mnist_point_cloud(;anomaly_class_ind::Int=1, noise=true, normalize=true)
	dp = get_mnist_point_cloud_datapath()

	# check if the data is there
	if !ispath(dp) || !all(map(x->x in readdir(dp), ["test.bson", "train.bson"]))
		process_raw_mnist_point_cloud()
	end
	
	# load bson data and join them together
	test = load(joinpath(dp, "test.bson"))
	train = load(joinpath(dp, "train.bson"))
	bag_labels = vcat(train[:bag_labels], test[:bag_labels])
	labels = vcat(train[:labels], test[:labels])
	bagids = vcat(train[:bagids], test[:bagids] .+ length(train[:bag_labels]))
	data = Float32.(hcat(train[:data], test[:data]))

	# add uniform noise to dequantize data
	if noise
		data = data .+ rand(size(data)...)
	end
	
	# choose anomaly class
	anomaly_class = sort(unique(bag_labels))[anomaly_class_ind]
	@info "Loading MNIST point cloud with anomaly class: $(anomaly_class)."

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
		process_ember()
	end

	# load train data
	sy = open(joinpath(dp, "y_train.dat"), "r")
	y_train = Mmap.mmap(sy, Vector{Float32})
	close(sy)
	sx = open(joinpath(dp, "X_train.dat"), "r")
	X_train = Mmap.mmap(sx, Array{Float32,2}, (2381,length(y_train)))
	close(sx)

	# load test data
	sy = open(joinpath(dp, "y_test.dat"), "r")
	y_tst = Mmap.mmap(sy, Vector{Float32})
	close(sy)
	sx = open(joinpath(dp, "X_test.dat"), "r")
	X_tst = Mmap.mmap(sx, Array{Float32,2}, (2381,length(y_tst)))
	close(sx)

	X_train, y_train, X_tst, y_tst
end

function load_sift(filename::String="capsule_together")
	file = h5open(datadir("sift_mvtec", "$filename.h5"))
	data = file["data"][:,:]
	labels = file["labels"][:]
	bagids = file["sizes"][:]
	return data, labels, bagids
end

function load_mvtec(dataset::String="capsule_together")
	data, labels, bagids = load_sift(dataset)
	idxes = Mill.length2bags([sum(bagids .== c) for c in sort(unique(bagids))])
	bags = Mill.BagNode(Mill.ArrayNode(data), idxes)

	obs0 = labels .== 0
    obs1 = labels .== 1

	bagids0 = idxes[obs0]
	bagids1 = idxes[obs1]

	dataix0 = vcat([collect(x) for x in bagids0]...)
	dataix1 = vcat([collect(x) for x in bagids1]...)
	f = [first(b) for b in bagids1]
	l = [last(b) for b in bagids1]
	l = l .- f[1] .+ 1
	f = f .- f[1] .+ 1
	
	newbagids = [fi:li for (fi, li) in zip(f, l)]

    return (
        normal = BagNode(ArrayNode(hcat([data[:, b] for b in bagids0]...)), bagids0),
        anomaly = BagNode(ArrayNode(hcat([data[:, b] for b in bagids1]...)), newbagids)
    )
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
	reindex(bagnode, inds)

A faster implementation of Base.getindex.
"""
function reindex(bagnode, inds)
	obs_inds = bagnode.bags[inds]
	new_bagids = vcat(map(x->repeat([x[1]], length(x[2])), enumerate(obs_inds))...)
	data = bagnode.data.data[:,vcat(obs_inds...)]
	new_bags = GroupAD.seqids2bags(new_bagids)
	BagNode(ArrayNode(data), new_bags)
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

	tr_n, val_n, tst_n = map(is -> reindex(data_normal, is), split_inds)
	tr_a, val_a, tst_a = map(is -> reindex(data_anomalous, is), split_inds_anomalous)

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
    leave_one_in(data, seed=seed)

Prepares MNIST point cloud data in the leave-one-in setting.
The data have to be loaded with `load_data("MNIST_in",anomaly_class=class)`.

The class is intended to be the normal class in this case! Since normal class
would be underrepresented, the data is further cut such that validation and test
contain normal and anomalous samples in 1:1 ratio.
"""
function leave_one_in(data; seed=nothing)
    train, val, test = data
	# count number of normal samples in validation data
    l0 = sum(val[2] .== 0)

    # set seed and get indices of length == l0
    (seed === nothing) ? nothing : Random.seed!(seed)
    inds = sample(l0+1:length(val[2]),l0)

    # get the subset of anomalous data in validation and test data
    val_an = (GroupAD.reindex(val[1],inds),val[2][inds])
    test_an = (GroupAD.reindex(test[1],inds),test[2][inds])

    # create new data
    # number of normal and anomalous samples in val/test is 1:1
    val_new = (cat(val[1][1:l0],val_an[1]), vcat(val[2][1:l0],val_an[2]))
    test_new = (cat(test[1][1:l0],test_an[1]), vcat(test[2][1:l0],test_an[2]))

    return (train, val_new, test_new)
end


"""
    leave_one_out(data, seed=seed)

Prepares MNIST point cloud data in the leave-one-out setting.
The data should be loaded with `load_data("MNIST_out",anomaly_class=class)`.

The anomaly class in underrepresented, therefore the ratio of normal and anomalous
samples in validation and test data is made 1:1. Training data is left as before.
"""
function leave_one_out(data; seed=nothing)
	train, val, test = data
	# count number of anomalous samples in validation data
	l0 = sum(val[2] .== 1)
	(seed === nothing) ? nothing : Random.seed!(seed)
    inds = sample(1:length(val[2])-l0,l0)

	# get the subset of normal data in validation and test data
    val_n = (GroupAD.reindex(val[1],inds),val[2][inds])
    test_n = (GroupAD.reindex(test[1],inds),test[2][inds])

    # create new data
    # number of normal and anomalous samples in val/test is 1:1
	val_len = length(val[2])
    val_new = (cat(val_n[1],GroupAD.reindex(val[1],val_len-l0+1:val_len)),vcat(val_n[2],val[2][val_len-l0+1:end]))
	test_new = (cat(test_n[1],GroupAD.reindex(test[1],val_len-l0+1:val_len)),vcat(test_n[2],test[2][val_len-l0+1:end]))

    return (train, val_new, test_new)
end

"""
	load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
		contamination::Real=0.0, normalize=true, anomaly_class=1)

Returns 3 tuples of (data, labels) representing train/validation/test part. Arguments are the splitting ratios for normal data, seed and training data contamination.

Apart from `mnist_point_cloud` a.k.a. `MNIST`, the available datasets can be obtained by `readdir(GroupAD.get_mill_datapath())`.
"""
function load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, method = "leave-one-out",
	contamination::Real=0.0, kwargs...)

	# extract data and labels
	if dataset in ["MNIST", "mnist_point_cloud"]
		data_normal, data_anomalous, _, _ = load_mnist_point_cloud(;kwargs...)
	elseif occursin("events", dataset)
		data_normal, data_anomalous = load_lhco(dataset; kwargs...)
	elseif dataset in mvtec_datasets
		data_normal, data_anomalous = load_mvtec(dataset; kwargs...)
	elseif dataset == "modelnet"
		return load_modelnet(;kwargs...)
	else
		data_normal, data_anomalous = load_mill_data(dataset; kwargs...)
	end
	
	# now do the train/validation/test split
	if method == "leave-one-in"
		return train_val_test_split(data_anomalous, data_normal, ratios; seed=seed, contamination=contamination)
	else
		return train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
	end
end
