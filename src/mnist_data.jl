using Mill
# split without test dataset
function train_val_test_split(data_normal::BagNode, data_anomalous::BagNode, l_normal, l_anomaly, ratios=(0.6, 0.2, 0.2); seed=nothing,
    contamination::Real=0.0)
    # split the normal data, add some anomalies to the train set and divide
    # the rest between validation and test
    (0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")

    # split normal indices
    indices = 1:nobs(data_normal)
    split_inds = train_val_test_inds(indices, ratios; seed=seed)

    # select anomalous indices
    # select anomalous indices
    indices_anomalous = 1:nobs(data_anomalous)
    vtr = (1 - contamination) / 2 # validation/test ratio
	if contamination == 0 && ratios[3] == 0
    	split_inds_anomalous = train_val_test_inds(indices_anomalous, (contamination, 1, 0); seed=seed)
	else
		split_inds_anomalous = train_val_test_inds(indices_anomalous, (contamination, vtr, vtr); seed=seed)
	end

    # split to train, validation, test
	tr_n, val_n, tst_n = map(is -> data_normal[is], split_inds)
	tr_nl, val_nl, tst_nl = map(is -> l_normal[is], split_inds)
	tr_a, val_a, tst_a = map(is -> data_anomalous[is], split_inds_anomalous)
	tr_al, val_al, tst_al = map(is -> l_anomaly[is], split_inds_anomalous)

    # cat it together
    tr_x = cat(tr_n, tr_a)
    val_x = cat(val_n, val_a)
    tst_x = cat(tst_n, tst_a)

    # now create labels (normal/anomaly)
    tr_y = vcat(zeros(Float32, nobs(tr_n)), ones(Float32, nobs(tr_a)))
    val_y = vcat(zeros(Float32, nobs(val_n)), ones(Float32, nobs(val_a)))
	tst_y = vcat(zeros(Float32, nobs(tst_n)), ones(Float32, nobs(tst_a)))
	
	# create labels as numbers from 0 to 9
	tr_l = vcat(tr_nl, tr_al)
	val_l = vcat(val_nl, val_al)
	tst_l = vcat(tst_nl, tst_al)

    #(tr_x, tr_y, tr_l), (val_x, val_y, val_l), (tst_x, tst_y, tst_l)
	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end

"""

	load_mnist(dataset::String,anomaly_number::Int;noise=true,normalize=true)

Function to load MNIST dataset. `dataset::String` can be either train or test.
Choose `anomaly_number` to mark one digit as anomalous. The data is only integers,
if you want to dequantitze the data, `noise=true` adds uniform noise. Data can be
normalized to N(0,1) if `normalize=true`.
"""
function load_mnist(dataset::String,anomaly_number::Int;noise=true,normalize=true)
	# load data
	data = BSON.load(datadir("2d_mnist","$dataset.BSON"))
	x = Float32.(data[:data])
	# add uniform noise to dequantize data
	if noise
		x = x .+ rand(Uniform(0,1),size(x))
	end
	bagids = data[:bagids]
	y = data[:labels]
	bag_labels = data[:bag_labels]
	
	# split to 0/1 classes
	c0 = y .!= anomaly_number
	c1 = y .== anomaly_number
	bags0 = seqids2bags(bagids[c0[:]])
	bags1 = seqids2bags(bagids[c1[:]])

	# bag labels
	l0 = bag_labels .!= anomaly_number
	l1 = bag_labels .== anomaly_number
	l_normal = bag_labels[l0]
	l_anomaly = bag_labels[l1]

	# transform data
	if normalize
		x = standardize(x)
	end

	# return normal and anomalous bags (and their labels)
	(normal = BagNode(ArrayNode(x[:,c0[:]]), bags0), anomaly = BagNode(ArrayNode(x[:,c1[:]]), bags1), l_normal = l_normal, l_anomaly = l_anomaly)
end

"""

    MNIST_train_test(anomaly_number,ratios=(0.8,0.2,0);noise=true,normalize="normal",seed=31,contamination::Real=0.0)

Function to load MNIST dataset where train dataset is used for
training and validation, test dataset only used for testing.

Easiest way to load data is to use:

    `train, validation, test = MNIST_train_test(3)`

which loads data, splits training dataset and marks number 3 as anomalous.
Ratios are different: train dataset is split to 0.8/0.2 for train/validation
and test dataset is the full test dataset as defined in original MNIST.
"""
function MNIST_train_test(anomaly_number,ratios=(0.8,0.2,0);noise=true,normalize=true,seed=31,contamination::Real=0.0)
    dtn, dta, ln, la = load_mnist("train",anomaly_number;noise=noise,normalize=normalize)
    train, validation, _ = train_val_test_split(dtn,dta,ln,la,ratios,seed=seed,contamination=contamination)
    dsn, dsa, lsn, lsa = load_mnist("test",anomaly_number;noise=noise,normalize=normalize)
    test = (cat(dsn, dsa), vcat(zeros(nobs(dsn))), ones(nobs(dsa)), vcat(ln, la))
    return train, validation, test
end
