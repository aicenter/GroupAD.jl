using GroupAD
using GroupAD: load_data
using GroupAD.Models: unpack_mill
using BSON
using DataFrames
using Latexify
using Mill

dp = GroupAD.get_mnist_point_cloud_datapath()
test = load(joinpath(dp, "test.bson"))
train = load(joinpath(dp, "train.bson"))

bag_labels = vcat(train[:bag_labels], test[:bag_labels])
labels = vcat(train[:labels], test[:labels])
bagids = vcat(train[:bagids], test[:bagids] .+ length(train[:bag_labels]))
data = Float32.(hcat(train[:data], test[:data]))
data = data .+ rand(size(data)...)
data = GroupAD.standardize(data)

obs = GroupAD.seqids2bags(bagids)
mill_mnist = BagNode(ArrayNode(data), obs)

mnist, _ = unpack_mill((mill_mnist, []))

mnist_df = DataFrame(:data => mnist, :class => bag_labels)
g = groupby(mnist_df, :class)

how_many = map(x -> size(x,1), g)
classes = map(x -> x[1,:class], g)

mean_card = map(x -> mean(size.(x[:, :data], 2)), g)
med_card = map(x -> median(size.(x[:, :data], 2)), g)
min_card = map(x -> minimum(size.(x[:, :data], 2)), g)
max_card = map(x -> maximum(size.(x[:, :data], 2)), g)

mnist_summary = DataFrame(
    :class => classes,
    :quantity => how_many,
    :mean_cardinality => mean_card,
    :minimum_cardinality => min_card,
    :median_cardinality => med_card,
    :maximum_cardinality => max_card
)
sort!(mnist_summary, :class)

tex = latexify(mnist_summary, env=:tabular, fmt=x->round(x, digits=1), booktabs=true)