using GroupAD
using Mill
using DrWatson
using ProgressBars: tqdm

function unpack_mill(dt::T) where T <: Tuple{BagNode,Any}
    bag_labels = dt[2]
	bag_data = [dt[1][i].data.data for i in 1:Mill.length(dt[1])]
    return bag_data, bag_labels
end

mill_short =  ["BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2", "Mutagenesis1",
    "Mutagenesis2", "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein", "Tiger", "UCSBBreastCancer", "WinterWren"]

df = Dict()
for dataset_name in tqdm(mill_short)
    data = GroupAD.load_data(dataset_name)
    train, val, test = data
    train_x, train_y = unpack_mill(train)
    val_x, val_y = unpack_mill(val)
    test_x, test_y = unpack_mill(test)
    bag_sizes = vcat(
        size.(train_x, 2),
        size.(val_x, 2),
        size.(test_x, 2)
    )
    bag_labels = vcat(train_y, val_y, test_y)
    df[dataset_name] = Dict(:sizes => bag_sizes, :labels => bag_labels)
end

save("sizes_and_labels.bson", df)