"""
	unpack_mill(dt)

Takes Tuple of BagNodes and bag labels and returns
both in a format that is fit for Flux.train!
"""
function unpack_mill(dt; add_c=false)
    bag_labels = dt[2]
	bag_data = [dt[1][i].data.data for i in 1:length(bag_labels)]
    if add_c
        return add_cardinality(bag_data), bag_labels
    else
        return bag_data, bag_labels
    end
end

function add_cardinality(data)
    for i in 1:length(data)
        bag = data[i]
        n = size(bag,2)
        nvec = Float32.(repeat([n],n) .+ rand(n))
        data[i] = vcat(bag, nvec')
    end
    return data
end