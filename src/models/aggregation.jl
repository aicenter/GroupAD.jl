"""
	aggregate(x::Mill.BagNode, agf::Function)
	aggregate(data::Tuple, agf::Function)

Aggregate the individual bags into vectors using an aggregation function.
"""
function aggregate(x::Mill.BagNode, agf::Function)
	# preallocate the array
	# we could do cat(map...) but that stops working for large bags
	y = zeros(eltype(x.data.data), size(x.data.data,1), length(x.bags))
	for (i,bag) in enumerate(x.bags)
		y[:,i] = agf(x.data.data[:,bag], dims=2)
	end
	y
end

aggregate(data::Tuple, agf::Function) = Tuple(map(d->(aggregate(d[1], agf), d[2]), data))
