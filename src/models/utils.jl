"""
	function build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; activation::String="relu", lastlayer::String="")

Creates a chain with `nlayers` layers of `hdim` neurons with transfer function `activation`.
input and output dimension is `idim` / `odim`
If lastlayer is no specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense.
It is also possible to specify dimensions in a vector.

```juliadoctest
julia> build_mlp(4, 11, 1, 3, activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp([4, 11, 11, 1], activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp(4, 11, 1, 3, activation="relu", lastlayer="tanh")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, tanh))
```
"""
build_mlp(ks::Vector{Int}, fs::Vector) = Flux.Chain(map(i -> Dense(i[2],i[3],i[1]), zip(fs,ks[1:end-1],ks[2:end]))...)

build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; kwargs...) =
	build_mlp(vcat(idim, fill(hdim, nlayers-1)..., odim); kwargs...)

function build_mlp(ks::Vector{Int}; activation::String = "relu", lastlayer::String = "")
	activation = (activation == "linear") ? "identity" : activation
	fs = Array{Any}(fill(eval(:($(Symbol(activation)))), length(ks) - 1))
	if !isempty(lastlayer)
		fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
	end
	build_mlp(ks, fs)
end

"""
	unpack_mill(dt<:Tuple{BagNode,Any})

Takes Tuple of BagNodes and bag labels and returns
both in a format that is fit for Flux.train!
"""
function unpack_mill(dt::T) where T <: Tuple{BagNode,Any}
    bag_labels = dt[2]
	bag_data = [dt[1][i].data.data for i in 1:Mill.length(dt[1])]
    return bag_data, bag_labels
end
"""
	unpack_mill(dt<:Tuple{Array,Any})

To ensure reproducibility of experimental loop and the fit! function for models,
this function returns unchanged input, if input is a Tuple of Arrays.
Used in toy problems.
"""
function unpack_mill(dt::T) where T <: Tuple{Array,Any}
    bag_labels = dt[2]
	bag_data = dt[1]
    return bag_data, bag_labels
end

"""
    RandomBagBatches(data;batchsize::Int=32,randomize=true)

Creates random batch for bag data which are an array of
arrays. If data length is smaller than batchsize, returns
the full data.
"""
function RandomBagBatches(data;batchsize::Int=32,randomize=true)
    l = length(data)
	if batchsize > l
		return data
	end
    if randomize
        idx = sample(1:l,batchsize)
		return (data)[idx]
    else
		idx = sample(1:l-batchsize)
        return data[idx:idx+batchsize-1]
    end
end