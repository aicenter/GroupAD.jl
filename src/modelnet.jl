using HDF5
using Random
using StatsBase
using Mill

function tensors_to_mill(data)
    n = size(data, 2)
    lengths = repeat([n], size(data, 3))
    idxes = Mill.length2bags(lengths)
    flatten = reshape(data, 3, n * size(data, 3))
    return BagNode(ArrayNode(flatten), idxes)
end

function load_modelnet(npoints=2048; method="chair", validation::Bool=true, ratio=0.2, seed::Int=666, kwargs...)
    # method is actually the class that we want out,
    # because this is the only way not to change too many lines
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = _load_modelnet10(npoints, method; validation=validation, ratio=ratio, seed=seed)
    X_train = tensors_to_mill(X_train)
    X_val = tensors_to_mill(X_val)
    X_test = tensors_to_mill(X_test)
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
end

function train_test_split(X, y, ratio=0.2; seed=nothing)
    # simple util function
    (seed!==nothing) ? Random.seed!(seed) : nothing

    N = size(X,3)
    idx_samples = sample(1:N, Int(floor(N*ratio)), replace=false)
    idx_bool = zeros(Bool,N)
    idx_bool[idx_samples] .= true
    
    X_val = X[:,:,idx_bool]
    Y_val = y[idx_bool]
    X_train = X[:,:,.!idx_bool]
    Y_train = y[.!idx_bool]

    (seed!==nothing) ? Random.seed!() : nothing
    return (X_train, Y_train), (X_val, Y_val)
end


function _load_modelnet10(npoints=2048, type="all"; validation::Bool=true, ratio=0.2, seed::Int=666)
    """
    npoints     ... Number of points per object ( 512 / 1024 / 2048 )
    type        ... Type data -> \"all\" or one-class name e.g. \"chair\", \"monitor\"
    validation  ... Return validation set (\"true\") or not (\"false\")
    seed        ... Random seed for validation split.
    """

    #load data
    data = HDF5.h5open("/home/maskomic/projects/GroupAD.jl/data/modelnet10_$(npoints).h5")
    X_train, X_test, Y_train, Y_test = data["X_train"]|>read, data["X_test"]|>read, data["Y_train"]|>read, data["Y_test"]|>read

    titles = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

    # if validation
    #     (X_train,Y_train), (X_val,Y_val) = train_test_split(X_train, Y_train, ratio, seed=seed)
    #     if type in titles
    #         idx = findmax(titles .== type)[2]
    #         X_train = X_train[:, :, Y_train .== idx]
    #         Y_train = zeros(Bool,size(Y_train[Y_train .== idx]))
    #         Y_val = Y_val .!= idx
    #         Y_test = Y_test .!= idx
    #     end
    #     data = ((X_train, Y_train), (X_val, Y_val), (X_test, Y_test)) 
    # else
    #     if type in titles
    #         idx = findmax(titles .== type)[2]
    #         X_train = X_train[:, :, Y_train .== idx]
    #         Y_train = zeros(Bool,size(Y_train[Y_train .== idx]))
    #         Y_test = Y_test .!= idx
    #     end
    #     data = ((X_train, Y_train), (X_test, Y_test)) 
    # end
    (X_train,Y_train), (X_val,Y_val) = train_test_split(X_train, Y_train, ratio, seed=seed)
    idx = findmax(titles .== type)[2]
    X_train = X_train[:, :, Y_train .== idx]
    Y_train = zeros(Bool,size(Y_train[Y_train .== idx]))
    Y_val = Y_val .!= idx
    Y_test = Y_test .!= idx
    data = ((X_train, Y_train), (X_val, Y_val), (X_test, Y_test)) 
    return data
end