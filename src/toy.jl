"""
    generate_cardinality_toy(Nn,Na;type=1)

Generate dataset from "Apple Paper". There are three types
of datasets. For more info go to Figure 7 in
Model-Based Learning for Point Patter Data.
"""
function generate_cardinality_toy(Nn,Na;type=1)
    # normal data is always the same
    normal_d = MvNormal([6,6],sqrt.([6,2]))
    anomalous_d = MvNormal([1,1],sqrt.([6,2]))
    normal = [rand(normal_d,rand(Poisson(37))) for i in 1:Nn]

    if type==1
         # 1) novelty separated, overlap in cardinality
        anomalous = [rand(anomalous_d,rand(Poisson(37))) for i in 1:Na]
    elseif type==2
        # 2) novelty and normal data partially overlap in feature and cardinality
        mod(Na,3) != 0 ? error("Na can't be divided by 3!") : nothing
        anomalous = vcat(
            [rand(normal_d,rand(Poisson(6))) for i in 1:Nn÷3-2],
            [rand(normal_d,rand(Poisson(100))) for i in 1:Na÷3-2],
            [rand(anomalous_d,rand(Poisson(37))) for i in 1:Na÷3-2],
        )
        # shuffle the data so that the categories are mixed
        shuffle!(anomalous)
    else
        # 3) low cardinality novelty overlap with normal data
        anomalous = [rand(normal_d,rand(Poisson(6))) for i in 1:Na]
    end
    return normal, anomalous
end

"""
    create_apple_toy(Nn=30, Na=30; type=1)

Returns data for toy problem from the Apple paper (Model-Based Learning for Point Pattern Data).
Type specifies the scenario used. Nn is the number of normal samples, Na number of anomalous samples.
There is no contamination parameter.
"""
function create_apple_toy(Nn=30, Na=30; type=1, seed=nothing)
    # train data
    normal, _ = generate_cardinality_toy(Nn, Na, type=type)
    train_data = normal
    train_labels = zeros(Int, Nn)
    # validation data
    n,a = generate_cardinality_toy(Nn, Na,type=type)
    val_data = vcat(n,a)
    val_labels = vcat(zeros(Int,Nn),ones(Int,Na))
    # test data
    n,a = generate_cardinality_toy(Nn, Na,type=type)
    test_data = vcat(n,a)
    test_labels = vcat(zeros(Int,Nn),ones(Int,Na))
    return ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
end

function load_data(dataset::String, Nn, Na; type=1, seed=nothing)
    if dataset == "toy"
        data = create_apple_toy(Nn, Na; type=type, seed=seed)
        return data
    else
        nothing
    end
end