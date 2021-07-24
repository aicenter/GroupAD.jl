"""
    generate_cardinality_toy(Nn,Na;type=1,seed=nothing)

Generate dataset from "Apple Paper". There are three types
of datasets. For more info go to Figure 7 in
Model-Based Learning for Point Patter Data.
"""
function generate_cardinality_toy(Nn,Na;type=1,seed=nothing)
    # set seed
    (seed === nothing) ? nothing : Random.seed!(seed)

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
    
    # reset seed
    (seed === nothing) ? nothing : Random.seed!()

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
    normal, anomalous = generate_cardinality_toy(Nn*3, Na*2, type=type, seed=seed)
    
    # train data
    train_data = normal[1:Nn]
    train_labels = zeros(Nn)

    # validation data
    val_data = vcat(normal[Nn+1:Nn*2],anomalous[1:Na])
    val_labels = vcat(zeros(Nn), ones(Na))

    # test data
    test_data = vcat(normal[2*Nn+1:end],anomalous[Na+1:end])
    test_labels = vcat(zeros(Nn), ones(Na))

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