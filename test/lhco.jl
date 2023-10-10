using Test
using GroupAD
using PyCall
using Mill

"""
    load_lhco(dataset = "events_anomalydetection_v2.h5")
	
This function loads the LHCO2020 dataset (the R&D version for now)
and processes it to get a Mill.jl datasets of normal and anomalous
samples.

Note: PyCall.jl must be installed, Python/3.8 loaded with pandas,
tables packages installed. If this version is used, the path to
Python must be `/mnt/appl/software/Python/3.8.6-GCCcore-10.2.0/bin/python`.
"""
function load_lhco_from_pandas(dataset = "events_anomalydetection_v2.h5")
	file = joinpath(get_lhco_datapath(), dataset)
	if occursin("Python/3.8.6-GCCcore-10.2.0", read(`which python`, String))
    	pd = pyimport("pandas")
	end

    data = Array{Float32}[]
    labels = Int[]

    for i in 0:100000:1100000
        df_test = pd.read_hdf(file, start=i, stop=i+100000)
        data_array = df_test[:values]

        for row in eachrow(data_array)#[1:100000, :])
            label = row[end] |> Int
            push!(labels, label)
            zeroix = findfirst(x -> x == 0.0, row) |> Int
            d = row[1:zeroix-1]
            al = zeros(Float32, 3, length(d)÷3)
            al[1,:] = d[1:3:end]
            al[2,:] = d[2:3:end]
            al[3,:] = d[3:3:end]
            push!(data, al)
        end
    end

    obs0 = labels .== 0
    obs1 = labels .== 1

    ls0 = size.(data[obs0], 2)
    ls1 = size.(data[obs1], 2)

    bagids1 = Mill.length2bags(ls1)
    bagids0 = Mill.length2bags(ls0)

    return (
        normal = BagNode(ArrayNode(hcat(data[obs0]...)), bagids0),
        anomaly = BagNode(ArrayNode(hcat(data[obs1]...)), bagids1)
    )
end

@testset "LHCO data" begin
    @test 1 == 1
end