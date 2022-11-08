timestart = time()

using GroupAD
using GroupAD: get_lhco_datapath
using Mill
using PyCall
using HDF5
using DrWatson

@info "Running a script to process LHCO R&D dataset and save it in nice format."

file = joinpath(get_lhco_datapath(), "events_anomalydetection_v2.h5")

# if occursin("Python/3.8.6-GCCcore-10.2.0", read(`which python`, String))
pd = pyimport("pandas")

data = Array{Float32}[]
labels = Int8[]

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

normal = hcat(data[obs0]...)
anomaly = hcat(data[obs1]...)

newfile = h5open("$(get_lhco_datapath())/RD.h5", "w")
g = create_group(newfile, "R&D")

d_normal = create_dataset(g, "normal", Float32, size(normal))
d_anomaly = create_dataset(g, "anomaly", Float32, size(anomaly))

write(d_normal, normal)
write(d_anomaly, anomaly)

close(newfile)

@info "Data saved."

save("$(get_lhco_datapath())/lhco_bagids.bson", Dict(:bagids0 => bagids0, :bagids1 => bagids1))

@info "Labels for Mill.jl saved."

@info "Processing finished in $(time() - timestart) seconds."