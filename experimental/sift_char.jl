using GroupAD
using GroupAD: load_data, load_mvtec, load_mill_data, load_mnist_point_cloud
using Mill
using Mill: nobs
using Statistics
using DataFrames
using PrettyTables

df = DataFrame[]

for d in mvtec_datasets
    @show d
    normal, anomaly = load_mvtec(d)
    ds = split(d, '_')[1]
    
    nn = nobs(normal)
    na = nobs(anomaly)

    ln = map(i -> size(normal[i].data.data, 2), 1:nn)
    la = map(i -> size(anomaly[i].data.data, 2), 1:na)

    # sn = minimum(ln), round(mean(ln), digits=1), maximum(ln)
    # sa = minimum(la), round(mean(la), digits=1), maximum(la)

    sn = round(Int, mean(ln)), round(Int, median(ln))
    sa = round(Int, mean(la)), round(Int, median(la))

    di = DataFrame(
        :dataset => ds,
        :bags => nn+na,
        :dim => 128,
        :normal_bags => nn,
        ([:mean_normal, :med_normal] .=> sn)...,
        :anomalous_bags => na,
        ([:mean_anomaly, :med_anomaly] .=> sa)...
        # ([:min_normal, :mean_normal, :max_normal] .=> sn)...,
        # ([:min_anomaly, :mean_anomaly, :max_anomaly] .=> sa)...
    )

    push!(df, di)
end

df = vcat(df...)

pretty_table(df, backend=Val(:latex), nosubheader=true, tf=tf_latex_booktabs)

### MIL ###

df = DataFrame[]

for d in mill_datasets
    @show d
    normal, anomaly = load_mill_data(d)
    ds = split(d, '_')[1]
    
    nn = nobs(normal)
    na = nobs(anomaly)

    ln = map(i -> size(normal[i].data.data, 2), 1:nn)
    la = map(i -> size(anomaly[i].data.data, 2), 1:na)

    # sn = minimum(ln), round(mean(ln), digits=1), maximum(ln)
    # sa = minimum(la), round(mean(la), digits=1), maximum(la)

    sn = round(Int, mean(ln)), round(Int, median(ln))
    sa = round(Int, mean(la)), round(Int, median(la))

    di = DataFrame(
        "dataset" => ds,
        "bags" => nn+na,
        "dim" => size(normal.data.data, 1),
        "normal bags" => nn,
        (["mean normal", "med normal"] .=> sn)...,
        "anomalous bags" => na,
        (["mean anomaly", "med anomaly"] .=> sa)...
    )

    push!(df, di)
end

df = vcat(df...)

pretty_table(df, backend=Val(:latex), nosubheader=true, tf=tf_latex_booktabs)


### MNIST ###

df = DataFrame[]

for d in 1:10
    @show d
    anomaly, normal, _, _ = load_mnist_point_cloud(anomaly_class_ind = d)
    ds = d - 1
    
    nn = nobs(normal)
    na = nobs(anomaly)

    ln = map(i -> size(normal[i].data.data, 2), 1:nn)
    la = map(i -> size(anomaly[i].data.data, 2), 1:na)

    # sn = minimum(ln), round(mean(ln), digits=1), maximum(ln)
    # sa = minimum(la), round(mean(la), digits=1), maximum(la)

    sn = round(Int, mean(ln)), round(Int, median(ln))
    sa = round(Int, mean(la)), round(Int, median(la))

    di = DataFrame(
        "dataset" => ds,
        "bags" => nn+na,
        "dim" => 3,
        "normal bags" => nn,
        (["mean normal", "med normal"] .=> sn)...,
        "anomalous bags" => na,
        (["mean anomaly", "med anomaly"] .=> sa)...
    )

    push!(df, di)
end

df = vcat(df...)

pretty_table(df, backend=Val(:latex), nosubheader=true, tf=tf_latex_booktabs)

lhco = GroupAD.load_lhco()
normal, anomaly = lhco

nn = nobs(normal)
na = nobs(anomaly)

ln = map(i -> size(normal[i].data.data, 2), 1:nn)
la = map(i -> size(anomaly[i].data.data, 2), 1:na)

ds = "R&D"

sn = round(Int, mean(ln)), round(Int, median(ln))
sa = round(Int, mean(la)), round(Int, median(la))

di = DataFrame(
    "dataset" => ds,
    "bags" => nn+na,
    "dim" => 3,
    "normal bags" => nn,
    (["mean normal", "med normal"] .=> sn)...,
    "anomalous bags" => na,
    (["mean anomaly", "med anomaly"] .=> sa)...
)