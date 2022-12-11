using GroupAD
using GroupAD: load_data, load_mvtec, load_mill_data
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
        :normal_bags => nn,
        :anomalous_bags => na,
        # ([:min_normal, :mean_normal, :max_normal] .=> sn)...,
        # ([:min_anomaly, :mean_anomaly, :max_anomaly] .=> sa)...
        ([:mean_normal, :med_normal] .=> sn)...,
        ([:mean_anomaly, :med_anomaly] .=> sa)...
    )

    push!(df, di)
end

df = vcat(df...)

pretty_table(df, backend=Val(:latex), nosubheader=true, tf=tf_latex_booktabs)


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
        "normal bags" => nn,
        "anomalous bags" => na,
        # ([:min_normal, :mean_normal, :max_normal] .=> sn)...,
        # ([:min_anomaly, :mean_anomaly, :max_anomaly] .=> sa)...
        (["mean normal", "med normal"] .=> sn)...,
        (["mean anomaly", "med anomaly"] .=> sa)...
    )

    push!(df, di)
end

df = vcat(df...)

pretty_table(df, backend=Val(:latex), nosubheader=true, tf=tf_latex_booktabs)