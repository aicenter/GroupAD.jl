using PrettyTables

"""
# MIL datasets summary table

Should feature:
- dataset name
- number of features
- number of bags
- number of normal/anomalous samples
- mean/meadian of cardinalities
"""

t = DataFrame[]
for dataset in mill_datasets
    d = GroupAD.load_data(dataset)
    bags = cat(d[1][1], d[2][1], d[3][1])
    no_bags = length(bags)
    no_features = size(bags.data.data,1)

    labels = vcat(d[1][2], d[2][2], d[3][2])
    no_anomalous = sum(labels)
    no_normal = no_bags - no_anomalous

    sizes = map(i -> size(bags[i], 2), 1:length(labels))
    _median = median(sizes)
    _mean = mean(sizes)

    df = DataFrame(
        :dataset => dataset,
        :bags => no_bags,
        :features => no_features,
        :normal => no_normal,
        :anomalous => no_anomalous,
        Symbol("median size") => _median,
        Symbol("mean size") => _mean
    )
    push!(t, df)
end

T = vcat(t...)
t = pretty_table(
    T,
    formatters = ft_printf("%5.1f"),
    backend=:latex, tf=tf_latex_booktabs, nosubheader=true
)

