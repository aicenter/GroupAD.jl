using StatsPlots

mill_names = [
    "BrownCreeper", "CorelAfrican", "CorelBeach", "Elephant", "Fox", "Musk1", "Musk2",
    "Mut1", "Mut2", "News1", "News2", "News3", "Protein",
    "Tiger", "UCSB-BC", "Web1", "Web2", "Web3", "Web4", "WinterWren"
]

"""
    groupedbar_matrix(df::DataFrame; group::Symbol, cols::Symbol, value::Symbol, groupnamefull=true)

Create groupnames, matrix and labels for given dataframe. Additionally serves as input to groupedbar plot
for MIL and MNIST results.

Kwargs
- `group`: controls the individual groups of barplots, e. g. dataset for MIL or class for MNIST,
- `col`: controls the individual bars, e. g. score type or aggregation function,
- `value`: controls the value of bars, usually AUC.
"""
function groupedbar_matrix(df::DataFrame; group::Symbol, cols::Symbol, value::Symbol, groupnamefull=false)
    df = sort(df, [group, cols])
    gdf = groupby(df, group)
    gdf_keys = keys(gdf)
    gdf_name = String(group)
    gdf_values = map(k -> values(k)[1], gdf_keys)
    if groupnamefull
        groupnames = map((name, value) -> "$name = $value", repeat([gdf_name], length(gdf_values)), gdf_values)
    else
        groupnames = map(value -> "$value", gdf_values)
    end
    colnames = unique(map(d -> d[:, cols], gdf))
    if length(colnames) != 1
        error("Grouping failed, DataFrame not sorted properly, colnames do not match!")
    end
    M = hcat(map(x -> x[:, value], gdf)...)'

    groupnames, M, String.(hcat(colnames[1]...))
end

"""
    mill_barplots(df, name [, labels]; legend_title="Model", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)

Plots a groupedbar plot for MIL datasets given results dataframe `df`, file savename `name` and
optionally new labels `labels`. Resulting figure is saved in `plotsdir("MIL", "$savename.pdf")`.

Use kwargs `group`, `cols` and `value` which are passed to `groupedbar_matrix` function. For more
information about the kwargs, look at the documentation for `groupedbar_matrix` function.

Variable `ind` serves as a means for different ordering of columns and therefore new labels.
"""
function mill_barplots(df, name; legend_title="Model", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)
    _, M, labels = groupedbar_matrix(df; kwargs...)
    groupnames = mill_names

    gb1 = groupedbar(
        groupnames[1:4], M[1:4, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:outerright,
        ylims=(0,1), bar_width=w1
    )
    #savefig(plotsdir("MIL", "test1.pdf"))
    gb2 = groupedbar(
        groupnames[5:12], M[5:12, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2
    )
    #savefig(plotsdir("MIL", "test2.pdf"))
    gb3 = groupedbar(
        groupnames[13:20], M[13:20, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2
    )
    #savefig(plotsdir("MIL", "test3.pdf"))

    p = plot(gb1,gb2,gb3,layout = (3,1), guidefontsize=5, tickfontsize=5, legendfontsize=5, legendtitlefontsize=5, size=(400,570))
    wsave(plotsdir("MIL", "$name.pdf"),p)
end
function mill_barplots(df, name, labels; legend_title="Model", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)
    _, M, _ = groupedbar_matrix(df; kwargs...)
    groupnames = mill_names
    if size(labels, 1) != 1
        labels = hcat(labels...)
    end

    gb1 = groupedbar(
        groupnames[1:4], M[1:4, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:outerright,
        ylims=(0,1), bar_width=w1
    )
    #savefig(plotsdir("MIL", "test1.pdf"))
    gb2 = groupedbar(
        groupnames[5:12], M[5:12, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2
    )
    #savefig(plotsdir("MIL", "test2.pdf"))
    gb3 = groupedbar(
        groupnames[13:20], M[13:20, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2
    )
    #savefig(plotsdir("MIL", "test3.pdf"))

    p = plot(gb1,gb2,gb3,layout = (3,1), guidefontsize=5, tickfontsize=5, legendfontsize=5, legendtitlefontsize=5, size=(400,570))
    wsave(plotsdir("MIL", "$name.pdf"),p)
end

"""
    mnist_barplots(df, name [, labels]; legend_title="Score", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)

Creates barplots for MNIST in two rows. Add the kwargs `group`, `cols` and `value` which are passed
to `groupedbar_matrix` function. Make sure that the input DataFrame is sorted by `group`
and `cols`.

The function automatically saves the output to folder `plots/MNIST/pdf`.
"""
function mnist_barplots(df, name; legend_title="Score", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)
    _, M, labels = groupedbar_matrix(df; kwargs...)
    groupnames = map(i -> "$i", 0:9)

    gb1 = groupedbar(
        groupnames[1:4], M[1:4, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:outerright,
        ylims=(0,1), bar_width=w1
    )
    gb2 = groupedbar(
        groupnames[5:10], M[5:10, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2,xlabel="digit"
    )

    p = plot(gb1,gb2,layout = (2,1), guidefontsize=5, tickfontsize=5, legendfontsize=5, legendtitlefontsize=5, size=(400,400))
    savefig(plotsdir("MNIST", "pdf", "$name.pdf"))
end
function mnist_barplots(df, name, labels; legend_title="Score", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)
    _, M, _ = groupedbar_matrix(df; kwargs...)
    groupnames = map(i -> "$i", 0:9)
    if size(labels, 1) != 1
        labels = hcat(labels...)
    end

    gb1 = groupedbar(
        groupnames[1:4], M[1:4, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:outerright,
        ylims=(0,1), bar_width=w1
    )
    gb2 = groupedbar(
        groupnames[5:10], M[5:10, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2,xlabel="digit"
    )

    p = plot(gb1,gb2,layout = (2,1), guidefontsize=5, tickfontsize=5, legendfontsize=5, legendtitlefontsize=5, size=(400,400))
    savefig(plotsdir("MNIST", "pdf", "$name.pdf"))
end