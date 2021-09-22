"""
    groupedbar_matrix(df::DataFrame; group::Symbol, cols::Symbol, value::Symbol, groupnamefull=true)

Create groupnames, matrix and labels for given dataframe.
"""
function groupedbar_matrix(df::DataFrame; group::Symbol, cols::Array{Symbol,1}, value::Symbol, groupnamefull=false)
    # this functions is deprecated, cols are not supported for more symbols
    gdf = groupby(df, group)
    gdf_keys = keys(gdf)
    gdf_name = String(group)
    gdf_values = map(k -> values(k)[1], gdf_keys)
    if groupnamefull
        groupnames = map((name, value) -> "$name = $value", repeat([gdf_name], length(gdf_values)), gdf_values)
    else
        groupnames = map(value -> "$value", gdf_values)
    end
    _colnames = gdf[1][:, cols]
    colnames = map((x, y) -> "$x & $y", _colnames[:, 1], _colnames[:, 2])
    M = hcat(map(x -> x[:, value], gdf)...)'

    groupnames, M, String.(hcat(colnames...))
end
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
    mill_results(modelname, mill_datasets [, groupkey]; info = true)

Calculates the results dataframe for full validation dataset for all
MIL datasets and chosen model.

If `groupkey` is included, finds best model for each category of groupkey.
"""
function mill_results(modelname, mill_datasets; info = true)
    res = []
    # full results folder
    folder = datadir("experiments", "contamination-0.0", modelname)

    for d in mill_datasets
        # dataset folder
        dfolder = joinpath(folder, d)
        model = find_best_model(dfolder) |> DataFrame
        #@info d
        insertcols!(model, :dataset => d)
        push!(res, model)
        if info
            @info "Best $modelname model for $d found."
        end
    end

    results = vcat(res...)
end
function mill_results(modelname, mill_datasets, groupkey; info = true)
    res = []
    # full results folder
    folder = datadir("experiments", "contamination-0.0", modelname)

    for d in mill_datasets
        # dataset folder
        dfolder = joinpath(folder, d)
        model = find_best_model(dfolder, groupkey) |> DataFrame
        #@info d
        insertcols!(model, :dataset => d)
        push!(res, model)
        if info
            @info "Best $modelname model for $d found."
        end
    end

    results = vcat(res...)
end


"""
    barplot_mill(modelname, results; sorted = false)

Plots and saves a barplot of all MIL datasets and given model.
"""
function barplot_mill(modelname, results; sorted = false, savef = false)
    if sorted
        res_sort = sort(results, :val_AUC_mean, rev=true)
    else
        res_sort = results
    end
    r = res_sort[:, [:val_AUC_mean, :test_AUC_mean]] |> Array
    p = groupedbar(
        res_sort[:, :dataset],r,xrotation=55,legendtitle=modelname,
        label=["val-AUC" "test-AUC"], ylabel="AUC", legend=:bottomright, ylims=(0,1))
    if savef
        savefig(plotsdir("barplot_$(modelname).png"))
    end
    return p
end

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
    savefig(plotsdir("MIL", "$name.pdf"))
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
    savefig(plotsdir("MIL", "$name.pdf"))
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