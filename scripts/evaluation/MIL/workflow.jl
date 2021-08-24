"""
    find_best_model(folder::String [, groupkey]; metric=:val_AUC)

Recursively goes through given folder and finds the best model based on
chosen metric, default is validation AUC.

If `groupkey` is present, returns the best model for each category of groupkey.
Group key can be both a symbol or an array of symbols.
"""
function find_best_model(folder::String; metric=:val_AUC)
    #folder = datadir("experiments", "contamination-0.0", modelname, dataset)
    data = GroupAD.Evaluation.results_dataframe(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
    params = point[:parameters]

    g = groupby(data, [keys(params)...])
    un = unique(map(x -> size(x), g))
    if length(un) != 1
        idx = findall(x -> size(x,1) > 5, g)
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing $(length(g) - length(idx)) groups out of $(length(g)) with less than 6 seeds."
        g = g[idx]
    end

    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]
    cdf = combine(g, map(x -> x => mean, metricsnames))
    sort!(cdf, :val_AUC_mean, rev=true)
    best_model = cdf[1,:]
end
function find_best_model(folder, groupkey, metric=:val_AUC)
    #folder = datadir("experiments", "contamination-0.0", modelname, dataset, "scenario=$scenario")
    data = GroupAD.Evaluation.results_dataframe(folder)
    point = load(GroupAD.Evaluation.collect_scores(folder)[1])
    params = point[:parameters]

    g_score = groupby(data, groupkey)
    g = map(x -> groupby(x, [keys(params)...]), g_score)
    un = unique(vcat(map(x -> unique(map(y -> size(y), x)), g)...))

    if length(un) != 1
        idx = findall.(x -> size(x,1) > 5, g)
        @warn "There are groups with different sizes (different number of seeds). Possible duplicate models or missing seeds.
        Removing $(sum(length.(g)) - sum(length.(idx))) groups out of $(sum(length.(g))) with less than 6 seeds."
        g = map(i -> g[i][idx[i]], 1:length(g))
    end

    metricsnames = [:val_AUC, :val_AUPRC, :test_AUC, :test_AUPRC]
    cdf = map(y -> combine(y, map(x -> x => mean, metricsnames)), g)
    cdf_sorted = map(x -> sort(x, :val_AUC_mean, rev=true), cdf)
    best_models = vcat(map(x -> DataFrame(x[1,:]), cdf_sorted)...)
end

"""
    groupedbar_matrix(df::DataFrame; group::Symbol, cols::Symbol, value::Symbol, groupnamefull=true)

Create groupnames, matrix and labels for given dataframe.
"""
function groupedbar_matrix(df::DataFrame; group::Symbol, cols::Array{Symbol,1}, value::Symbol, groupnamefull=false)
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
    gdf = groupby(df, group)
    gdf_keys = keys(gdf)
    gdf_name = String(group)
    gdf_values = map(k -> values(k)[1], gdf_keys)
    if groupnamefull
        groupnames = map((name, value) -> "$name = $value", repeat([gdf_name], length(gdf_values)), gdf_values)
    else
        groupnames = map(value -> "$value", gdf_values)
    end
    colnames = gdf[1][:, cols]
    M = hcat(map(x -> x[:, value], gdf)...)'

    groupnames, M, String.(hcat(colnames...))
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

function mnist_barplots(df, name; legend_title="Score", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)
    _, M, labels = groupedbar_matrix(df; kwargs...)
    groupnames = map(i -> "$i", 0:9)

    gb1 = groupedbar(
        groupnames[1:4], M[1:4, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:outerright,
        ylims=(0,1), bar_width=w1
    )
    #savefig(plotsdir("MIL", "test1.pdf"))
    gb2 = groupedbar(
        groupnames[5:10], M[5:10, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2,xlabel="digit"
    )

    p = plot(gb1,gb2,layout = (2,1), guidefontsize=5, tickfontsize=5, legendfontsize=5, legendtitlefontsize=5, size=(400,400))
    savefig(plotsdir("MNIST", "$name.pdf"))
end
function mnist_barplots(df, name, labels; legend_title="Score", ind=:, w1 = 0.5, w2 = 0.8, kwargs...)
    _, M, _ = groupedbar_matrix(df; kwargs...)
    groupnames = map(i -> "$i", 0:9)

    gb1 = groupedbar(
        groupnames[1:4], M[1:4, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:outerright,
        ylims=(0,1), bar_width=w1
    )
    #savefig(plotsdir("MIL", "test1.pdf"))
    gb2 = groupedbar(
        groupnames[5:10], M[5:10, ind], labels=labels, color_palette=:tab20,
        ylabel="test AUC", legendtitle=legend_title, legend=:none,
        ylims=(0,1), bar_width=w2,xlabel="digit"
    )
    #savefig(plotsdir("MIL", "test3.pdf"))

    p = plot(gb1,gb2,layout = (2,1), guidefontsize=5, tickfontsize=5, legendfontsize=5, legendtitlefontsize=5, size=(400,400))
    savefig(plotsdir("MNIST", "$name.pdf"))
end