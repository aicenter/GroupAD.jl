modelname = "vae_basic"
dataset = "MNIST"
method = "leave-one-out"
classes = 1:10

models_full = map(c -> find_best_model_scores(modelname, dataset, method, c), classes)
models_downsampled = map(c -> find_best_model_scores(modelname, "MNIST_downsampled", method, c), classes)

df_full = sort(vcat(models_full...)[:, [:class, :test_AUC_mean]])
df_down = sort(vcat(models_downsampled...)[:, [:class, :test_AUC_mean]])

g_full = groupby(df_full, :class)
g1 = map(x -> rename(x, :test_AUC_mean => Symbol("class=$(x[1,:class])")), g_full)
g11 = hcat(map(x -> x[!, 2], g1)...)

g_down = groupby(df_down, :class)
g2 = map(x -> rename(x, :test_AUC_mean => Symbol("class=$(x[1,:class])")), g_down)
g22 = hcat(map(x -> x[!, 2], g2)...)

G = vcat(g11,g22)
scorenames = ["reconstruction-sampled" "reconstruction-mean" "reconstruction" "reconstruction-sampled + downsample" "reconstruction-mean + downsample" "reconstruction + downsample"]

groupedbar(
    map(i -> "$i", 1:10), G',
    ylabel="AUC", labels=scorenames,
    ylims=(0,1), size=(1850,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerbottom
    )
savefig(plotsdir("barplots", "MNIST_$(modelname)_$(method).png"))

# to put corresponding scores next to each other
G2 = G[[1,4,2,5,3,6],:]
scorenames2 = ["reconstruction-sampled" "reconstruction-sampled + downsample" "reconstruction-mean" "reconstruction-mean + downsample" "reconstruction" "reconstruction + downsample"]

groupedbar(
    map(i -> "$i", 1:10), G2',
    ylabel="AUC", labels=scorenames2,
    ylims=(0,1), size=(1850,700), color_palette=:tab20,
    legendfontsize=12, tickfontsize=12, legend=:outerbottom
    )
savefig(plotsdir("barplots", "MNIST_$(modelname)_$(method)_2.png"))