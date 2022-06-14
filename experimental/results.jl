using DrWatson
@quickactivate

include(projectdir("experimental", "mill_evaluation.jl"))

#########################################################
###                        SMM                        ###
#########################################################

# simple SMM
results = mill_model_results("SMM")
pretty_table(results, nosubheader=true, tf = tf_unicode, crop=:none)

# SMM with cardinality kernel (3x)
results_c1 = mill_model_results("SMMC")
pretty_table(results_c1, nosubheader=true, tf = tf_unicode, crop=:none)

results_c2 = mill_model_results("_SMMC")
pretty_table(results_c2, nosubheader=true, tf = tf_unicode, crop=:none)

results_c3 = mill_model_results("SMMC2")
pretty_table(results_c3, nosubheader=true, tf = tf_unicode, crop=:none)

df = DataFrame(
    hcat(
        results.dataset,
        results.test_AUC,
        results_c1.test_AUC,
        results_c2.test_AUC,
        results_c3.test_AUC
    ),
    [:dataset, :SMM, Symbol("SMM-C 1"), Symbol("SMM-C 2"), Symbol("SMM-C 3")]
)
pretty_table(df, nosubheader=true, tf = tf_markdown, crop=:none)

#########################################################
###                        VAE                        ###
#########################################################

# VAE aggregation
results_vae_basic = mill_model_results("vae_basic")
pretty_table(results_vae_basic, nosubheader=true, tf = tf_unicode, crop=:none)

# VAE instances
filter_fun(df) = filter(:parameters => x -> !in("reconstructed_input", values(x)), df)
results_vae_instance_filt = mill_model_results("vae_instance"; filter_fun = filter_fun)
results_vae_instance = mill_model_results("vae_instance")
pretty_table(results_vae_instance, nosubheader=true, tf = tf_unicode, crop=:none)
pretty_table(results_vae_instance_filt, nosubheader=true, tf = tf_unicode, crop=:none)

df = DataFrame(hcat(results.dataset, results.test_AUC, results_c.test_AUC), [:dataset, :SMM, :SMMC])
df = DataFrame(hcat(results.dataset, results.test_AUC, results_c.test_AUC, results_vae_basic.test_AUC), [:dataset, :SMM, :SMMC, :aggVAE])
pretty_table(df, nosubheader=true, tf = tf_unicode, crop=:none)



### Find the difference between sum and sum+logU
vae_instance_results = mill_collect("vae_instance")
d = vae_instance_results[1]

cdfs = []
for d in vae_instance_results
    par_df = mapreduce(p -> DataFrame(Dict(keys(p) .=> values(p))), vcat, d.parameters)
    df = hcat(par_df, d[:, Not(:parameters)])
    dfs = filter(:type => x -> any(x .== ["sum", "logU"]), df)

    gdf = groupby(dfs, :type)
    cdf = combine(gdf, :val_AUC => maximum)
    @show d.dataset[1]
    @show cdf
    push!(cdfs, hcat(DataFrame(:dataset => [d.dataset[1], ""]), cdf))
end

pretty_table(vcat(cdfs...), nosubheader=true, crop=:none, hlines=1:2:42)
pretty_table(vcat(cdfs...), nosubheader=true, crop=:none, hlines=1:2:42, tf = tf_markdown)