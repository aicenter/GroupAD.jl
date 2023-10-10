using GroupAD
using DataFrames, DrWatson, BSON
using ValueHistories
using DistributionsAD, ConditionalDists, Flux
using Statistics

model = ARGS[1]
collection = ARGS[2]
dataset = ARGS[3]

@time df = collect_results("data/experiments/contamination-0.0/LHCO/$model/", subfolders=true, white_list = [:fit_t, :tst_eval_t], rexclude=[r"model_"])

safesave(
    datadir("times/$(model)_$dataset.bson"),
    Dict(
        :model => model,
        :dataset => dataset,
        :fit_t => df.fit_t,
        :tst_eval_t => df.tst_eval_t,
    )
)

for model in ["vae_basic", "PoolModel", "statistician", "vae_instance"]
    @time df = collect_results("data/experiments/contamination-0.0/LHCO/$model/", subfolders=true, white_list = [:fit_t, :tst_eval_t], rexclude=[r"model_"])

    safesave(
        datadir("times/$(model)_$dataset.bson"),
        Dict(
            :model => model,
            :dataset => dataset,
            :fit_t => df.fit_t,
            :tst_eval_t => df.tst_eval_t,
        )
    )
end

dataset = "mv_tec"
Threads.@threads for model in ["knn_basic", "vae_basic", "PoolModel", "statistician", "vae_instance"]
    @time df = collect_results("data/experiments/contamination-0.0/mv_tec/$model/", subfolders=true, white_list = [:fit_t, :tst_eval_t], rexclude=[r"model_"])

    safesave(
        datadir("times/$(model)_$dataset.bson"),
        Dict(
            :model => model,
            :dataset => dataset,
            :fit_t => df.fit_t,
            :tst_eval_t => df.tst_eval_t,
        )
    )
end

for model in ["knn_basic", "vae_basic", "PoolModel", "statistician", "vae_instance"]
    Threads.@threads for dataset in ["toothbrush", "capsule", "hazelnut", "pill", "screw"]
        @time df = collect_results("data/experiments/contamination-0.0/mv_tec/$model/$(dataset)_together", subfolders=true, white_list = [:fit_t, :tst_eval_t], rexclude=[r"model_"])

        safesave(
            datadir("times/$(model)_$dataset.bson"),
            Dict(
                :model => model,
                :dataset => dataset,
                :fit_t => df.fit_t,
                :tst_eval_t => df.tst_eval_t,
            )
        )
    end
end

for model in ["vae_basic", "knn_basic", "PoolModel", "statistician", "vae_instance"]
    Threads.@threads for dataset in mill_datasets
        df = collect_results("data/experiments/contamination-0.0_old_data/$model/$dataset/", subfolders=true, white_list = [:fit_t, :tst_eval_t], rexclude=[r"model_"])
        safesave(
            datadir("times/$(model)_$dataset.bson"),
            Dict(
                :model => model,
                :dataset => dataset,
                :fit_t => df.fit_t,
                :tst_eval_t => df.tst_eval_t,
            )
        )
    end
end

Threads.@threads for model in ["vae_basic", "knn_basic", "PoolModel", "statistician", "vae_instance"]
    dataset = "MNIST"
    df = collect_results("data/experiments/contamination-0.0_old_data/$model/$dataset/leave-one-in/", subfolders=true, white_list = [:fit_t, :tst_eval_t], rexclude=[r"model_"])
    safesave(
        datadir("times/$(model)_$dataset.bson"),
        Dict(
            :model => model,
            :dataset => dataset,
            :fit_t => df.fit_t,
            :tst_eval_t => df.tst_eval_t,
        )
    ) 
end



for model in ["knn_basic", "vae_basic", "statistician", "vae_instance"]
    for dataset in [
        "bathtub",
        "desk",
        "dresser",
        "night_stand",
        "sofa",
        "table",
        "bed",
        "chair",
        "monitor",
        "toilet"
    ]
        @time df = collect_results("data/experiments/contamination-0.0/modelnet/$model/$dataset", subfolders=true, white_list = [:fit_t, :tst_eval_t], rexclude=[r"model_"])

        safesave(
            datadir("times/$(model)_$dataset.bson"),
            Dict(
                :model => model,
                :dataset => "MN-$dataset",
                :fit_t => df.fit_t,
                :tst_eval_t => df.tst_eval_t,
            )
        )
    end
end

[
    "bathtub",
    "desk",
    "dresser",
    "night_stand",
    "sofa",
    "table",
    "bed",
    "chair",
    "monitor",
    "toilet"
]


using ProgressMeter
model = "knn_basic"
dataset = "MNIST"

fit_t = []
tst_eval_t = []

@showprogress for c in 1:10
    for s in 1:10
        files = readdir("data/experiments/contamination-0.0_old_data/$model/$dataset/leave-one-in/class_index=$c/seed=$s", join=true)
        for f in files
            if occursin("model_", f)
                continue
            end
            d = BSON.load(f)
            push!(fit_t, d[:fit_t])
            push!(tst_eval_t, d[:tst_eval_t])
        end
    end
end

safesave(
    datadir("times/$(model)_$dataset.bson"),
    Dict(
        :model => model,
        :dataset => dataset,
        :fit_t => fit_t,
        :tst_eval_t => tst_eval_t,
    )
)