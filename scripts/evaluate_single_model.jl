using DrWatson
@quickactivate
using GroupAD
using GroupAD.Evaluation
using ArgParse
using DataFrames
using PrettyTables

s = ArgParseSettings()
@add_arg_table! s begin
    "modelname"
        arg_type = String
        help = "model name"
        default = "vae_basic"
    "dataset"
        default = "Fox"
        arg_type = String
        help = "dataset"
    "groupkey"
        default = :nothing
        arg_type = Symbol
        help = "group key, e.g. `:aggregation`"
    "class"
        default = 1
        arg_type = Int
        help = "class for MNIST"
    "method"
        default = "leave-one-in"
        arg_type = String
        help = "method: leave-one-in or leave-one-out"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, groupkey, class, method = parsed_args

mill_datasets = [
    "BrownCreeper", "CorelBeach", "CorelAfrican", "Elephant", "Fox", "Musk1", "Musk2",
    "Mutagenesis1", "Mutagenesis2", "Newsgroups1", "Newsgroups2", "Newsgroups3", "Protein",
    "Tiger", "UCSBBreastCancer", "Web1", "Web2", "Web3", "Web4", "WinterWren"
]

"""
    evaluate_single_model(modelname, dataset; groupkey = nothing, class=1, method="leave-one-in")

Given `modelname` and `dataset` (+ `class` and `method` for MNIST dataset), finds the best model
and prints it. If `groupkey` is provided, finds the best models based on `groupkey`.
"""
function evaluate_single_model(modelname, dataset; groupkey = nothing, class=1, method="leave-one-in")
    if groupkey == :nothing
        if dataset in mill_datasets
            df = mill_results(modelname, [dataset])
        elseif dataset == "MNIST"
            folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$(class+1)")
            df = find_best_model(folder) |> DataFrame
        elseif dataset == "toy"
            nothing
        else
            error("Dataset \"$dataset\" for model \"$modelname\" not found.")
        end
    else
        if dataset in mill_datasets
            df = mill_results(modelname, [dataset], groupkey)
        elseif dataset == "MNIST"
            folder = datadir("experiments", "contamination-0.0", modelname, "MNIST", method, "class_index=$(class+1)")
            df = find_best_model(folder, groupkey) |> DataFrame
        elseif dataset == "toy"
            nothing
        else
            error("Dataset \"$dataset\" for model \"$modelname\" not found.")
        end
    end

    println("Full results DataFrame:")
    pdf_full = pretty_table(df)
    if groupkey == :nothing
        println("Small results DataFrame:")
        pdf_small = pretty_table(df[:, [:val_AUC_mean, :val_AUPRC_mean, :test_AUC_mean, :test_AUPRC_mean]])
    else
        println("Small results DataFrame:")
        pdf_small = pretty_table(df[:, [groupkey, :val_AUC_mean, :val_AUPRC_mean, :test_AUC_mean, :test_AUPRC_mean]])
    end
end

evaluate_single_model(modelname, dataset; groupkey = groupkey, class=class, method=method)