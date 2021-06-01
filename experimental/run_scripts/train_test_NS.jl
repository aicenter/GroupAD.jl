using Revise
using GroupAD
using DrWatson
using BSON, DataFrames, Random
using Distributions, LinearAlgebra
using Flux, GenerativeModels
using ProgressMeter
using EvalMetrics

include(expdir("utils","cardinality.jl"))
include(expdir("utils","evaluation.jl"))
include(expdir("utils","plotting.jl"))

using GroupAD.Models: RandomBagBatches, unpack_mill, vae_constructor, statistician_constructor

problem = "WinterWren"

# data load and split ---------------------------------------------------
### load mill data
train, val, test = load_data(problem)
# unpack bags and labels to vector form
add_c = false
train_data, train_labels = unpack_mill(train; add_c=add_c);
val_data, val_labels = unpack_mill(val; add_c=add_c);
test_data, test_labels = unpack_mill(test; add_c=add_c);

# cardinality plot for validation and test data
poisson, lognormal, p_val = cardinality_hist(train_data, val_data, val_labels, label="validation");
p_val
_, _, p_test = cardinality_hist(train_data, test_data, test_labels, label="test");
p_test

# model definition --------------------------------------------------------
bag = train_data[1]
xdim = size(bag,1)
hdim = 128
vdim = 32
cdim = 32
zdim = 64
act = "relu"
NS = statistician_constructor(idim=xdim,hdim=hdim,vdim=vdim,cdim=cdim,zdim=zdim,
                              nlayers=3,activation=act,var="diagonal")

# parameters, loss
ps = Flux.params(NS)
opt = ADAM(0.001)
loss(x) = -elbo(NS,x;β1=1.0,β2=1.0)
loss(bag)

# training --------------------------------------------------------------------------------------
val_normal = val_data[val_labels .== 0]
model = deepcopy(NS)
best_val_loss = Inf
epoch = 1
batching = true
iter = 1000

# calculate validation loss only on normal samples in validation data
@showprogress "Training..." 1 for i in 1:iter
    if batching
        batch = RandomBagBatches(train_data, batchsize=64)
        Flux.train!(loss,ps,batch,opt)
    else
        Flux.train!(loss,ps,train_data,opt)
    end
    val_loss = median(loss.(val_normal))
    if val_loss < best_val_loss
        @info "Epoch $epoch: $val_loss"
        best_val_loss = val_loss
        model = deepcopy(NS)
    elseif isnan(val_loss)
        @info "NaN loss. Training stopped."
        break
    end
    epoch += 1
    opt.eta *= 0.999
end

# evaluate results ------------------------------------------------------------------------------
L = 20
# calculate scores for latest model
#scores = score_report(model, test_data, test_labels, reconstruction_score);
scores = score_report(model, test_data, test_labels, reconstruction_score, L);
p1 = plot_roc_scores(test_labels,scores);

scores = score_report(model, test_data, test_labels, mean_reconstruction_score);
p2 = plot_roc_scores(test_labels,scores);

scores = score_report(model, test_data, test_labels, reconstruction_score_instance_mean);
p3 = plot_roc_scores(test_labels,scores);

scores = [[kl_scores(model, bag) for bag in test_data],[kl_scores(model, bag, "z") for bag in test_data]]
p4 = plot_roc_scores(test_labels,scores);

chamfer = [chamfer_distance(bag, reconstruct(model, bag)) for bag in test_data]
rocplot(test_labels, chamfer)

MMD = [mmd(IMQKernel(0.1), bag, reconstruct(model, bag)) for bag in test_data]
rocplot(test_labels, MMD)

plot(p1,p2,p3,p4,layout=(2,2),title=["sampled" "mean" "instance mean" "KL score"],size=(700,600))