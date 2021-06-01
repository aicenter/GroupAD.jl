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

# data load and split -------------------------------------------------------------------------------------
### load mill data
problem = "WinterWren"
normal, anomalous = load_mill_data(problem,normalize=true);
# train/val/test split
train, val, test = train_val_test_split(normal,anomalous,seed=1);
# unpack bags and labels to vector form
train_data, train_labels = unpack_mill(train);
val_data, val_labels = unpack_mill(val);
test_data, test_labels = unpack_mill(test);

# fit cardinality distribution -----------------------------------------------------------------------------
# cardinality plot for validation and test data
poisson, lognormal, kernel, p_val = cardinality_hist(train_data, val_data, val_labels, label="validation");
p_val
_, _, _, p_test = cardinality_hist(train_data, test_data, test_labels, label="test");
p_test

# definice modelu -----------------------------------------------------------------------------------
bag = train_data[1]
xdim = size(bag,1)
hdim = 64
zdim = 8
act = "swish"
batchsize = 64

vae = vae_constructor(idim=xdim,hdim=hdim,zdim=zdim,activation=act,nlayers=3,var="diag")

# vezmeme parametry a definujeme opt
ps = Flux.params(vae)
opt = ADAM(0.001)

# loss function definition
loss(x) = -elbo(vae,x)
loss(bag) # compile loss

data = hcat(train_data...)
batch_data = Flux.Data.DataLoader(data,batchsize=batchsize)
batch_data = train_data

# training --------------------------------------------------------------------------------------
val_normal = val_data[val_labels .== 0]
model = deepcopy(vae)
best_val_loss = Inf
epoch = 1
batching = false
iter = 1000

# loop
using ProgressMeter
@showprogress "Training..." 1 for i in 1:iter
    if batching
        batch = RandomBagBatches(train_data,batchsize=256)
        b = Flux.Data.DataLoader(hcat(batch...),batchsize=batchsize)
        Flux.train!(loss,ps,b,opt)
    else
        Flux.train!(loss,ps,batch_data,opt)
    end
    val_loss = mean(loss.(val_normal))
    if val_loss < best_val_loss
        @info "Epoch $epoch: $val_loss"
        best_val_loss = val_loss
        model = deepcopy(vae)
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
plot_roc_scores(test_labels,scores)

plot(p1,p2,layout=(1,2),title=["sampled" "mean"],size=(700,400))