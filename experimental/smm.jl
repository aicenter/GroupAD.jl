using DrWatson
@quickactivate

using GroupAD
using GroupAD: load_data
using GroupAD.Models: unpack_mill
using GroupAD.Models: Chamfer, MMD
# include("experimental/smm_src.jl")

data = load_data("toy", 300, 300; scenario=2, seed=2053)
data = load_data("BrownCreeper", seed=2053)

normal, anomalous = smm_gauss_toy(40,20)
Xtrain = normal[1:20]
ytrain = zeros(20)
Xtest = vcat(normal[21:40], anomalous)
ytest = vcat(zeros(20), ones(20))

train, val, test = data
Xtrain, ytrain = unpack_mill(train)
Xval, yval = unpack_mill(val)
Xtest, ytest = unpack_mill(test)
x1 = Xtrain[1]
x2 = Xtrain[2]

mmd(GaussianKernel(1.0), x1, x2)
chamfer_distance(x1,x2)

X = hcat(vcat(Xtrain, Xtest, Xval)...)
scatter2(X)
savefig("obr.svg")

###############
### Chamfer ###
###############

# 1/median

# train
M1 = pairwise(Chamfer(), Xtrain)
h = 1/median(M1)
#h = 1.5
kernel_train = exp.(.- h .* M1 .^ 2)
kernel_train = M1

using LIBSVM
model = svmtrain(kernel_train, kernel=Kernel.Precomputed; svmtype=OneClassSVM)

# validation data
M1val = pairwise(Chamfer(), Xtrain, Xval)
kernel_val = exp.(.- h .* M1val)
pred, dec = svmpredict(model, kernel_val)

binary_eval_report(yval, .- dec[1,:])

# test data
M1test = pairwise(Chamfer(), Xtrain, Xtest)
kernel_test = exp.(.- h .* M1test)
kernel_test = exp.(.- h .* M1test .^ 2)
pred, dec = svmpredict(model, kernel_test)
accuracy(ytest, pred)
accuracy(ytest, .!pred)

# kNN on distance matrix if known labels
DM = pairwise(Chamfer(), Xtest, Xval)
foreach(k -> dist_knn(k, DM, yval, ytest), 1:100)

# using created structures
# create very easy data

using Distributions
Xtrain = [randn(2,rand(Poisson(60))) for _ in 1:100]
Xtest = vcat(
    [randn(2,rand(Poisson(60))) for _ in 1:100],
    [randn(2,rand(Poisson(60))) .+ [30,30] for _ in 1:100],
)
ytest = vcat(zeros(100), ones(100))

data = load_data("toy", 300, 300; scenario=2, seed=2053)
data = load_data("BrownCreeper", seed=2053)

train, val, test = data
Xtrain, ytrain = unpack_mill(train)
Xval, yval = unpack_mill(val)
Xtest, ytest = unpack_mill(test)

model = SMM(Chamfer(), Xtrain)
model = SMM(MMD(GaussianKernel(0.1f0)), Xtrain)
pred, dec = predict(model, Xtrain, Xtest)

accuracy(ytest, pred)
binary_eval_report(ytest, .- dec[1,:])

thr = median(dec[1,:])
ynew = dec[1,:] .> thr
accuracy(ytest, .!ynew)


function SMM_experiment(data, metric::String, kernel::String=nothing; min=0, max=10, values=100)
    if metric == "MMD"
        γ = rand(Uniform(min, max), values)
        for gi in γ
            rocs = zeros(length(γ))
            if kernel == "Gaussian"
                SMM_experiment(data, MMD(GaussianKernel(gi)); min=min, max=max, values=values)
            elseif kernel == "IMQ"
                SMM_experiment(data, MMD(IMQKernel(gi)); min=min, max=max, values=values)
            end
        end
    elseif metric == "Chamfer"
        h, vauc, tauc = SMM_experiment(data, Chamfer(); min=min, max=max, values=values)
    end
    return h, vauc, tauc
end

function SMM_experiment(data, distance::Union{MMD, Chamfer}; min=0, max=10, values=100)
    # sample outer kernel width
    h = Float32.(rand(Uniform(min, max), values))

    # unpack data
    train, val, test = data
    Xtrain, ytrain = unpack_mill(train)
    Xval, yval = unpack_mill(val)
    Xtest, ytest = unpack_mill(test)

    # allocate results vectors
    val_rocs = zeros(length(h))
    test_rocs = zeros(length(h))

    for (i, hi) in enumerate(h)
        # fit SMM model
        model = SMM(distance, Xtrain, hi)
        pred_v, dec_v = predict(model, Xtrain, Xval)
        pred_t, dec_t = predict(model, Xtrain, Xtest)

        roc_val = binary_eval_report(yval, .- dec_v[1,:])["au_roccurve"]
        roc_test = binary_eval_report(ytest, .- dec_t[1,:])["au_roccurve"]

        @show roc_val
        @show roc_test

        val_rocs[i] = roc_val
        test_rocs[i] = roc_test
    end

    # either return full results
    # return h, val_rocs, test_rocs

    # or find the best result on validation and only return that combination
    ix = findmax(val_rocs)[2]
    return h[ix], val_rocs[ix], test_rocs[ix]
end

SMM_experiment(data, "Chamfer", "string"; min=0, max=20, values=10)