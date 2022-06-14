using DrWatson
@quickactivate

using GroupAD
using GroupAD: load_data
using GroupAD.Models: unpack_mill
using GroupAD.Models: Chamfer, MMD
using Distances

data = load_data("toy", 300, 300; scenario=2, seed=2053)
data = load_data("BrownCreeper", seed=2053)

train, val, test = data
Xtrain, ytrain = unpack_mill(train)
Xval, yval = unpack_mill(val)
Xtest, ytest = unpack_mill(test)

###############
### Chamfer ###
###############

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
scatterplot(dec[1,:], marker=".", color=yval .+ 2 .|> Int)

# test data
M1test = pairwise(Chamfer(), Xtrain, Xtest)
kernel_test = exp.(.- h .* M1test)
kernel_test = exp.(.- h .* M1test .^ 2)
pred, dec = svmpredict(model, kernel_test)

binary_eval_report(ytest, .- dec[1,:])
scatterplot(dec[1,:], marker=".", color=ytest .+ 2 .|> Int)

#######################################################
### Adding cardinality to the model as a new kernel ###
#######################################################

using GroupAD.Models: PEuclidean

# calculate cardinalities
ctrain = length.(Xtrain)
cval = length.(Xval)
ctest = length.(Xtest)

M1 = pairwise(Chamfer(), Xtrain)
h = 1/median(pairwise(PEuclidean(), Xtrain))
kernel_train = exp.(.- h .* M1 .^ 2)

c = 1/median(pairwise(TotalVariation(), ctrain))
kernel_ctrain = exp.(.- c .* pairwise(TotalVariation(), ctrain))
kt = kernel_train .* kernel_ctrain
model_wo_c = svmtrain(kernel_train, kernel=Kernel.Precomputed; svmtype=OneClassSVM, nu=0.9)
model_w_c = svmtrain(kt, kernel=Kernel.Precomputed; svmtype=OneClassSVM, nu=0.9)


M2 = pairwise(Chamfer(), Xtrain, Xval)
kernel_val = exp.(.- h .* M2)
kernel_cval = exp.(.- c .* pairwise(TotalVariation(), ctrain, cval))
kv = kernel_val .* kernel_cval
pred, dec = svmpredict(model_wo_c, kernel_val)
pred_c, dec_c = svmpredict(model_w_c, kv)

m1 = .- dec[1,:]
m2 = .- dec_c[1,:]

scatterplot(m1[:], marker=".", color=yval .+ 2 .|> Int)
scatterplot(m2[:], marker=".", color=yval .+ 2 .|> Int)
binary_eval_report(yval, m1[:])
binary_eval_report(yval, m2[:])