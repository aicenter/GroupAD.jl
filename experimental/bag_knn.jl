using DrWatson
@quickactivate

using GroupAD
using GroupAD: load_data
using GroupAD.Models: Chamfer, MMD, unpack_mill
using IPMeasures
using Distances
using Statistics
using EvalMetrics

seed = 2051
data = load_data("Fox", seed=seed)

train, val, test = data
Xtrain, ytrain = unpack_mill(train)
Xval, yval = unpack_mill(val)
Xtest, ytest = unpack_mill(test)

x1, x2 = Xtrain[1], Xtest[end]

# get the distance matrix for the neighbors and sort it by columns
M = pairwise(Chamfer(), Xtest, Xtrain)
Msorted = sort(M, dims=2)

# choose k-neighbors and calculate the distances
k = 5

# kappa
dists = Msorted[:, k]
binary_eval_report(ytest, dists)
# gamma
dists = mean(Msorted[:, 2:k], dims=2)[:]
binary_eval_report(ytest, dists)

# pairwise Euclidean distance
using Distances: UnionMetric
import Distances: result_type

peuclidean(X, Y) = median(pairwise(Euclidean(), X,Y))

struct PEuclidean <: UnionMetric end
(dist::PEuclidean)(x, y) = peuclidean(x, y)
result_type(dist::PEuclidean, x, y) = Float32

M = pairwise(PEuclidean(), Xtest, Xtrain)
sort!(M, dims=2)
# kappa
dists = M[:, k]
binary_eval_report(ytest, dists)
# gamma
dists = mean(M[:, 2:k], dims=2)[:]
binary_eval_report(ytest, dists)

### finding the best kernel width

mg = pairwise(PEuclidean(), Xtrain)
m = 1/median(mg)
γ = rand(Uniform(0.5m, 1.5m))

k = sample(1:3:51)
M = pairwise(MMD(GaussianKernel(γ)), Xtest, Xtrain)
sort!(M, dims=2)
# kappa
dists = M[:, k]
binary_eval_report(ytest, dists)
# gamma
dists = mean(M[:, 2:k], dims=2)[:]
binary_eval_report(ytest, dists)