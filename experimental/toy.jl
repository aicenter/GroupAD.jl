using LinearAlgebra
using Distributions
using StatsBase, Random

using Plots
ENV["GKSwstype"] = "100"


"""
    smm_gauss_toy(Nn, Na, [type = 1])

Generates toy problem of 2D gaussians.
If type is specified, creates anomaly dataset with different
cardinalities according to type = [1,2,3].
"""
function smm_gauss_toy(Nn,Na)
    Σ = Symmetric([0.01 0.008; 0.008 0.01])
    normal = [rand(MvNormal(rand(2),Σ),rand(Poisson(100))) for i in 1:Nn]

    Σan = Symmetric([0.01 -0.008; -0.008 0.01])
    anomalous1 = [rand(MvNormal(rand(2),Σan),rand(Poisson(100))) for i in 1:Na÷2]
    anomalous2 = [rand(MvNormal(rand(2) .+ sample([-2,2],2) .* [0.8,0.6],Σ),rand(Poisson(100))) for i in 1:Na÷2]
    return normal, shuffle(vcat(anomalous1,anomalous2))
end
function smm_gauss_toy(Nn, Na, type)
    # normal data is same for all
    Σ = Symmetric([0.01 0.008; 0.008 0.01])
    normal = [rand(MvNormal(rand(2),Σ),rand(Poisson(50))) for i in 1:Nn]

    # anomalous data has different cardinalities depending on type
    # there are always point anomalies with same cardinality but different mean values
    Σan = Symmetric([0.01 -0.008; -0.008 0.01])
    anomalous1 = [rand(MvNormal(rand(2) .+ sample([-1,2],2) .* [0.8,0.6],Σ),rand(Poisson(50))) for i in 1:Na÷2]
    if type==1
        anomalous2 = [rand(MvNormal(rand(2),Σan),rand(Poisson(50))) for i in 1:Na÷2]
    elseif type==2
        Naa = Na÷2
        mod(Naa,3) != 0 ? error("`Na` needs to be divisible by 6!") : nothing
        anomalous2 = vcat(
            [rand(MvNormal(rand(2),Σ),rand(Poisson(10))) for i in 1:Naa÷3-2],
            [rand(MvNormal(rand(2),Σan),rand(Poisson(50))) for i in 1:Naa÷3-2],
            [rand(MvNormal(rand(2),Σ),rand(Poisson(90))) for i in 1:Naa÷3-2]
        )
    else
        anomalous2 = [rand(MvNormal(rand(2),Σ),rand(Poisson(10))) for i in 1:Na÷2]
    end
    anomalous = vcat(anomalous1,anomalous2)
    return normal, anomalous
end

"""
Example:

normal, anomalous = smm_gauss_toy(100,40)
normal, anomalous = smm_gauss_toy(150,96,2)
X = hcat(normal...)
Y = hcat(anomalous...)

scatter2(X,color=:green,markersize=1,markerstrokewidth=0,label="normal")
scatter2!(Y,color=:red,markersize=1,markerstrokewidth=0,label="anomalous",legend=:outerright)
"""