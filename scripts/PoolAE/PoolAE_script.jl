using DrWatson
@quickactivate
include(srcdir("models", "utils.jl"))
include(srcdir("models", "PoolAE.jl"))

using Plots
using StatsPlots
ENV["GKSwstype"] = "100"

data1 = [randn(2,rand(Poisson(20))) .+ [2.1, -1.4] for _ in 1:100]
data2 = [randn(2,rand(Poisson(20))) .+ [-2.1, 1.4] for _ in 1:100]
data3 = [randn(2,rand(Poisson(20))) for _ in 1:100]
data4 = [randn(2,rand(Poisson(50))) .+ [2.1, -1.4] for _ in 1:100]

train_data = vcat(data1,data2)
val_data = vcat(
    [randn(2,rand(Poisson(20))) .+ [2.1, -1.4] for _ in 1:100],
    [randn(2,rand(Poisson(20))) .+ [-2.1, 1.4] for _ in 1:100]
)

model = pm_constructor(;idim=2, hdim=8, zdim=2, poolf=mean_max)
opt = ADAM()
ps = Flux.params(model)
loss(x) = pm_variational_loss(model, x)

for i in 1:100
    Flux.train!(loss, ps, train_data, opt)
    @info i mean(loss.(val_data))
end

scatter = Plots.scatter
scatter! = Plots.scatter!

X = hcat(val_data...)
Y = hcat([reconstruct(model, x) for x in val_data]...)

scatter(X[1,:],X[2,:], markersize=2, markerstrokewidth=0)
scatter!(Y[1,:],Y[2,:], markersize=2, markerstrokewidth=0)
savefig("val_data.png")

E = hcat([encoding(model, x) for x in val_data]...)
scatter(E[1,:],E[2,:],zcolor=vcat(zeros(Int, 100),ones(Int, 100)))
savefig("enc.png")

E_an1 = hcat([encoding(model, x) for x in data3]...)
E_an2 = hcat([encoding(model, x) for x in data4]...)
scatter(E[1,:],E[2,:],label="normal")
scatter!(E_an1[1,:],E_an1[2,:],label="anomalous 1")
scatter!(E_an2[1,:],E_an2[2,:],label="anomalous 2")
savefig("enc_anomaly.png")

# different pooling fuction (with cardinality)
model = pm_constructor(;idim=2, hdim=8, zdim=2, poolf=mean_max_card)
opt = ADAM()
ps = Flux.params(model)
loss(x) = pm_variational_loss(model, x)

for i in 1:100
    Flux.train!(loss, ps, train_data, opt)
    @info "$i: $(mean(loss.(val_data)))"
end

X = hcat(val_data...)
Y = hcat([reconstruct(model, x) for x in val_data]...)

scatter(X[1,:],X[2,:], markersize=2, markerstrokewidth=0)
scatter!(Y[1,:],Y[2,:], markersize=2, markerstrokewidth=0)
savefig("val_data_card.png")

E = hcat([encoding(model, x) for x in val_data]...)
scatter(E[1,:],E[2,:],zcolor=vcat(zeros(Int, 100),ones(Int, 100)))
savefig("enc_card.png")

E_an1 = hcat([encoding(model, x) for x in data3]...)
E_an2 = hcat([encoding(model, x) for x in data4]...)
scatter(E[1,:],E[2,:];label="normal", legend=:bottomright)
scatter!(E_an1[1,:],E_an1[2,:],label="anomalous 1")
scatter!(E_an2[1,:],E_an2[2,:],label="anomalous 2")
savefig("enc_anomaly_card.png")

E_all = hcat(E, E_an1, E_an2)
card = vcat(
    map(x -> size(x, 2), val_data),
    map(x -> size(x ,2), data3),
    map(x -> size(x ,2), data4)
)
scatter(E_all[1,:], E_all[2,:], zcolor=card, color=:jet)
savefig("enc_card.png")


model = pm_constructor(;idim=2, hdim=8, zdim=2, poolf=mean_max)
opt = ADAM()
ps = Flux.params(model)
loss(x) = pm_variational_loss(model, x; β=10)

for i in 1:200
    Flux.train!(loss, ps, train_data, opt)
    @info i mean(loss.(val_data))
end

X = hcat(val_data...)
Y = hcat([reconstruct(model, x) for x in val_data]...)

scatter(X[1,:],X[2,:], markersize=2, markerstrokewidth=0)
scatter!(Y[1,:],Y[2,:], markersize=2, markerstrokewidth=0)
savefig("val_data_card_β=10.png")

E = hcat([encoding(model, x) for x in val_data]...)
scatter(E[1,:],E[2,:],zcolor=vcat(zeros(Int, 100),ones(Int, 100)))
savefig("enc_card_β=10.png")

E_an1 = hcat([encoding(model, x) for x in data3]...)
E_an2 = hcat([encoding(model, x) for x in data4]...)
scatter(E[1,:],E[2,:];label="normal", legend=:bottomright)
scatter!(E_an1[1,:],E_an1[2,:],label="anomalous 1")
scatter!(E_an2[1,:],E_an2[2,:],label="anomalous 2")
savefig("enc_anomaly_card_β=10.png")

E_all = hcat(E, E_an1, E_an2)
card = vcat(
    map(x -> size(x, 2), val_data),
    map(x -> size(x ,2), data3),
    map(x -> size(x ,2), data4)
)
scatter(E_all[1,:], E_all[2,:], zcolor=card, color=:jet)
savefig("enc_card_β=10.png")