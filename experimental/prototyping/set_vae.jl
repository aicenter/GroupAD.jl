# using Flux3D: chamfer_distance

struct sVAE
    static
    pooling
    prior
    encoder
    decoder
end

Flux.@functor sVAE

function sVAE(static_network, poolf::Function, zdim::Int, enc::ConditionalMvNormal, dec::ConditionalMvNormal)
    W = first(Flux.params(enc))
    μ = fill!(similar(W, zdim), 0)
    σ = fill!(similar(W, zdim), 1)
    prior = DistributionsAD.TuringMvNormal(μ, σ)
    sVAE(static_network, poolf, prior, enc, dec)
end

function svae_constructor(;idim, zdim, pdim, hdim, var, activation, nlayers, poolf=mean)
    # if mean-max pooling, output pdim must be half the original
    poolf == mean_max_pool ? sdim = pdim÷2 : sdim = pdim

    stat_net = build_mlp(idim, hdim, sdim, nlayers, activation=activation)

    encoder_map = Chain(
            build_mlp(pdim, hdim, hdim, nlayers-1, activation=activation)...,
            ConditionalDists.SplitLayer(hdim, [zdim, zdim], [identity, safe_softplus])
            )
    encoder = ConditionalMvNormal(encoder_map)
        
    # decoder - we will optimize only a shared scalar variance for all dimensions
    if var=="scalar"
        decoder_map = Chain(
            build_mlp(zdim, hdim, hdim, nlayers-1, activation=activation)...,
            ConditionalDists.SplitLayer(hdim, [idim, 1], [identity, safe_softplus])
        )
    else
        decoder_map = Chain(
            build_mlp(zdim, hdim, hdim, nlayers-1, activation=activation)...,
            ConditionalDists.SplitLayer(hdim, [idim, idim], [identity, safe_softplus])
        )
    end
    decoder = ConditionalMvNormal(decoder_map)

    model = sVAE(stat_net, poolf, zdim, encoder, decoder)
end

model = svae_constructor(poolf=mean_max_pool,idim=idim, hdim=hdim, zdim=zdim, pdim=pdim, var="scalar", activation=activation, nlayers=nlayers)

function Base.show(io::IO, m::sVAE)
    sn = repr(m.static)
    sn = sizeof(sn)>70 ? "($(sn[1:70-3])...)" : sn
    pool = repr(m.pooling)
    p = repr(m.prior)
    p = sizeof(p)>70 ? "($(p[1:70-3])...)" : p
    e = repr(m.encoder)
    e = sizeof(e)>70 ? "($(e[1:70-3])...)" : e
    d = repr(m.decoder)
    d = sizeof(d)>70 ? "($(d[1:70-3])...)" : d
    msg = """$(nameof(typeof(m))):
     static  = $(sn)
     pooling = $(pool)
     prior   = $(p)
     encoder = $(e)
     decoder = $(d)
    """
    print(io, msg)
end

import GenerativeModels: elbo
function elbo(m::sVAE, x; β=1)
    # static network and pooling
    n = size(x, 2)
    ξ = m.static(x)
    p = m.pooling(ξ, dims=2)

    z = hcat([rand(m.encoder, p) for _ in 1:n]...)

    # reconstruction error
    llh = mean(logpdf(m.decoder, x, z))

    # KLD with `condition`ed encoder
    kld = mean(kl_divergence(condition(m.encoder, p), m.prior))

    llh - β*kld
end

function likelihood(m::sVAE, x; poolf=mean)
    n = size(x, 2)
    ξ = m.static(x)
    p = m.pooling(ξ, dims=2)
    z = hcat([rand(m.encoder, p) for _ in 1:n]...)
    llh = -poolf(logpdf(m.decoder, x, z))
end

function kl_score(m::sVAE, x)
    ξ = m.static(x)
    p = m.pooling(ξ, dims=2)
    kld = mean(kl_divergence(condition(m.encoder, p), m.prior))
end

# with another network
function mean_max_pool(x; dims=dims)
    y1 = mean(x, dims = dims)
    y2 = maximum(x, dims = dims)
    return vcat(y1,y2)
end

function encoding(m::sVAE, x)
    ξ = m.static(x)
    p = m.pooling(ξ, dims=2)
end

function latent_encoding(m::sVAE, x)
    n = size(x, 2)
    ξ = m.static(x)
    p = m.pooling(ξ, dims=2)
    z = mean(m.encoder, p)
end


########
# WinterWren data?
problem = "WinterWren"

# data load and split ---------------------------------------------------
### load mill data
train, val, test = load_data(problem)
# unpack bags and labels to vector form
train_data, train_labels = unpack_mill(train);
val_data, val_labels = unpack_mill(val);
test_data, test_labels = unpack_mill(test);

# model definition
bag = train_data[1]
xdim = size(bag,1)
hdim = 64
pdim = 4
zdim = 4
act = "swish"
setvae = svae_constructor(poolf=mean_max_pool,idim=xdim, hdim=hdim, zdim=zdim, pdim=pdim, var="diag", activation=activation, nlayers=nlayers)

ps = Flux.params(setvae)
loss(x) = -elbo(setvae, x)
loss(bag)
opt = ADAM()

# training --------------------------------------------------------------------------------------
val_normal = val_data[val_labels .== 0]
model = deepcopy(setvae)
best_val_loss = Inf
epoch = 1
batching = false
iter = 100

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
        model = deepcopy(setvae)
    elseif isnan(val_loss)
        @info "NaN loss. Training stopped."
        break
    end
    epoch += 1
    opt.eta *= 0.999
end


scores = [Float64[],Float64[],Float64[],Float64[]]
scores[1] = loss.(test_data)
scores[2] = [likelihood(model, x) for x in test_data]
scores[3] = [likelihood(model, x; poolf=sum) for x in test_data]
scores[4] = [kl_score(model, x) for x in test_data]

plot_roc_scores(test_labels, scores)
binary_eval_report(val_labels, scores)


# UMAP for pooling ------------------------------------------

function pool_umap(data, labels; poolf=mean)
    X = hcat(data[labels .== 0])
    Y = hcat(data[labels .== 1])

    Xm = hcat(poolf.(X, dims=2)...)
    Ym = hcat(poolf.(Y, dims=2)...)

    emb = umap(hcat(Xm,Ym),2)
    scatter2(emb, zcolor=labels)
end

problem = "CorelAfrican"
train, val, test = load_data(problem,seed=1)
train_data, train_labels = unpack_mill(train);
val_data, val_labels = unpack_mill(val);
test_data, test_labels = unpack_mill(test);
pool_umap(val_data, val_labels, poolf=mean)

pool_umap(val_data, val_labels, poolf=maximum)
pool_umap(val_data, val_labels, poolf=mean_max_pool)

# model encoding ----------------------------------------------------------
E = hcat([encoding(model, x) for x in val_data]...)
scatter2(E[[3,4],:],color=Int.(val_labels),label="")

Etv = hcat([encoding(model, x) for x in vcat(train_data, val_data)]...)
tv_labels = Int.(vcat(train_labels .+ 2, val_labels))
scatter2(Etv[[1,3],:], zcolor=tv_labels)

using UMAP
emb = umap(E, 2)
scatter2(emb, zcolor=Int.(val_labels), label="")

emb = umap(Etv, 2)
scatter2(emb, zcolor=tv_labels, label="")

