using Distributions
using ValueHistories
using Flux
using MLDataPattern: RandomBatches
using StatsBase
using Random
using Mill

########################
### Model definition ###
########################

# structure for β parameters
struct β_par
    μ
    Σ
end

function Base.show(io::IO, b::β_par)
    nm = """Dictionary β: 
    μ = $(b.μ)
    Σ = $(b.Σ)
    """
	print(io, nm)
end

"""
MGMM model with parameters
- β = {μ, Σ}
- χ
- α
- γ
- Φ

The Mixture of Gaussian Mixture model from paper
Hierarchical Probabilistic Models for Group Anomaly Detection
available at http://proceedings.mlr.press/v15/xiong11a/xiong11a.pdf
"""
mutable struct MGMM
    β::β_par
    χ
    α
    γ
    Φ
end

function Base.show(io::IO, m::MGMM)
    nm = """MGMM model
        Parameters:
            β = {μ, Σ}, χ, α, γ, Φ
        Size:
            T = $(size(m.α,1)) (topics)
            K = $(size(m.χ,1)) (clusters)
    """
	print(io, nm)
end

"""
    MGMM_constructor(data;K::Int, T::Int, init_seed=nothing)

Constructs the MGMM model. For given data (for input dimension), K, and T,
returns a randomly initialized model.
"""
function MGMM_constructor(data;K::Int, T::Int, init_seed=nothing, kwargs...)
    idim = size(data[1],1)
    M = length(data)
    Nm = size.(data, 2)

    # if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing

    alpha = normalize(rand(T),1)
    chi = rand(Dirichlet(repeat([2],K)),T)
    #beta = β_par(randn(idim,K) .* 2, ones(idim,K) ./ 2)
    beta = β_par(randn(idim,K), ones(idim,K))
    gamma = rand(Dirichlet(repeat([2],T)),M)
    phi = [rand(Dirichlet(repeat([2],K)),Nm[m]) for m in 1:M]

    # reset seed
	(init_seed !== nothing) ? Random.seed!() : nothing

    return MGMM(beta, chi, alpha, gamma, phi)
end

"""
    toMixtureModel(m::MGMM)

Constructs a MixtureModel from Distributions package.
"""
function toMixtureModel(m::MGMM)
    β = m.β
    χ = m.χ
    α = m.α
    mm = [
        MixtureModel([MvNormal(β.μ[:,j],β.Σ[:,j]) for j in 1:size(β.μ,2)],χ[:,i])
        for i in 1:size(χ,2)
    ]
    MixtureModel(mm, α[:])
end

####################################################
### functions to calculate parameter estimations ###
####################################################

function phi_func(data, m::MGMM)
    gamma, chi, beta = m.γ, m.χ, m.β
    K = size(chi, 1)
    Cm = chi * gamma
    MM = [MvNormal(beta.μ[:,k],beta.Σ[:,k]) for k in 1:K]
    ph = [vcat([logpdf(MM[k],Gm) for k in 1:K]'...) for Gm in data]
    
    for m in 1:length(data)
        ph[m] = ph[m] .+ Cm[:,m]
    end

    phi_new = softmax.(ph)
    return phi_new
end

function gamma_func(m::MGMM)
    gamma, alpha, phi, chi = m.γ, m.α, m.Φ, m.χ
    T = size(chi, 2)
    gamma_new = similar(gamma)
    for m in 1:size(gamma,2)
        gamma_star = softmax([log(alpha[t]) + sum(phi[m]'*log.(chi[:,t])) for t in 1:T])
        gamma_new[:,m] = gamma_star
    end
    return gamma_new
end

alpha_func(m::MGMM) = 1/sum(m.γ)*sum(m.γ,dims=2)

function chi_func(m::MGMM)
    phi, gamma = m.Φ, m.γ
    sphi = hcat(sum.(phi,dims=2)...)
    w = (1 ./ sum(sphi*gamma',dims=1))[:]
    chi_new = hcat(w .* eachrow((sphi*gamma')')...)
    return chi_new
end

"""
    star(data, m::MGMM)

Calculates new estimated parameters for the model.
"""
function star(data, m::MGMM)
    gamma, alpha, chi, beta = m.γ, m.α, m.χ, m.β
    phi = phi_func(data, m)
    m.Φ = phi
    gamma = gamma_func(m)
    m.γ = gamma
    alpha = alpha_func(m)
    m.α = alpha
    chi = chi_func(m)
    m.χ = chi
    return m
end

# return the new β parameters for Gaussian Mixture
"""
    beta_star(data, m::MGMM; varfloor = 1e-3, output = false)

Calculates new means and variances of Gaussian mixtures.
If variances get too small, initialize them randomly again.
If `output = true`, prints a warning everytime the variance
is floored.
"""
function beta_star(data, m::MGMM; varfloor = 1e-4, output = false)
    phi = m.Φ
    X = hcat(data...)

    Φ = hcat(phi...)
    NN = sum(Φ, dims=2)
    F = Φ * X'
    μ1 = F ./ NN
    S = Φ * (X .* X)'
    Σ2 = S ./ NN - μ1 .^ 2

    tooSmall = any(Σ2 .< varfloor, dims=2)[:]
    if (any(tooSmall))
        ind = findall(tooSmall)
        if output
            @warn("Variances had to be floored ", ind)
        end
        Σ2[ind, :] .= 1
        μ1[ind, :] .= randn(size(μ1[ind, :]))
    end

    beta = β_par(collect(μ1'),collect(Σ2'))
end

# calculates the number of parameters in the β dictionary
Base.length(β::β_par) = length(β.μ) + length(β.Σ)

"""
    number_of_parameters(m::MGMM)

Calculates the number of parameters for the MGMM model.
"""
function number_of_parameters(m::MGMM)
    nbeta = length(m.β)
    nchi = length(m.χ)
    nalpha = length(m.α)
    return nbeta + nchi + nalpha
end

function Base.isnan(m::MGMM)
    not_nan = !isnan(sum(m.β.μ))
    not_nan = not_nan * !isnan(sum(m.β.Σ))
    not_nan = not_nan * !isnan(sum(m.χ)) * !isnan(sum(m.α)) * !isnan(sum(hcat(m.Φ...)))
    return !not_nan
end

"""
StatsBase.fit!(model::MGMM, data::Tuple, loss::Function; max_train_time=82800, lr=0.001, 
    batchsize=64, patience=30, check_interval::Int=10, kwargs...)

Function to fit MGMM model.
"""
function StatsBase.fit!(model::MGMM, data::Tuple, loss::Function;
max_iters=1000, max_train_time=82800, patience=100,
check_interval::Int=1, kwargs...)

    history = MVHistory()

    tr_model = deepcopy(model)
    _patience = patience

    # prepare data for bag model
    tr_x, tr_l = unpack_mill(data[1])
    vx, vl = unpack_mill(data[2])
    val_x = vx[vl .== 0]

    best_val_loss = Inf
    i = 1
    start_time = time()

    lossf(x) = loss(tr_model, x)

    # infinite for loop via RandomBatches
    for batch in RandomBatches(tr_x, 10)

        # do EM for MGMM
        tr_model = star(tr_x, tr_model)

        if isnan(tr_model)
            @warn "NaN alpha values. Training stopped."
            break
        end

        tr_model.β = beta_star(tr_x, tr_model)
        # only batch training loss
        train_loss = mean(lossf.(tr_x))

        push!(history, :training_loss, i, train_loss)
        if mod(i, check_interval) == 0

            # validation/early stopping
            val_loss = mean(lossf.(val_x))
            @info "$i - loss: $(train_loss) (batch) | $(val_loss) (validation)"

            if isnan(val_loss) || isnan(train_loss)
				error("Encountered invalid values in loss function.")
			end

            push!(history, :validation_likelihood, i, val_loss)
            
            if val_loss < best_val_loss
                best_val_loss = val_loss
                _patience = patience

                # this should save the model at least once
                # when the validation loss is decreasing 
                model = deepcopy(tr_model)
            else # else stop if the model has not improved for `patience` iterations
                _patience -= 1
                if _patience == 0
                    @info "Stopped training after $(i) iterations."
                    break
                end
            end
        end
        if (time() - start_time > max_train_time) | (i > max_iters) # stop early if time is running out
            model = deepcopy(tr_model)
            @info "Stopped training after $(i) iterations, $((time() - start_time) / 3600) hours."
            break
        end
        i += 1
    end
    # again, this is not optimal, the model should be passed by reference and only the reference should be edited
    (history = history, iterations = i, model = model, npars = number_of_parameters(model))
end

#######################
### Score functions ###
#######################

function scale_me(x)
    min = minimum(x)
    max = maximum(x)
    r = max - min
    return @. (x - min)/r
end

# calculate phi for points that were not in training data
function phi_score(Gm, m::MGMM)
    beta = m.β
    K = size(beta.μ, 2)
    MM = [MvNormal(beta.μ[:,k],beta.Σ[:,k]) for k in 1:K]
    ph = vcat([pdf(MM[k],Gm) for k in 1:K]'...)
    γ = hcat(normalize.(eachcol(ph),1)...)
    # check NaN values and if so -> equal probability? random probability?
    for (i,col) in enumerate(eachcol(γ))
        if sum(isnan.(col))> 0
            γ[:,i] = normalize(rand(length(col)),1)
        end
    end
    return γ
end

"""
    topic_score(m::MGMM, data::Mill.Bagnode)

Returns the topic score for data.
"""
function topic_score(m::MGMM, x::AbstractArray; n = 1000)
    beta, chi, Alpha = m.β, m.χ, m.α
    T = size(chi, 2)
    φa = phi_score(x, m)
    θa = normalize(sum(φa, dims=2),1)[:]
    # what to do if we get NaN values?
    # meaning that some points don't belong to either clusters
    Z = rand(Multinomial(n,θa))
    multinomial_model = MixtureModel([Multinomial(n,chi[:,t]) for t in 1:T],Alpha[:])
    -mean(logpdf(multinomial_model, Z))
end
function topic_score(m::MGMM, data::Mill.BagNode)
    dt, _ = unpack_mill((data,[]))
    map(x -> topic_score(m, x), dt)
end

"""
    point_score(m::MGMM, data::Mill.Bagnode)

Returns the point score for data.
"""
function point_score(m::MGMM, x::AbstractArray)
    MM = toMixtureModel(m)
    -sum(logpdf(MM, x))
end
function point_score(m::MGMM, data::Mill.BagNode)
    dt, _ = unpack_mill((data,[]))
    map(x -> point_score(m, x), dt)
end

"""
    MGMM_score(m::MGMM, data)

Calculates the estimated joint score for MGMM model. Both topic
and point scores are calculated and scaled to [0,1]. Then the scores
are added up to make the final score.
"""
function MGMM_score(m::MGMM, data)
    dt, _ = unpack_mill((data,[]))
    TS = map(x -> topic_score(m, x), dt)
    PS = map(x -> point_score(m, x), dt)

    TS_scaled = scale_me(TS)
    PS_scaled = scale_me(PS)
    TS_scaled .+ PS_scaled
end