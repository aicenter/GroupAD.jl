####################################
### NeuralStatistician functions ###
####################################

# likelihoods

"""
    likelihood(model::NeuralStatistician, bag)

Calculates the likelihood of instances in a bag
and returns a vector.
"""
function likelihood(model::NeuralStatistician, bag)
    v = model.instance_encoder(bag)
    p = mean(v,dims=2)
    c = rand(model.encoder_c, p)
    h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
    z = rand(model.encoder_z, h)
    llh = -logpdf(model.decoder, bag, z)
end

function mean_likelihood(model::NeuralStatistician, bag)
    v = model.instance_encoder(bag)
    p = mean(v,dims=2)
    c = mean(model.encoder_c, p)
    h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
    z = mean(model.encoder_z, h)
    llh = -logpdf(model.decoder, bag, z)
end

function elbo(m::NeuralStatistician, x::AbstractArray;β1=1.0,β2=1.0)
    # instance network
    v = m.instance_encoder(x)
    p = mean(v, dims=2)

    # sample latent for context
    c = rand(m.encoder_c, p)

    # sample latent for instances
    h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
    z = rand(m.encoder_z, h)
	
    # 3 terms - likelihood, kl1, kl2
    llh = mean(logpdf(m.decoder, x, z))
    kld1 = mean(kl_divergence(condition(m.encoder_c, v), m.prior_c))
    kld2 = mean(kl_divergence(condition(m.encoder_z, h), condition(m.conditional_z, c)))
    llh - β1 * kld1 - β2 * kld2
end

# kl scores
function kl_scores(model::NeuralStatistician, bag, space="c")
    v = model.instance_encoder(bag)
    p = mean(v, dims=2)
    c = rand(model.encoder_c, p)
    h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
    z = rand(model.encoder_z, h)
    if space == "c"
        return kld1 = mean(kl_divergence(condition(model.encoder_c, v), model.prior_c))
    elseif space == "z"
        kld2 = mean(kl_divergence(condition(model.encoder_z, h), condition(model.conditional_z, c)))
    else
        error("Choose latent space either space=\"c\" or space=\"z\"!")
    end
end

#######################
### VAE likelihoods ###
#######################

function likelihood(model::VAE, bag)
    z = rand(model.encoder, bag)
    llh = -logpdf(model.decoder, bag, z)
end

function mean_likelihood(model::VAE, bag)
    z = mean(model.encoder, bag)
    llh = -logpdf(model.decoder, bag, z)
end

### score functions ###

"""
    reconstruction_score(model, bag, [pc])

Returns sum of log-likelihoods for instances in the bag.
When `pc` is specified, corrects the value with cardinality.
"""
function reconstruction_score(model, bag)
    llh = likelihood(model, bag)
    sum(llh)
end
function reconstruction_score(model, bag, pc)
    n = size(bag,2)
    llh = sum(likelihood(model,bag)) - logpdf(pc,n)
end

"""
    mean_reconstruction_score(model, bag)

Returns the sum of mean instance log-likelihoods.
"""
function mean_reconstruction_score(model, bag)
    llk = sum(mean_likelihood(model, bag))
end
function mean_reconstruction_score(model, bag, pc)
    n = size(bag,2)
    llk = sum(mean_likelihood(model, bag)) - logpdf(pc,n)
end

"""
    reconstruction_score_instance_mean(model, bag)
"""
function reconstruction_score_instance_mean(model, bag)
    llh = likelihood(model, bag)
    mean(llh)
end
function reconstruction_score_instance_mean(model, bag, pc)
    n = size(bag,2)
    llh = mean(likelihood(model,bag)) - logpdf(pc,n)
end

################# Reconstruction ##########################
function reconstruct(model::NeuralStatistician, bag)
    v = model.instance_encoder(bag)
    p = mean(v,dims=2)
    c = rand(model.encoder_c, p)
    h = hcat([vcat(v[1:end,i], c) for i in 1:size(v, 2)]...)
    z = rand(model.encoder_z, h)
    llh = -logpdf(model.decoder, bag, z)
    mean(model.decoder, z)
end