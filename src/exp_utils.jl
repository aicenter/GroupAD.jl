"""
    expdir(args...)

Returns the path to folder `experimental` in the project.
Acts the same as e. g. `datadir`.
"""
function expdir(args...)
	return projectdir("experimental", args...)
end

using DistributionsAD, ConditionalDists, IPMeasures, Distributions
using IPMeasures: _kld_gaussian
const TuMvNormal = Union{DistributionsAD.TuringDenseMvNormal,
                         DistributionsAD.TuringDiagMvNormal,
                         DistributionsAD.TuringScalMvNormal}

IPMeasures.kl_divergence(p::ConditionalDists.BMN, q::ConditionalDists.BMN) = _kld_gaussian(p,q)
IPMeasures.kl_divergence(p::MvNormal, q::MvNormal) = _kld_gaussian(p,q)
IPMeasures.kl_divergence(p::TuMvNormal, q::TuMvNormal) = _kld_gaussian(p,q)
IPMeasures.kl_divergence(p::ConditionalDists.BMN, q::MvNormal) = _kld_gaussian(p,q)
IPMeasures.kl_divergence(p::ConditionalDists.BMN, q::TuMvNormal) = _kld_gaussian(p,q)
IPMeasures.kl_divergence(p::ConditionalDists.BMN, q::ConditionalDists.BMN) = _kld_gaussian(p,q)
