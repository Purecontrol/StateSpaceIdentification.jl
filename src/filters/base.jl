const POSITIVE_PRECISION = 10e-3

"""
$(TYPEDEF)

`AbstractFilter` is an abstract type representing a method used for filtering. In other words, if the model
is a state-space model, and given `N` data points, the filter computes:

```math
\\forall t \\in \\{1, ..., N\\}, \\mathbb{P}(X_t \\mid y_1, ..., y_{t-1})
```

Each filter has its own methods for computing the distribution described above.
"""
abstract type AbstractFilter{Z <: Real} end

"""
$(TYPEDEF)

`AbstractFilterOutput` is an abstract type representing the structure for storing all the
outputs of the different filters.
"""
abstract type AbstractFilterOutput{Z <: Real} end

"""
$(TYPEDEF)

`AbstractFilterState` is an abstract type representing the structure for storing all
the information at each step of the filtering process.
"""
abstract type AbstractFilterState{Z <: Real} end

"""
$(TYPEDSIGNATURES)

Abstract ``get_filter_output`` function that has to be defined for all subtypes of AbstractFilter.
"""
function get_filter_output(
        filter_method::F,
        model::ForecastingModel,
        observation_data
)::AbstractFilterOutput where {F <: AbstractFilter}
    return error("The function `get_filter_output` has to be defined for subtype of AbstractFilter.")
end

"""
$(TYPEDSIGNATURES)

Abstract ``filtering!(s::AbstractStateSpaceSystem, f::AbstractFilter, ...)`` function that has to be defined for all subtypes of AbstractFilter.
"""
function filtering!(
        sys::AbstractStateSpaceSystem{Z},
        filter_method::AbstractFilter{Z},
        observation_data,
        exogenous_data,
        control_data,
        parameters
) where {Z <: Real}
    return error("The function `filtering!(s::AbstractStateSpaceSystem, f::AbstractFilter, ...)` has to be defined for subtype of AbstractFilter.")
end

"""
$(TYPEDSIGNATURES)

Abstract ``filtering!(fo::AbstractFilterOutput, s::AbstractStateSpaceSystem, f::AbstractFilter, ...)`` function that has to be defined for all subtypes of AbstractFilter.
"""
function filtering!(
        filter_output::AbstractFilterOutput{Z},
        sys::AbstractStateSpaceSystem{Z},
        filter_method::AbstractFilter{Z},
        observation_data,
        exogenous_data,
        control_data,
        parameters
) where {Z <: Real}
    return error("The function `filtering!(fo::AbstractFilterOutput, s::AbstractStateSpaceSystem, f::AbstractFilter, ...)` has to be defined for subtype of AbstractFilter.")
end

#################################################################################################
# Deterministic Filters
#################################################################################################

"""
$(TYPEDEF)

`AbstractDeterministicFilter` is a subtype of `AbstractFilter` for filters that compute the distribution by 
directly determining the parameters of the distribution. For example, if the distribution is a Gaussian,
the filter computes estimates of the mean (`\\mu`) and variance (`\\sigma^2`), which fully characterize the distribution.
"""
abstract type AbstractDeterministicFilter{Z <: Real} <: AbstractFilter{Z} end

"""
$(TYPEDEF)

`AbstractGaussianDeterministicFilter` is a subtype of `AbstractDeterministicFilter` designed for target distributions that are
Gaussian. The output distributions are `TimeSeries` of `GaussianStateStochasticProcess`.
"""
abstract type AbstractGaussianDeterministicFilter{Z <: Real} <: AbstractDeterministicFilter{Z} end

"""
$(TYPEDEF)

`AbstractGaussianFilterOutput` is a subtype of `AbstractFilterOutput` to store the outputs of 
`AbstractGaussianDeterministicFilter`.
"""
abstract type AbstractGaussianFilterOutput{Z} <: AbstractFilterOutput{Z} end

#################################################################################################
# Stochastic Filters
#################################################################################################

"""
$(TYPEDEF)

`AbstractSimulationBasedFilter` is a subtype of `AbstractFilter` for filters that approximate the distribution using 
samples. At each timestep, the distribution is approximated using a sample of `n_particles` particles.
"""
abstract type AbstractSimulationBasedFilter{Z <: Real} <: AbstractFilter{Z} end

"""
$(TYPEDEF)

`AbstractStochasticMonteCarloFilter` is a subtype of `AbstractSimulationBasedFilter` where the sampling uses Sequential Monte Carlo
methods.
"""
abstract type AbstractStochasticMonteCarloFilter{Z <: Real} <: AbstractSimulationBasedFilter{Z} end

"""
$(TYPEDEF)

`AbstractStochasticMonteCarloFilterOutput` is a subtype of `AbstractFilterOutput` to store the outputs of 
`AbstractStochasticMonteCarloFilter`.
"""
abstract type AbstractStochasticMonteCarloFilterOutput{Z} <: AbstractFilterOutput{Z} end