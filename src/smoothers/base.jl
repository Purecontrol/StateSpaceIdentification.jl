"""
$(TYPEDEF)

`AbstractSmoother` is an abstract type representing a method used for smoothing. In other words, if the model
is a state-space model, and given `N` data points, the smoother computes:

```math
\\forall t \\in \\{1, ..., N\\}, \\mathbb{P}(X_t \\mid y_1, ..., y_{N})
```

Each smoother has its own methods for computing the distribution described above and most of the times, it is
based on the output of a filter.
"""
abstract type AbstractSmoother end

"""
$(TYPEDEF)

`AbstractSmootherOutput` is an abstract type representing the structure for storing all the
outputs of the different smoothers.
"""
abstract type AbstractSmootherOutput{Z <: Real} end

"""
$(TYPEDEF)

`AbstractSmootherState` is an abstract type representing the structure for storing all
the information at each step of the smoothing process.
"""
abstract type AbstractSmootherState{Z <: Real} end

"""
$(TYPEDSIGNATURES)

Abstract ``get_smoother_output`` function that has to be defined for all subtypes of AbstractSmoother.
"""
function get_smoother_output(
        smoother_method::S,
        model::ForecastingModel,
        observation_data
)::AbstractSmootherOutput where {S <: AbstractSmoother}
    return error("The function `get_smoother_output` has to be defined for subtype of AbstractSmoother.")
end


"""
$(TYPEDSIGNATURES)

Abstract ``smoothing!(so::AbstractSmootherOutput, fo::AbstractFilterOutput, sys::AbstractStateSpaceSystem, s::AbstractSmoother, ...)`` function that has to be defined for all subtypes of AbstractSmoother.
"""
function smoothing!(
        smoother_output::AbstractSmootherOutput,
        filter_output::AbstractFilterOutput,
        sys::AbstractStateSpaceSystem,
        smoother_method::S,
        observation_data,
        exogenous_data,
        control_data,
        parameters
) where {S <: AbstractSmoother}
    return error("The function `smoothing!(so::AbstractSmootherOutput, fo::AbstractFilterOutput, sys::AbstractStateSpaceSystem, s::AbstractSmoother, ...)` has to be defined for subtype of AbstractSmoother.")
end

#################################################################################################
# Deterministic Smoothers
#################################################################################################

"""
$(TYPEDEF)

`AbstractDeterministicSmoother` is a subtype of `AbstractSmoother` for smoothers that compute the distribution by 
directly determining the parameters of the distribution. For example, if the distribution is a Gaussian,
the smoother computes estimates of the mean (`\\mu`) and variance (`\\sigma^2`), which fully characterize the distribution.
"""
abstract type AbstractDeterministicSmoother <: AbstractSmoother end

"""
$(TYPEDEF)

`AbstractGaussianDeterministicSmoother` is a subtype of `AbstractDeterministicSmoother` designed for target distributions that are
Gaussian. The output distributions are `TimeSeries` of `GaussianStateStochasticProcess`.
"""
abstract type AbstractGaussianDeterministicSmoother <: AbstractDeterministicSmoother end

"""
$(TYPEDEF)

`AbstractGaussianSmootherOutput` is a subtype of `AbstractSmootherOutput` to store the outputs of 
`AbstractGaussianDeterministicSmoother`.
"""
abstract type AbstractGaussianSmootherOutput{Z} <: AbstractSmootherOutput{Z} end

#################################################################################################
# Stochastic Smoothers
#################################################################################################

"""
$(TYPEDEF)

`AbstractSimulationBasedSmoother` is a subtype of `AbstractSmoother` for smoothers that approximate the distribution using 
samples. At each timestep, the distribution is approximated using a sample of `n_particles` particles.
"""
abstract type AbstractSimulationBasedSmoother <: AbstractSmoother end

"""
$(TYPEDEF)

`AbstractStochasticMonteCarloSmoother` is a subtype of `AbstractSimulationBasedSmoother` where the sampling uses Sequential Monte Carlo
methods.
"""
abstract type AbstractStochasticMonteCarloSmoother <: AbstractSimulationBasedSmoother end

"""
$(TYPEDEF)

`AbstractStochasticMonteCarloSmootherOutput` is a subtype of `AbstractSmootherOutput` to store the outputs of 
`AbstractStochasticMonteCarloSmoother`.
"""
abstract type AbstractStochasticMonteCarloSmootherOutput{Z} <: AbstractSmootherOutput{Z} end