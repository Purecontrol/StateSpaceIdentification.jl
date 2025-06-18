module StateSpaceIdentification

using Distributions
using DocStringExtensions
using LinearAlgebra
using ComponentArrays

DEFAULT_REAL_TYPE = Float64

include("utils.jl")
include("time_series.jl")
include("systems/base.jl")
include("models.jl")

include("systems/gaussian_linear_state_space.jl")
include("systems/non_linear_gaussian_state_space.jl")
include("systems/non_parametric_gaussian_state_space.jl")

include("filters/base.jl")
include("smoothers/base.jl")
include("filter_smoother.jl")

include("filters/kalman_filter.jl")
include("filters/ensemble_kalman_filter.jl")
include("filters/extended_kalman_filter.jl")
include("filters/particle_filters.jl")

include("smoothers/kalman_smoother.jl")
include("smoothers/ensemble_kalman_smoother.jl")
include("smoothers/extended_kalman_smoother.jl")
include("smoothers/particle_smoothers.jl")

include("fit.jl")


export GaussianStateStochasticProcess, ParticleSwarmState, TimeSeries
export ForecastingModel, default_filter, default_smoother
export GaussianLinearStateSpaceSystem, GaussianNonLinearStateSpaceSystem, GaussianNonParametricStateSpaceSystem, LocalLinearRegressor
export filtering, update, update!, smoothing, filtering_and_smoothing
export KalmanFilter, KalmanSmoother
export EnsembleKalmanFilter, EnsembleKalmanSmoother
export ExtendedKalmanFilter, ExtendedKalmanSmoother
export ParticleFilter, ConditionalParticleFilter, AncestorTrackingSmoother, BackwardSimulationSmoother
export ExpectationMaximization, numerical_MLE #better to have only one fit function

end