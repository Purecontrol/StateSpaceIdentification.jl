module StateSpaceIdentification

using Distributions
using DocStringExtensions
using LinearAlgebra
using InteractiveUtils

include("utils.jl")
include("time_series.jl")
include("systems/base.jl")
include("models.jl")

include("systems/gaussian_linear_state_space.jl")
include("systems/non_linear_gaussian_state_space.jl")
# include("systems/non_parametric_gaussian_state_space.jl")

include("filters/base.jl")
include("smoothers/base.jl")
include("filter_smoother.jl")

include("filters/kalman_filter.jl")
include("filters/ensemble_kalman_filter.jl")
include("filters/extended_kalman_filter.jl")
# include("filters/particle_filter.jl")
# include("filters/conditional_particle_filter.jl")

include("smoothers/kalman_smoother.jl")
include("smoothers/ensemble_kalman_smoother.jl")
include("smoothers/extended_kalman_smoother.jl")
# include("smoothers/ancestor_tracking_smoother.jl")
# include("smoothers/backward_simulation_smoother.jl")

include("fit.jl")

# export time_series, models, fit, filter_smoother



export GaussianStateStochasticProcess, ParticleSwarmState, TimeSeries
export ForecastingModel, default_filter, default_smoother
export GaussianLinearStateSpaceSystem, GaussianNonLinearStateSpaceSystem
export filtering, update, update!, smoothing, filtering_and_smoothing
export KalmanFilter, KalmanSmoother
export EnsembleKalmanFilter, EnsembleKalmanSmoother
export ExtendedKalmanFilter, ExtendedKalmanSmoother
export ExpectationMaximization



export numerical_MLE, EM, EM_EnKS, SEM, SEM_CPF, npSEM_CPF, LLR

###### DEV IMPORT : TO STANDARDIZE #######
export EM_EnKS2

export k_choice

export EnsembleKalmanFilter, KalmanFilter, ParticleFilter, ConditionalParticleFilter
export KalmanSmoother,
       EnsembleKalmanSmoother, BackwardSimulationSmoother, AncestorTrackingSmoother


end # module StateSpaceIdentification
