module StateSpaceIdentification

include("utils.jl")
include("time_series.jl")
include("systems/base.jl")
include("models.jl")

include("systems/non_linear_gaussian_state_space.jl")
include("systems/gaussian_linear_state_space.jl")

include("filters/base.jl")
include("smoothers/base.jl")
include("filter_smoother.jl")

include("filters/ensemble_kalman_filter.jl")
include("filters/kalman_filter.jl")
include("filters/particle_filter.jl")

include("smoothers/ancestor_sampling.jl")
include("smoothers/ancestor_tracking_smoother.jl")
include("smoothers/backward_simulation_smoother.jl")
include("smoothers/ensemble_kalman_smoother.jl")
include("smoothers/kalman_smoother.jl")

include("fit.jl")


export time_series, models, fit, filter_smoother

export GaussianLinearStateSpaceSystem
export GaussianNonLinearStateSpaceSystem

export GaussianStateStochasticProcess
export ForecastingModel

export numerical_MLE
export EM
export EM_EnKS
export SEM

###### DEV IMPORT : TO STANDARDIZE #######
export EM_EnKS2

export filter, smoother, forecast

export EnsembleKalmanFilter, KalmanFilter, ParticleFilter, ConditionalParticleFilter
export KalmanSmoother, EnsembleKalmanSmoother, BackwardSimulationSmoother, AncestorTrackingSmoother

end # module StateSpaceIdentification
