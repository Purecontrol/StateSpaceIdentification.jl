using LinearAlgebra

mutable struct KalmanSmootherState{Z} <: AbstractFilterState{Z}

    # Filtered and predicted state
    smoothed_state_μ::Vector{Z}
    smoothed_state_Σ::Matrix{Z}
    autocov_state::Matrix{Z}

    # Matrices and vectors
    r::Vector{Z}
    N::Matrix{Z}

    function KalmanSmootherState{Z}(n_X, n_Y) where {Z <: Real}
        new{Z}(zeros(Z, n_X), zeros(Z, n_X, n_X), zeros(Z, n_X, n_X), zeros(Z, n_X), zeros(Z, n_X, n_X))
    end
end

mutable struct KalmanSmoother{Z <: Real} <: AbstractGaussianDeterministicSmoother{Z}
    state::KalmanSmootherState{Z}

    function KalmanSmoother{Z}(n_X, n_Y) where {Z <: Real}
        new{Z}(KalmanSmootherState{Z}(n_X, n_Y))
    end

    function KalmanSmoother(model::ForecastingModel{Z, T, S}) where {Z <: Real, T <: AbstractStateSpaceSystem, S <: AbstractState{Z}}
        return KalmanSmoother{Z}(model.system.n_X, model.system.n_Y)
    end
end

mutable struct KalmanSmootherOutput{Z <: Real} <: AbstractGaussianSmootherOutput{Z}

    # Predicted and filtered states
    smoothed_state::TimeSeries{Z, GaussianStateStochasticProcess{Z}}
    autocov_state::Vector{Matrix{Z}}

    r::Vector{Vector{Z}}
    N::Vector{Matrix{Z}}

    function KalmanSmootherOutput(model::ForecastingModel{Z, T, St}, y_t) where {Z <: Real, T <: AbstractStateSpaceSystem, St <: AbstractState{Z}}
        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt) * (t - 1)
                   for t in 1:(n_obs + 1)]

        smoothed_state = TimeSeries{Z, GaussianStateStochasticProcess{Z}}(
            t_index, n_obs + 1, model.system.n_X)
        autocov_state = Array{Array{Z, 2}, 1}(undef, n_obs)

        # Define matricess
        r = Array{Array{Z, 1}, 1}(undef, n_obs + 1)
        N = Array{Array{Z, 2}, 1}(undef, n_obs + 1)

        return new{Z}(smoothed_state, autocov_state, r, N)
    end
end

function get_smoother_output(smoother_method::KalmanSmoother, model::ForecastingModel, observation_data)
    return KalmanSmootherOutput(model, observation_data)
end


function smoothing!(
        smoother_output::KalmanSmootherOutput{Z},
        filter_output::KalmanFilterOutput{Z},
        sys::GaussianLinearStateSpaceSystem{Z},
        smoother_method::KalmanSmoother{Z},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        parameters::Vector{Z}
) where {Z <: Real}
    n_obs = size(observation_data, 1)

    # Initialize smoother
    initialize_smoother!(smoother_output, smoother_method.state, filter_output.predicted_state[end])

    t_step_table = collect(range(
        filter_output.predicted_state[1].t, length = n_obs, step = sys.dt))

    # Backward recursions
    @inbounds for (t, t_step) in Iterators.reverse(enumerate(t_step_table))

        # Get current matrix H
        H = sys.H_t(exogenous_data[t, :], parameters, t_step)#::Matrix{Z}

        # Get output from filter
        inv_S = inv(filter_output.S[t])
        v = filter_output.v[t]
        L = filter_output.L[t]
        predicted_state_μ = filter_output.predicted_state[t].μ_t
        predicted_state_Σ = filter_output.predicted_state[t].Σ_t
        predicted_state_Σ_lag_1 = filter_output.predicted_state[t + 1].Σ_t

        # Update state
        update_smoother_state!(
            smoother_method.state,
            view(observation_data, t, :),
            H,
            inv_S,
            v,
            L,
            predicted_state_μ,
            predicted_state_Σ,
            predicted_state_Σ_lag_1
        )

        save_state_in_smoother_output!(smoother_output, smoother_method.state, t)
    end

    return smoother_output
end

function update_smoother_state!(
        kalman_state::KalmanSmootherState{Z},
        y::SubArray{Z},
        H,
        inv_S::Matrix{Z},
        v::Vector{Z},
        L::Matrix{Z},
        predicted_state_μ::Vector{Z},
        predicted_state_Σ::Matrix{Z},
        predicted_state_Σ_lag_1::Matrix{Z}
) where {Z <: Real}

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Update autocovariance
    kalman_state.autocov_state = (predicted_state_Σ * transpose(L) *
                                  (I - kalman_state.N * predicted_state_Σ_lag_1))'

    # Backward step
    kalman_state.r = transpose(H[ivar_obs, :]) * inv_S * v + transpose(L) * kalman_state.r
    kalman_state.N = transpose(H[ivar_obs, :]) * inv_S * H[ivar_obs, :] +
                     transpose(L) * kalman_state.N * L

    # Update smoothed state and covariance
    kalman_state.smoothed_state_μ = predicted_state_μ + predicted_state_Σ * kalman_state.r
    kalman_state.smoothed_state_Σ = predicted_state_Σ -
                                    predicted_state_Σ * kalman_state.N * predicted_state_Σ
end

function save_state_in_smoother_output!(
        smoother_output::KalmanSmootherOutput{Z},
        smoother_state::KalmanSmootherState{Z},
        t::Int64
) where {Z <: Real}

    # Save smoothed state
    smoother_output.smoothed_state[t].μ_t = smoother_state.smoothed_state_μ
    smoother_output.smoothed_state[t].Σ_t = smoother_state.smoothed_state_Σ
    smoother_output.autocov_state[t] = smoother_state.autocov_state

    # Save matrix values
    smoother_output.N[t] = smoother_state.N
    smoother_output.r[t] = smoother_state.r
end

function initialize_smoother!(
        smoother_output::KalmanSmootherOutput{Z},
        smoother_state::KalmanSmootherState{Z},
        last_predicted_state::GaussianStateStochasticProcess{Z}
) where {Z <: Real}

    # Initialize KalmanSmoother state
    smoother_state.smoothed_state_μ = last_predicted_state.μ_t
    smoother_state.smoothed_state_Σ = last_predicted_state.Σ_t

    # Save initial predicted state
    smoother_output.smoothed_state[end].μ_t = smoother_state.smoothed_state_μ
    smoother_output.smoothed_state[end].Σ_t = smoother_state.smoothed_state_Σ

    # Save initial value of r and N
    smoother_output.N[end] = smoother_state.N
    smoother_output.r[end] = smoother_state.r
end
