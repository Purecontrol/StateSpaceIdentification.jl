using LinearAlgebra


mutable struct ExtendedKalmanSmootherState

    # Filtered and predicted state
    smoothed_state_μ#::Vector{Float64}
    smoothed_state_σ#::Matrix{Float64}
    autocov_state

    # Matrices and vectors
    r#::Matrix{Float64}
    N#::Matrix{Float64}


    function ExtendedKalmanSmootherState(n_X, n_Y)
        
        new(zeros(n_X), zeros(n_X, n_X), zeros(n_X, n_X), zeros(n_X), zeros(n_X, n_X))

    end

end


mutable struct ExtendedKalmanSmoother <: AbstractSmoother

    smoother_state::ExtendedKalmanSmootherState

    function ExtendedKalmanSmoother(n_X, n_Y)

        new(ExtendedKalmanSmootherState(n_X, n_Y))

    end


    function ExtendedKalmanSmoother(model::ForecastingModel)

        new(ExtendedKalmanSmootherState(model.system.n_X, model.system.n_Y))

    end

end


mutable struct ExtendedKalmanSmootherOutput <: SmootherOutput

    # Predicted and filtered states
    smoothed_state::TimeSeries{GaussianStateStochasticProcess}
    autocov_state::Array{Array{Float64, 2}, 1}
  
    r::Array{Array{Float64, 1}, 1}
    N::Array{Array{Float64, 2}, 1}

    function ExtendedKalmanSmootherOutput(model::ForecastingModel, y_t)

        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]

        smoothed_state = TimeSeries{GaussianStateStochasticProcess}(n_obs+1, model.system.n_X, t_index)
        autocov_state = Array{Array{Float64, 2}, 1}(undef, n_obs)

        # Define matricess
        r = Array{Array{Float64, 1}, 1}(undef, n_obs+1)
        N = Array{Array{Float64, 2}, 1}(undef, n_obs+1)

        return new(smoothed_state, autocov_state, r, N)

    end

end


function get_smoother_output(smoother::ExtendedKalmanSmoother, model, y_t)

    return ExtendedKalmanSmootherOutput(model, y_t)

end


function type_of_state(kf::ExtendedKalmanSmoother)

    return GaussianStateStochasticProcess

end


# General KF
function smoother!(smoother_output::ExtendedKalmanSmootherOutput, filter_output::ExtendedKalmanFilterOutput, sys::GaussianNonLinearStateSpaceSystem, smoother::ExtendedKalmanSmoother, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    initialize_smoother!(smoother_output, smoother.smoother_state, filter_output.predicted_state[end])

    # Backward recursions
    @inbounds for t in (n_obs):-1:1

        # Get current t_step
        t_step = filter_output.predicted_state[1].t + (t-1)*sys.dt

        # Get current matrix H
        H = sys.dH_t(exogenous_variables[t, :], parameters, t_step)
        inv_S = inv(filter_output.S[t])
        v = filter_output.v[t]
        L = filter_output.L[t]
        predicted_state_μ = filter_output.predicted_state[t].μ_t
        predicted_state_σ = filter_output.predicted_state[t].σ_t
        predicted_state_σ_lag_1 = filter_output.predicted_state[t+1].σ_t

        update_smoother_state!(smoother.smoother_state, y_t[t, :], H, inv_S, v, L, predicted_state_μ, predicted_state_σ, predicted_state_σ_lag_1)

        save_state_in_smoother_output!(smoother_output, smoother.smoother_state, t)

    end

    return smoother_output

end


function update_smoother_state!(kalman_state, y, H, inv_S, v, L, predicted_state_μ, predicted_state_σ, predicted_state_σ_lag_1)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Update autocovariance
    kalman_state.autocov_state = predicted_state_σ*transpose(L)*(I - kalman_state.N*predicted_state_σ_lag_1)

    # Backward step
    kalman_state.r = transpose(H[ivar_obs, :])*inv_S*v + transpose(L)*kalman_state.r
    kalman_state.N = transpose(H[ivar_obs, :])*inv_S*H[ivar_obs, :] + transpose(L)*kalman_state.N*L

    # Update smoothed state and covariance
    kalman_state.smoothed_state_μ = predicted_state_μ + predicted_state_σ*kalman_state.r
    kalman_state.smoothed_state_σ = predicted_state_σ - predicted_state_σ*kalman_state.N*predicted_state_σ

end


function save_state_in_smoother_output!(smoother_output::ExtendedKalmanSmootherOutput, smoother_state::ExtendedKalmanSmootherState, t::Int64)

    # Save smoothed state
    smoother_output.smoothed_state[t].μ_t = smoother_state.smoothed_state_μ
    smoother_output.smoothed_state[t].σ_t = smoother_state.smoothed_state_σ
    smoother_output.autocov_state[t] =smoother_state.autocov_state

    # Save matrix values
    smoother_output.N[t] = smoother_state.N
    smoother_output.r[t] = smoother_state.r

end


function initialize_smoother!(smoother_output::ExtendedKalmanSmootherOutput, smoother_state::ExtendedKalmanSmootherState, last_predicted_state)

    # Initialize KalmanSmoother state
    smoother_state.smoothed_state_μ = last_predicted_state.μ_t
    smoother_state.smoothed_state_σ = last_predicted_state.σ_t

    # Save initial predicted state
    smoother_output.smoothed_state[end].μ_t = smoother_state.smoothed_state_μ
    smoother_output.smoothed_state[end].σ_t = smoother_state.smoothed_state_σ

    # Save initial value of r and N
    smoother_output.N[end] = smoother_state.N
    smoother_output.r[end] = smoother_state.r

end