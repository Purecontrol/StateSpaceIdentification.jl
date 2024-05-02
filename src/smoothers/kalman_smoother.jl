using LinearAlgebra


mutable struct KalmanSmootherState

    # Filtered and predicted state
    smoothed_state_μ#::Vector{Float64}
    smoothed_state_σ#::Matrix{Float64}
    autocov_state

    # Matrices and vectors
    r#::Matrix{Float64}
    N#::Matrix{Float64}


    function KalmanSmootherState(n_X, n_Y)
        
        new(zeros(n_X), zeros(n_X, n_X), zeros(n_X, n_X), zeros(n_X), zeros(n_X, n_X))

    end

end


mutable struct KalmanSmoother <: AbstractSmoother

    smoother_state::KalmanSmootherState

    function KalmanSmoother(n_X, n_Y)

        new(KalmanSmootherState(n_X, n_Y))

    end


    function KalmanSmoother(model::ForecastingModel)

        new(KalmanSmootherState(model.system.n_X, model.system.n_Y))

    end

end


mutable struct KalmanSmootherOutput <: SmootherOutput

    # Predicted and filtered states
    smoothed_state::TimeSeries{GaussianStateStochasticProcess}
    autocov_state::Array{Array{Float64, 2}, 1}
  
    r::Array{Array{Float64, 1}, 1}
    N::Array{Array{Float64, 2}, 1}

    function KalmanSmootherOutput(model::ForecastingModel, y_t)

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


function get_smoother_output(smoother::KalmanSmoother, model, y_t)

    return KalmanSmootherOutput(model, y_t)

end


function type_of_state(kf::KalmanSmoother)

    return GaussianStateStochasticProcess

end


# General KF
function smoother!(smoother_output::KalmanSmootherOutput, filter_output::KalmanFilterOutput, sys::GaussianLinearStateSpaceSystem, smoother::KalmanSmoother, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    initialize_smoother!(smoother_output, smoother.smoother_state, filter_output.predicted_state[end])

    # Backward recursions
    @inbounds for t in (n_obs):-1:1

        # Get current t_step
        t_step = filter_output.predicted_state[1].t + (t-1)*sys.dt

        # Get current matrix H
        H = sys.H_t(exogenous_variables[t, :], parameters, t_step)
        inv_S = inv(filter_output.S[t])
        v = filter_output.v[t]
        L = filter_output.L[t]
        predicted_state_μ = filter_output.predicted_state[t].μ_t
        predicted_state_σ = filter_output.predicted_state[t].σ_t
        predicted_state_σ_lag_1 = filter_output.predicted_state[t+1].σ_t
        update_smoother_state!(smoother.smoother_state, y_t[t, :], H, inv_S, v, L, predicted_state_μ, predicted_state_σ, predicted_state_σ_lag_1)
        
        # x_f = filter_output.predicted_state[t+1].μ_t
        # x_a = filter_output.filtered_state[t].μ_t
        # p_f = filter_output.predicted_state[t+1].σ_t
        # p_a = filter_output.filtered_state[t].σ_t
        # M = sys.A_t(exogenous_variables[t, :], parameters, t_step)
        # update_smoother_state_test!(smoother.smoother_state, x_f, x_a, p_f, p_a, M)


        save_state_in_smoother_output!(smoother_output, smoother.smoother_state, t)

    end

    # @inbounds for t in 1:(n_obs-1)

    #     # Get current t_step
    #     t_step = filter_output.predicted_state[1].t + (t-1)*sys.dt

    #     p_a = filter_output.filtered_state[t].σ_t
    #     p_a_1 = filter_output.filtered_state[t+1].σ_t

    #     M = filter_output.M[t+1]
    #     A_transition = sys.A_t(exogenous_variables[t+1, :], parameters, t_step)
    #     H = sys.H_t(exogenous_variables[t+1, :], parameters, t_step)
    #     inv_S = inv(filter_output.S[t+1])
    #     K_f = M*inv_S
        
    #     A = (I - K_f * H) * A_transition * p_a
    #     B = (smoother_output.smoothed_state[t+1].σ_t - p_a_1) * inv(p_a_1)
    #     smoother_output.autocov_state[t] = A + B * A

    # end
    # H = sys.H_t(exogenous_variables[n_obs, :], parameters, 0)
    # K_f = filter_output.M[end]*inv(filter_output.S[end])
    # A_transition = sys.A_t(exogenous_variables[n_obs, :], parameters, 0)
    # smoother_output.autocov_state[end] = (I - K_f*H)*A_transition*filter_output.filtered_state[end-1].σ_t

    return smoother_output

end

function update_smoother_state_test!(kalman_state, x_f, x_a, p_f, p_a, M)

    K = p_a * M' * inv(p_f)
    kalman_state.smoothed_state_μ = x_a + K*(kalman_state.smoothed_state_μ - x_f)
    kalman_state.smoothed_state_σ = p_a + K*(kalman_state.smoothed_state_σ - p_f)*K'

end


function update_smoother_state!(kalman_state, y, H, inv_S, v, L, predicted_state_μ, predicted_state_σ, predicted_state_σ_lag_1)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Update autocovariance
    kalman_state.autocov_state = (predicted_state_σ*transpose(L)*(I - kalman_state.N*predicted_state_σ_lag_1))'

    # Backward step
    kalman_state.r = transpose(H[ivar_obs, :])*inv_S*v + transpose(L)*kalman_state.r
    kalman_state.N = transpose(H[ivar_obs, :])*inv_S*H[ivar_obs, :] + transpose(L)*kalman_state.N*L

    # Update smoothed state and covariance
    kalman_state.smoothed_state_μ = predicted_state_μ + predicted_state_σ*kalman_state.r
    kalman_state.smoothed_state_σ = predicted_state_σ - predicted_state_σ*kalman_state.N*predicted_state_σ

end


function save_state_in_smoother_output!(smoother_output::KalmanSmootherOutput, smoother_state::KalmanSmootherState, t::Int64)

    # Save smoothed state
    smoother_output.smoothed_state[t].μ_t = smoother_state.smoothed_state_μ
    smoother_output.smoothed_state[t].σ_t = smoother_state.smoothed_state_σ
    smoother_output.autocov_state[t] =smoother_state.autocov_state

    # Save matrix values
    smoother_output.N[t] = smoother_state.N
    smoother_output.r[t] = smoother_state.r

end


function initialize_smoother!(smoother_output::KalmanSmootherOutput, smoother_state::KalmanSmootherState, last_predicted_state)

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
