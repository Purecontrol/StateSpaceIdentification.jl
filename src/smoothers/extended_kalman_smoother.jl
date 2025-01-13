const ExtendedKalmanSmootherState{Z} = KalmanSmootherState{Z} where {Z <: Real}

mutable struct ExtendedKalmanSmoother{Z <: Real} <: AbstractDeterministicSmoother{Z}
    state::ExtendedKalmanSmootherState{Z}
    dH_t::ObservationNonLinearProvider{Z}

    function ExtendedKalmanSmoother{Z}(n_X, n_Y, dH_t) where {Z <: Real}
        new{Z}(ExtendedKalmanSmootherState{Z}(n_X, n_Y), dH_t)
    end

    function ExtendedKalmanSmoother(model::ForecastingModel{Z, T, S}, dH_t) where {Z <: Real, T <: AbstractStateSpaceSystem, S <: AbstractState{Z}}
        
        dH_t = isa(dH_t, ObservationNonLinearProvider) ? dH_t : ObservationNonLinearProvider{Z}(dH_t)
        
        return ExtendedKalmanSmoother{Z}(model.system.n_X, model.system.n_Y, dH_t)
    end
end

const ExtendedKalmanSmootherOutput{Z} = KalmanSmootherOutput{Z} where {Z <: Real}

function get_smoother_output(smoother::ExtendedKalmanSmoother, model::ForecastingModel, observation_data)
    return ExtendedKalmanSmootherOutput(model, observation_data)
end

# General KF
function smoothing!(
        smoother_output::ExtendedKalmanSmootherOutput{Z},
        filter_output::ExtendedKalmanFilterOutput{Z},
        sys::S,
        smoother_method::ExtendedKalmanSmoother{Z},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        parameters::Vector{Z}
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z}}
    n_obs = size(observation_data, 1)   

    initialize_smoother!(
        smoother_output,
        smoother_method.state,
        filter_output.predicted_state[end]
    )

    t_step_table = collect(range(
        filter_output.predicted_state[1].t, length = n_obs, step = sys.dt))

    # Backward recursions
    @inbounds for (t, t_step) in Iterators.reverse(enumerate(t_step_table))

        # Get current matrix H (pseudo-matrix H if the system is linear in x) # to check if predicted state is correct here.
        H = smoother_method.dH_t(filter_output.predicted_state[t].μ_t, exogenous_data[t, :], parameters, t_step)

        inv_S = inv(filter_output.S[t])
        v = filter_output.v[t]
        L = filter_output.L[t]
        predicted_state_μ = filter_output.predicted_state[t].μ_t
        predicted_state_Σ = filter_output.predicted_state[t].Σ_t
        predicted_state_Σ_lag_1 = filter_output.predicted_state[t + 1].Σ_t

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
        kalman_state,
        y,
        H,
        inv_S,
        v,
        L,
        predicted_state_μ,
        predicted_state_Σ,
        predicted_state_Σ_lag_1
)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Update autocovariance
    kalman_state.autocov_state = predicted_state_Σ * transpose(L) *
                                 (I - kalman_state.N * predicted_state_Σ_lag_1)

    # Backward step
    kalman_state.r = transpose(H[ivar_obs, :]) * inv_S * v + transpose(L) * kalman_state.r
    kalman_state.N = transpose(H[ivar_obs, :]) * inv_S * H[ivar_obs, :] +
                     transpose(L) * kalman_state.N * L

    # Update smoothed state and covariance
    kalman_state.smoothed_state_μ = predicted_state_μ + predicted_state_Σ * kalman_state.r
    kalman_state.smoothed_state_Σ = predicted_state_Σ -
                                    predicted_state_Σ * kalman_state.N * predicted_state_Σ
end

# function save_state_in_smoother_output!(
#         smoother_output::ExtendedKalmanSmootherOutput,
#         smoother_state::ExtendedKalmanSmootherState,
#         t::Int64
# )

#     # Save smoothed state
#     smoother_output.smoothed_state[t].μ_t = smoother_state.smoothed_state_μ
#     smoother_output.smoothed_state[t].σ_t = smoother_state.smoothed_state_σ
#     smoother_output.autocov_state[t] = smoother_state.autocov_state

#     # Save matrix values
#     smoother_output.N[t] = smoother_state.N
#     smoother_output.r[t] = smoother_state.r
# end

# function initialize_smoother!(
#         smoother_output::ExtendedKalmanSmootherOutput,
#         smoother_state::ExtendedKalmanSmootherState,
#         last_predicted_state
# )

#     # Initialize KalmanSmoother state
#     smoother_state.smoothed_state_μ = last_predicted_state.μ_t
#     smoother_state.smoothed_state_σ = last_predicted_state.σ_t

#     # Save initial predicted state
#     smoother_output.smoothed_state[end].μ_t = smoother_state.smoothed_state_μ
#     smoother_output.smoothed_state[end].σ_t = smoother_state.smoothed_state_σ

#     # Save initial value of r and N
#     smoother_output.N[end] = smoother_state.N
#     smoother_output.r[end] = smoother_state.r
# end
