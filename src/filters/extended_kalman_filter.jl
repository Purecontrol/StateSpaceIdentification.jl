const ExtendedKalmanFilterState{Z} = KalmanFilterState{Z} where {Z <: Real}

mutable struct ExtendedKalmanFilter{Z <: Real} <: AbstractDeterministicFilter{Z}
    init_state_x::GaussianStateStochasticProcess{Z}
    state::ExtendedKalmanFilterState{Z}
    #TODO : autodiff dM_t, dH_t
    dM_t::TransitionNonLinearProvider{Z}
    dH_t::ObservationNonLinearProvider{Z}

    """
    Constructor with gaussian init state.
    """
    function ExtendedKalmanFilter(init_state::GaussianStateStochasticProcess{Z},
            n_X, n_Y, dM_t, dH_t) where {Z <: Real}
        new{Z}(init_state, ExtendedKalmanFilterState(init_state, n_X, n_Y), dM_t, dH_t)
    end

    """Constructor with given type K."""
    function ExtendedKalmanFilter{K}(init_state::GaussianStateStochasticProcess,
            n_X, n_Y, dM_t, dH_t) where {K <: Real}
        new{K}(init_state, ExtendedKalmanFilterState{K}(init_state, n_X, n_Y), dM_t, dH_t)
    end

    """
    Constructor with particle init state.
    """
    function ExtendedKalmanFilter(
            init_state::ParticleSwarmState{Z}, n_X, n_Y, dM_t, dH_t) where {Z <: Real}
        μ_t = vcat(mean(init_state.particles_state, dims = 2)...)
        Σ_t = var(init_state.particles_state, dims = 2)
        gaussian_init_state = GaussianStateStochasticProcess(init_state.t, μ_t, Σ_t)

        new{Z}(gaussian_init_state,
            ExtendedKalmanFilterState(gaussian_init_state, n_X, n_Y), dM_t, dH_t)
    end

    function ExtendedKalmanFilter(model::ForecastingModel{Z, T, S},
        dM_t,
        dH_t;
        type = Z) where {
        Z <: Real, T <: AbstractStateSpaceSystem, S <: AbstractState{Z}}
        
        dM_t = isa(dM_t, TransitionNonLinearProvider) ? dM_t : TransitionNonLinearProvider{Z}(dM_t)
        dH_t = isa(dH_t, ObservationNonLinearProvider) ? dH_t : ObservationNonLinearProvider{Z}(dH_t)
        
        return ExtendedKalmanFilter{type}(model.current_state, model.system.n_X, model.system.n_Y, dM_t, dH_t)
    end
end

const ExtendedKalmanFilterOutput{Z} = KalmanFilterOutput{Z} where {Z <: Real}

function get_filter_output(
    filter_method::ExtendedKalmanFilter, model::ForecastingModel, observation_data)
    return ExtendedKalmanFilterOutput(model, observation_data)
end

# function get_last_state(filter_output::ExtendedKalmanFilterOutput)
#     return filter_output.predicted_state[end]
# end

function filtering!(
        filter_output::ExtendedKalmanFilterOutput{Z},
        sys::S,
        filter_method::ExtendedKalmanFilter{Z},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        parameters::Vector{Z},
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z}}

    n_obs = size(observation_data, 1)

    # Save initial state
    save_initial_state_in_filter_output!(filter_output, filter_method.state)

    t_step_table = collect(range(
        filter_method.init_state_x.t, length = n_obs, step = sys.dt))

    @inbounds for (t, t_step) in enumerate(t_step_table)

        # Get current noise matrix R, Q
        ex = exogenous_data[t, :]
        R = sys.R_t(ex, parameters, t_step)
        Q = sys.Q_t(ex, parameters, t_step)

        # Define actual transition and observation operators
        @inline M(x) = transition(sys, x, ex, control_data[t, :], parameters, t_step)
        @inline H(x) = observation(sys, x, ex, parameters, t_step)

        # Define actual transition and observation derivative operators
        @inline dM(x) = filter_method.dM_t(x, ex, control_data[t, :], parameters, t_step)
        @inline dH(x) = filter_method.dH_t(x, ex, parameters, t_step)

        update_filter_state!(
            filter_method.state,
            view(observation_data, t, :),
            M,
            H,
            R,
            Q,
            dM,
            dH
        )

        save_state_in_filter_output!(filter_output, filter_method.state, t)
    end

    return filter_output
end

function update_filter_state!(
        kalman_state::ExtendedKalmanFilterState{Z},
        y,
        M,
        H,
        R,
        Q,
        dM,
        dH
) where {Z <: Real}

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Compute innovations and stuff for predicted and filtered states
    jacH = dH(kalman_state.predicted_state_μ)
    kalman_state.v = y[ivar_obs] - H(kalman_state.predicted_state_μ)[ivar_obs]
    kalman_state.S = jacH[ivar_obs, :] * kalman_state.predicted_state_Σ *
                     transpose(jacH[ivar_obs, :]) +
                     Q[ivar_obs, ivar_obs]
    kalman_state.M = kalman_state.predicted_state_Σ * transpose(jacH[ivar_obs, :])
    inv_S = inv(kalman_state.S)

    # Update states (Update step)
    kalman_state.filtered_state_μ = kalman_state.predicted_state_μ +
                                    kalman_state.M * inv_S * kalman_state.v
    kalman_state.filtered_state_Σ = kalman_state.predicted_state_Σ -
                                    kalman_state.M * inv_S * transpose(kalman_state.M)

    # Forecast step
    kalman_state.predicted_state_μ = M(kalman_state.filtered_state_μ)
    F = dM(kalman_state.filtered_state_μ)
    kalman_state.predicted_state_Σ = transpose(F) * kalman_state.filtered_state_Σ * F + R

    # Compute stuff for Kalman smoother
    kalman_state.K = F * kalman_state.M * inv_S #to check
    kalman_state.L = F - kalman_state.K * jacH[ivar_obs, :] #to check

    # Update likelihood
    if !isempty(ivar_obs)
        kalman_state.llk += -log(2 * pi) / 2 -
                            (1 / 2) * (log(det(kalman_state.S)) +
                             kalman_state.v' * inv_S * kalman_state.v)
    end
end

# function save_state_in_filter_output!(
#         filter_output::ExtendedKalmanFilterOutput{Z},
#         filter_state::ExtendedKalmanFilterState{Z},
#         t::Int
# ) where {Z <: Real}

#     # Save predicted state
#     filter_output.predicted_state[t + 1].μ_t = filter_state.predicted_state_μ
#     filter_output.predicted_state[t + 1].σ_t = filter_state.predicted_state_σ

#     # Save filtered state
#     filter_output.filtered_state[t].μ_t = filter_state.filtered_state_μ
#     filter_output.filtered_state[t].σ_t = filter_state.filtered_state_σ

#     # Save matrix values
#     filter_output.K[t] = filter_state.K
#     filter_output.M[t] = filter_state.M
#     filter_output.L[t] = filter_state.L
#     filter_output.S[t] = filter_state.S
#     filter_output.v[t] = filter_state.v

#     # Save likelihood
#     filter_output.llk = filter_state.llk
# end

# function save_initial_state_in_filter_output!(
#         filter_output::ExtendedKalmanFilterOutput{Z},
#         filter_state::ExtendedKalmanFilterState{Z}
# ) where {Z <: Real}

#     # Save initial predicted state
#     filter_output.predicted_state[1].μ_t = filter_state.predicted_state_μ
#     filter_output.predicted_state[1].Σ_t = filter_state.predicted_state_Σ
# end
