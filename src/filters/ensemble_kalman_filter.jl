mutable struct EnsembleKalmanFilterState{Z <: Real} <: AbstractFilterState{Z}

    # Filtered and predicted state
    predicted_particles_swarm::Matrix{Z}
    filtered_particles_swarm::Matrix{Z}
    observed_particles_swarm::Matrix{Z}

    # Matrices and vectors

    # Likelihood
    llk::Z

    function EnsembleKalmanFilterState(
            init_state::GaussianStateStochasticProcess{Z},
            n_X::Int,
            n_Y::Int,
            n_particles::Int
    ) where {Z <: Real}
        predicted_particles_swarm = rand(
            MvNormal(init_state.μ_t, init_state.Σ_t), n_particles)

        new{Z}(
            predicted_particles_swarm,
            zeros(Z, n_X, n_particles),
            zeros(Z, n_Y, n_particles),
            Z(0.0)
        )
    end

    function EnsembleKalmanFilterState(
            init_state::ParticleSwarmState{Z},
            n_X::Int,
            n_Y::Int,
            n_particles::Int
    ) where {Z <: Real}
        n_particles_init_state = size(init_state.particles_state, 2)
        if n_particles_init_state != n_particles
            @warn "The number of particles of the filter is different from the number of particles of the current state."

            μ_t = vcat(mean(init_state.particles_state, dims = 2)...)
            σ_t = var(init_state.particles_state, dims = 2)

            predicted_particles_swarm = rand(MvNormal(μ_t, σ_t), n_particles)

        else
            predicted_particles_swarm = init_state.particles_state
        end

        new{Z}(
            predicted_particles_swarm,
            zeros(Z, n_X, n_particles),
            zeros(Z, n_Y, n_particles),
            Z(0.0)
        )
    end
end

mutable struct EnsembleKalmanFilter{Z <: Real} <: AbstractStochasticMonteCarloFilter{Z}
    init_state_x::AbstractState{Z}
    state::EnsembleKalmanFilterState{Z}
    n_particles::Int64
    positive::Bool

    function EnsembleKalmanFilter(
            init_state::AbstractState{Z}, n_X, n_Y, n_particles, positive) where {Z <: Real}
        new{Z}(
            init_state,
            EnsembleKalmanFilterState(init_state, n_X, n_Y, n_particles),
            n_particles,
            positive
        )
    end

    function EnsembleKalmanFilter(
            model::ForecastingModel{Z};
            n_particles = 30,
            positive = false
    ) where {Z <: Real}
        return EnsembleKalmanFilter(
            model.current_state,
            model.system.n_X,
            model.system.n_Y,
            n_particles,
            positive
        )
    end
end

mutable struct EnsembleKalmanFilterOutput{Z <: Real} <:
               AbstractStochasticMonteCarloFilterOutput{Z}

    # Predicted, filtered and observed states
    predicted_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}
    filtered_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}
    observed_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}

    llk::Z

    function EnsembleKalmanFilterOutput(
            model::ForecastingModel{Z}, y_t, n_particles) where {Z <: Real}
        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt) * (t - 1)
                   for t in 1:(n_obs + 1)]

        predicted_particles_swarm = TimeSeries{Z, ParticleSwarmState{Z}}(
            t_index,
            n_obs + 1,
            model.system.n_X;
            n_particles = n_particles
        )
        filtered_particles_swarm = TimeSeries{Z, ParticleSwarmState{Z}}(
            t_index[1:(end - 1)],
            n_obs,
            model.system.n_X;
            n_particles = n_particles
        )
        observed_particles_swarm = TimeSeries{Z, ParticleSwarmState{Z}}(
            t_index,
            n_obs,
            model.system.n_Y;
            n_particles = n_particles
        )

        return new{Z}(
            predicted_particles_swarm,
            filtered_particles_swarm,
            observed_particles_swarm,
            Z(0.0)
        )
    end
end

function get_filter_output(filter::EnsembleKalmanFilter, model::ForecastingModel, observation_data)
    return EnsembleKalmanFilterOutput(model, observation_data, filter.n_particles)
end

# TODO : standardize with all AbstractStochasticMonteCarloFilterOutput
function get_last_state(filter_output::EnsembleKalmanFilterOutput)
    return filter_output.predicted_particles_swarm[end]
end

function filtering!(
        filter_output::EnsembleKalmanFilterOutput{Z},
        sys::S,
        filter_method::EnsembleKalmanFilter{Z},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        parameters::Vector{Z}
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z}}
    n_obs = size(observation_data, 1)

    # Save initial state
    save_initial_state_in_filter_output!(filter_output, filter_method.state)

    t_step_table = collect(range(
        filter_method.init_state_x.t, length = n_obs, step = sys.dt))

    n_particles = filter_method.n_particles
    T_M = (Matrix(I, n_particles, n_particles) .- 1 / n_particles)

    @inbounds for (t, t_step) in enumerate(t_step_table)

        # Get current noise matrix R, Q
        ex = exogenous_data[t, :]
        R = sys.R_t(ex, parameters, t_step)
        Q = sys.Q_t(ex, parameters, t_step)

        # Define actual transition and observation operators
        @inline M(x) = transition(sys, x, ex, control_data[t, :], parameters, t_step)
        @inline H(x) = observation(sys, x, ex, parameters, t_step)

        update_filter_state!(
            filter_method.state,
            view(observation_data, t, :),
            M,
            H,
            R,
            Q,
            T_M,
            n_particles,
            filter_method.positive
        )

        save_state_in_filter_output!(filter_output, filter_method.state, t)
    end

    return filter_output
end

# KF for optimization : i don't even know if this one is working
# function filter!(
#         sys::StateSpaceSystem,
#         filter::EnsembleKalmanFilter,
#         y_t,
#         exogenous_variables,
#         control_variables,
#         parameters
# )
#     n_obs = size(y_t, 1)

#     @inbounds for t in 1:n_obs

#         # Get current t_step
#         t_step = filter.init_state_x.t + (t - 1) * sys.dt

#         R = sys.R_t(exogenous_variables[t, :], parameters, t_step)
#         Q = sys.Q_t(exogenous_variables[t, :], parameters, t_step)

#         # Define actual transition and observation operators
#         function M(x)
#             return transition(
#                 sys,
#                 x,
#                 exogenous_variables[t, :],
#                 control_variables[t, :],
#                 parameters,
#                 t_step
#             )
#         end

#         function H(x)
#             return observation(sys, x, exogenous_variables[t, :], parameters, t_step)
#         end

#         update_filter_state!(
#             filter.state,
#             y_t[t, :],
#             M,
#             H,
#             R,
#             Q,
#             filter.n_particles,
#             filter.positive
#         )
#     end

#     return filter.state.llk / n_obs
# end

function update_filter_state!(
        filter_state::EnsembleKalmanFilterState{Z},
        y,
        M,
        H,
        R,
        Q,
        T_M,
        n_particles,
        positive
) where {Z <: Real}

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Observation equation
    filter_state.observed_particles_swarm = H(filter_state.predicted_particles_swarm)[
        ivar_obs, :]

    # Approximation of Var(H(x)) and Cov(x, v)
    if !isempty(ivar_obs)
        ef_states = filter_state.predicted_particles_swarm * T_M
        ef_obs = filter_state.observed_particles_swarm * T_M
        HPHt = (ef_obs * ef_obs') ./ (n_particles - 1) # Approximation of Var(H(x))
        PHt = ((ef_states * ef_obs') ./ (n_particles - 1)) # Approximation of Cov(x, v)

        # Compute innovations and stuff for predicted and filtered states
        S = HPHt + Q[ivar_obs, ivar_obs]
        inv_S = inv(S)
        K = PHt * inv_S
        v = y[ivar_obs] .- filter_state.observed_particles_swarm +
            rand(MvNormal(Q), n_particles)[ivar_obs, :]

        # Update prediction
        filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm +
                                                K * v

    else
        filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm
    end

    # Forecast step
    if positive
        filter_state.predicted_particles_swarm = max.(
            M(filter_state.filtered_particles_swarm) + rand(MvNormal(R), n_particles),
            POSITIVE_PRECISION
        )
    else
        filter_state.predicted_particles_swarm = M(filter_state.filtered_particles_swarm) +
                                                 rand(MvNormal(R), n_particles)
    end

    # Update llk
    if !isempty(ivar_obs)
        ṽ = y[ivar_obs] - vcat(mean(filter_state.observed_particles_swarm, dims = 2)...)
        filter_state.llk += -log(2 * pi) / 2 - (1 / 2) * (logdet(S) + ṽ' * inv_S * ṽ)
    end
end

function save_state_in_filter_output!(
        filter_output::EnsembleKalmanFilterOutput{Z},
        filter_state::EnsembleKalmanFilterState{Z},
        t::Int
) where {Z <: Real}

    # Save predicted state
    filter_output.predicted_particles_swarm[t + 1].particles_state = filter_state.predicted_particles_swarm

    # Save filtered and observed particles swarm
    filter_output.filtered_particles_swarm[t].particles_state = filter_state.filtered_particles_swarm
    filter_output.observed_particles_swarm[t].particles_state = filter_state.observed_particles_swarm

    # Save likelihood
    filter_output.llk = filter_state.llk
end

function save_initial_state_in_filter_output!(
        filter_output::EnsembleKalmanFilterOutput{Z},
        filter_state::EnsembleKalmanFilterState{Z}
) where {Z <: Real}

    # Save initial predicted state
    filter_output.predicted_particles_swarm[1].particles_state = filter_state.predicted_particles_swarm
end
