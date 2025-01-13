mutable struct EnsembleKalmanSmootherState{Z <: Real} <: AbstractSmootherState{Z}

    # Filtered and predicted state
    smoothed_particles_swarm::Matrix{Z}

    function EnsembleKalmanSmootherState{Z}(n_X, n_particles) where {Z <: Real}
        new{Z}(zeros(Z, n_X, n_particles))
    end
end

mutable struct EnsembleKalmanSmoother{Z <: Real} <: AbstractStochasticMonteCarloSmoother{Z}
    state::EnsembleKalmanSmootherState{Z}
    n_particles::Int

    function EnsembleKalmanSmoother{Z}(n_X, n_particles) where {Z <: Real}
        new{Z}(EnsembleKalmanSmootherState{Z}(n_X, n_particles), n_particles)
    end

    function EnsembleKalmanSmoother(
            model::ForecastingModel{Z}; n_particles = 30) where {Z <: Real}
        return EnsembleKalmanSmoother{Z}(model.system.n_X, n_particles)
    end
end

mutable struct EnsembleKalmanSmootherOutput{Z <: Real} <:
               AbstractStochasticMonteCarloSmootherOutput{Z}
    smoothed_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}

    function EnsembleKalmanSmootherOutput(
            model::ForecastingModel{Z}, y_t, n_particles) where {Z <: Real}
        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt) * (t - 1)
                   for t in 1:(n_obs + 1)]

        smoothed_particles_swarm = TimeSeries{Z, ParticleSwarmState{Z}}(
            t_index,
            n_obs + 1,
            model.system.n_X;
            n_particles = n_particles
        )

        return new{Z}(smoothed_particles_swarm)
    end
end

function get_smoother_output(
        smoother_method::EnsembleKalmanSmoother, model::ForecastingModel, observation_data)
    return EnsembleKalmanSmootherOutput(
        model, observation_data, smoother_method.n_particles)
end

function smoothing!(
        smoother_output::EnsembleKalmanSmootherOutput{Z},
        filter_output::EnsembleKalmanFilterOutput{Z},
        sys::S,
        smoother_method::EnsembleKalmanSmoother,
        observation_data,
        exogenous_data,
        control_data,
        parameters
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z}}
    n_obs = size(observation_data, 1)

    initialize_smoother!(
        smoother_output,
        smoother_method.state,
        filter_output.predicted_particles_swarm[end]
    )

    t_step_table = collect(range(
        filter_output.predicted_particles_swarm[1].t, length = n_obs, step = sys.dt))

    # Backward recursions
    @inbounds for (t, t_step) in Iterators.reverse(enumerate(t_step_table))

        update_smoother_state!(smoother_method.state,
            filter_output.predicted_particles_swarm[t + 1].particles_state,
            filter_output.filtered_particles_swarm[t].particles_state)

        save_state_in_smoother_output!(smoother_output, smoother_method.state, t)
    end

    return smoother_output
end

function update_smoother_state!(smoother_state::EnsembleKalmanSmootherState{Z}, Xp, Xf) where {Z <: Real}
    Paf = cov(Xf, Xp, dims = 2)
    Pff = cov(Xp, Xp, dims = 2)
    K = Paf * pinv(Pff)

    smoother_state.smoothed_particles_swarm = Xf +
                                              K *
                                              (smoother_state.smoothed_particles_swarm - Xp)
end

function save_state_in_smoother_output!(
        smoother_output::EnsembleKalmanSmootherOutput{Z},
        smoother_state::EnsembleKalmanSmootherState{Z},
        t::Int
) where {Z <: Real}

    # Save smoothed state
    smoother_output.smoothed_particles_swarm[t].particles_state = smoother_state.smoothed_particles_swarm
end

function initialize_smoother!(
        smoother_output::EnsembleKalmanSmootherOutput{Z},
        smoother_state::EnsembleKalmanSmootherState{Z},
        last_predicted_state
) where {Z <: Real}

    # Initialize KalmanSmoother state
    smoother_state.smoothed_particles_swarm = last_predicted_state.particles_state

    # Save initial predicted state
    smoother_output.smoothed_particles_swarm[end].particles_state = last_predicted_state.particles_state
end
