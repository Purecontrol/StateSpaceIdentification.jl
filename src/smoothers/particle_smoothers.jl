#############################################################################
############################# Ancestor Tracking #############################
#############################################################################

mutable struct AncestorTrackingState{Z <: Real} <: AbstractSmootherState{Z}

    # Filtered and predicted state
    smoothed_particles_swarm::Matrix{Z}
    ind_smoothing::Vector{Int64}

    function AncestorTrackingState{Z}(n_X, n_Y, n_particles) where {Z <: Real}
        new{Z}(zeros(Z, n_X, n_particles), zeros(Z, n_particles))
    end
end

mutable struct AncestorTrackingSmoother{Z <: Real} <: AbstractStochasticMonteCarloSmoother{Z}
    state::AncestorTrackingState{Z}
    n_particles::Int64

    function AncestorTrackingSmoother{Z}(n_X, n_Y, n_particles) where {Z <: Real}
        new{Z}(AncestorTrackingState{Z}(n_X, n_Y, n_particles), n_particles)
    end

    function AncestorTrackingSmoother(model::ForecastingModel{Z}; n_particles = 30)  where {Z <: Real}
        return AncestorTrackingSmoother{Z}(model.system.n_X, model.system.n_Y, n_particles)
    end
end

mutable struct AncestorTrackingSmootherOutput{Z <: Real} <: AbstractStochasticMonteCarloSmootherOutput{Z}
    smoothed_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}

    function AncestorTrackingSmootherOutput(model::ForecastingModel{Z}, y_t, n_particles) where {Z <: Real}
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

function get_smoother_output(smoother::AncestorTrackingSmoother, model::ForecastingModel, y_t)
    return AncestorTrackingSmootherOutput(model, y_t, smoother.n_particles)
end

function smoothing!(
        smoother_output::AncestorTrackingSmootherOutput{Z},
        filter_output::ParticleFilterOutput{Z},
        sys::S,
        smoother_method::AncestorTrackingSmoother{Z},
        observation_data,
        exogenous_data,
        control_data,
        parameters
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z}}
    n_obs = size(observation_data, 1)
    n_filtering = size(filter_output.sampling_weights, 2)

    initialize_smoother!(
        smoother_output,
        smoother_method.state,
        filter_output.predicted_particles_swarm[end],
        smoother_method.n_particles,
        n_filtering
    )

    # Backward recursions
    @inbounds for t in (n_obs):-1:1

        predicted_particle_swarm = filter_output.predicted_particles_swarm[t].particles_state
        I_t = filter_output.ancestor_indices[t, :]

        update_smoother_state!(smoother_method.state, predicted_particle_swarm, I_t)

        save_state_in_smoother_output!(smoother_output, smoother_method.state, t)
    end

    return smoother_output
end

function update_smoother_state!(smoother_state::AncestorTrackingState{Z}, Xp, I_t) where {Z <: Real}
    smoother_state.ind_smoothing = I_t[smoother_state.ind_smoothing]
    smoother_state.smoothed_particles_swarm = Xp[:, smoother_state.ind_smoothing]
end

function save_state_in_smoother_output!(
        smoother_output::AncestorTrackingSmootherOutput{Z},
        smoother_state::AncestorTrackingState{Z},
        t::Int
) where {Z <: Real}

    # Save smoothed state
    smoother_output.smoothed_particles_swarm[t].particles_state = smoother_state.smoothed_particles_swarm
end

function initialize_smoother!(
        smoother_output::AncestorTrackingSmootherOutput,
        smoother_state::AncestorTrackingState,
        last_predicted_state,
        n_smoothing,
        n_filtering
)
    smoother_state.ind_smoothing = sample_discrete(
        (1 / n_filtering) .* ones(n_filtering), n_smoothing)[1, :]

    # Initialize KalmanSmoother state
    # smoother_state.smoothed_particles_swarm .= last_predicted_state.particles_state[:, ind_smoothing']
    smoother_state.smoothed_particles_swarm = last_predicted_state.particles_state[
        :, smoother_state.ind_smoothing]

    # Save initial predicted state
    smoother_output.smoothed_particles_swarm[end].particles_state = smoother_state.smoothed_particles_swarm
end

#############################################################################
####################### Backward Simulation Smoother ########################
#############################################################################

mutable struct BackwardSimulationState{Z <: Real} <: AbstractSmootherState{Z}

    # Filtered and predicted state
    smoothed_particles_swarm::Matrix{Z}

    function BackwardSimulationState{Z}(n_X, n_Y, n_particles) where {Z <: Real}
        new{Z}(zeros(Z, n_X, n_particles))
    end
end

mutable struct BackwardSimulationSmoother{Z <: Real} <: AbstractStochasticMonteCarloSmoother{Z}
    state::BackwardSimulationState{Z}
    n_particles::Int

    function BackwardSimulationSmoother{Z}(n_X, n_Y, n_particles) where {Z <: Real}
        new{Z}(BackwardSimulationState{Z}(n_X, n_Y, n_particles), n_particles)
    end

    function BackwardSimulationSmoother(model::ForecastingModel{Z}; n_particles = 30) where {Z <: Real}
        return BackwardSimulationSmoother{Z}(model.system.n_X, model.system.n_Y, n_particles)
    end
end

mutable struct BackwardSimulationSmootherOutput{Z <: Real} <: AbstractStochasticMonteCarloSmootherOutput{Z}
    smoothed_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}

    function BackwardSimulationSmootherOutput(model::ForecastingModel{Z}, y_t, n_particles) where {Z <: Real}
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

function get_smoother_output(smoother_method::BackwardSimulationSmoother, model::ForecastingModel, y_t)
    return BackwardSimulationSmootherOutput(model, y_t, smoother_method.n_particles)
end

function smoothing!(
        smoother_output::BackwardSimulationSmootherOutput{Z},
        filter_output::ParticleFilterOutput{Z},
        sys::S,
        smoother_method::BackwardSimulationSmoother{Z},
        observation_data,
        exogenous_data,
        control_data,
        parameters
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z}}
    n_obs = size(observation_data, 1)
    n_filtering = size(filter_output.sampling_weights, 2)

    initialize_smoother!(
        smoother_output,
        smoother_method.state,
        filter_output.predicted_particles_swarm[end],
        filter_output.sampling_weights[end, :],
        smoother_method.n_particles,
        n_filtering
    )

    t_step_table = collect(range(
        filter_output.predicted_particles_swarm[1].t, length = n_obs, step = sys.dt))

    # Backward recursions
    @inbounds for (t, t_step) in Iterators.reverse(enumerate(t_step_table))

        predicted_particle_swarm = filter_output.predicted_particles_swarm[t].particles_state
        predicted_particles_swarm_mean = filter_output.predicted_particles_swarm_mean[
            t + 1, :, :]
        sampling_weights = filter_output.sampling_weights[t + 1, :]

        σ = inv(sys.R_t(exogenous_data[t, :], parameters, t_step))
        # σ = sys.R_t(exogenous_variables[t, :], parameters, t_step)

        update_smoother_state!(
            smoother_method.state,
            predicted_particle_swarm,
            predicted_particles_swarm_mean,
            sampling_weights,
            σ,
            smoother_method.n_particles,
            n_filtering,
            sys.n_X
        )

        save_state_in_smoother_output!(smoother_output, smoother_method.state, t)
    end

    return smoother_output
end

function update_smoother_state!(
        smoother_state::BackwardSimulationState{Z},
        Xp,
        Xp_mean,
        W,
        σ,
        n_smoothing,
        n_filtering,
        n_X
) where {Z <: Real}
    v = smoother_state.smoothed_particles_swarm[:, :, [CartesianIndex()]] .-
        Xp_mean[:, [CartesianIndex()], :]
    smoothing_weights = zeros(n_smoothing, n_filtering)
    @inbounds for i in 1:n_X
        @inbounds for j in 1:n_X
            smoothing_weights += -0.5 * v[i, :, :] * σ[i, j] .* v[j, :, :]
        end
    end
    logmax = maximum(smoothing_weights, dims = 2)
    smoothing_weights = exp.(smoothing_weights .- logmax) .* W[[CartesianIndex()], :]
    smoothing_weights ./= sum(smoothing_weights, dims = 2)
    # for i in 1:n_smoothing
    #     innov = repeat(smoother_state.smoothed_particles_swarm[:, i], 1, n_filtering) - Xp_mean[:, :]
    #     logmax = max((-0.5 * sum(innov' * σ .* innov', dims=2))...)
    #     wei_res = vcat(exp.(-0.5 * sum(innov' * σ .* innov', dims=2)  .- logmax)...).*W
    #     smoothing_weights[i, :] = wei_res/sum(wei_res)
    # end

    # for i in 1:n_smoothing
    #     wei_res = zeros(n_filtering)
    #     for j in 1:n_filtering
    #         μ = vec(Xp_mean[:, j])
    #         d = MvNormal(μ, σ)
    #         wei_res[j] = pdf(d, smoother_state.smoothed_particles_swarm[:, i])
    #     end
    #     wei_res = wei_res.*W
    #     smoothing_weights[i, :] = wei_res/sum(wei_res)
    # end

    ind_smoothing = sample_discrete(smoothing_weights', 1, n_exp = n_smoothing)[:, 1]
    # ind_smoothing = zeros(Int, n_smoothing)
    # for i in 1:n_smoothing
    #     # ind_smoothing[i] = Int(sample_discrete(smoothing_weights[i, :], 1)[1, 1])
    #     ind_smoothing[i] = Int(rand(Categorical(smoothing_weights[i, :])))
    # end

    # smoother_state.smoothed_particles_swarm .= Xp[:, ind_smoothing]
    smoother_state.smoothed_particles_swarm = Xp[:, ind_smoothing]
end

function save_state_in_smoother_output!(
        smoother_output::BackwardSimulationSmootherOutput{Z},
        smoother_state::BackwardSimulationState{Z},
        t::Int
) where {Z <: Real}

    # Save smoothed state
    smoother_output.smoothed_particles_swarm[t].particles_state = smoother_state.smoothed_particles_swarm
end

function initialize_smoother!(
        smoother_output::BackwardSimulationSmootherOutput{Z},
        smoother_state::BackwardSimulationState{Z},
        last_predicted_state,
        last_sampling_weights,
        n_smoothing,
        n_filtering
) where {Z <: Real}
    ind_smoothing = sample_discrete((1 / n_filtering) .* ones(n_filtering), n_smoothing)[
        1, :]
    # ind_smoothing = sample_discrete(last_sampling_weights, n_smoothing)[1, :]

    # Initialize KalmanSmoother state
    # smoother_state.smoothed_particles_swarm .= last_predicted_state.particles_state[:, ind_smoothing']
    smoother_state.smoothed_particles_swarm = last_predicted_state.particles_state[
        :, ind_smoothing]

    # Save initial predicted state
    smoother_output.smoothed_particles_swarm[end].particles_state = smoother_state.smoothed_particles_swarm
end
