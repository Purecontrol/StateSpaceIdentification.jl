using LinearAlgebra


mutable struct AncestorTrackingState

    # Filtered and predicted state
    smoothed_particles_swarm
    ind_smoothing

    function AncestorTrackingState(n_X, n_Y, n_particles)
        
        new(zeros(Float64, n_X, n_particles), zeros(Float64, n_X, n_particles))

    end

end


mutable struct AncestorTrackingSmoother <: AbstractSmoother

    n_particles::Int64
    smoother_state::AncestorTrackingState

    function AncestorTrackingSmoother(n_X, n_Y, n_particles)

        new(n_particles, AncestorTrackingState(n_X, n_Y, n_particles))

    end

    function AncestorTrackingSmoother(model::ForecastingModel; n_particles=30)

        new(n_particles, AncestorTrackingState(model.system.n_X, model.system.n_Y, n_particles))

    end

end


mutable struct AncestorTrackingSmootherOutput <: SmootherOutput

    smoothed_state::TimeSeries{ParticleSwarmState}

    function AncestorTrackingSmootherOutput(model::ForecastingModel, y_t, n_particles)

        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]

        smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, model.system.n_X, t_index; n_particles=n_particles)

        return new(smoothed_particles_swarm)

    end

end


function get_smoother_output(smoother::AncestorTrackingSmoother, model, y_t)

    return AncestorTrackingSmootherOutput(model, y_t, smoother.n_particles)

end


function smoother!(smoother_output::AncestorTrackingSmootherOutput, filter_output::ParticleFilterOutput, sys::StateSpaceSystem, smoother::AncestorTrackingSmoother, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)
    n_filtering = size(filter_output.sampling_weights, 2)

    initialize_smoother!(smoother_output, smoother.smoother_state, filter_output.predicted_particles_swarm[end], smoother.n_particles, n_filtering)

    # Backward recursions
    @inbounds for t in (n_obs):-1:1

        predicted_particle_swarm = filter_output.predicted_particles_swarm[t].particles_state
        I_t = filter_output.ancestor_indices[t, :]

        update_smoother_state!(smoother.smoother_state, predicted_particle_swarm, I_t)

        save_state_in_smoother_output!(smoother_output, smoother.smoother_state, t)

    end

    return smoother_output

end


function update_smoother_state!(smoother_state::AncestorTrackingState, Xp, I_t)

    smoother_state.ind_smoothing = I_t[smoother_state.ind_smoothing]
    smoother_state.smoothed_particles_swarm = Xp[:, smoother_state.ind_smoothing]

end


function save_state_in_smoother_output!(smoother_output::AncestorTrackingSmootherOutput, smoother_state::AncestorTrackingState, t::Int64)

    # Save smoothed state
    smoother_output.smoothed_state[t].particles_state = smoother_state.smoothed_particles_swarm

end


function initialize_smoother!(smoother_output::AncestorTrackingSmootherOutput, smoother_state::AncestorTrackingState, last_predicted_state, n_smoothing, n_filtering)

    smoother_state.ind_smoothing = sample_discrete((1/n_filtering).*ones(n_filtering), n_smoothing)[1, :]

    # Initialize KalmanSmoother state
    # smoother_state.smoothed_particles_swarm .= last_predicted_state.particles_state[:, ind_smoothing']
    smoother_state.smoothed_particles_swarm = last_predicted_state.particles_state[:, smoother_state.ind_smoothing]

    # Save initial predicted state
    smoother_output.smoothed_state[end].particles_state = smoother_state.smoothed_particles_swarm

end


# function ancestor_tracking_smoothing(y_t, exogenous_variables, filter_output, model, parameters; n_smoothing=30)

#     n_obs = size(y_t, 1)
#     n_X = model.system.n_X

#     # Create output structure
#     t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]
#     smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, 1, t_index; n_particles=n_smoothing)

#     # Get output filter
#     predicted_particles_swarm = filter_output.predicted_particles_swarm
#     last_sampling_weight = filter_output.sampling_weights[end, :]
#     ancestor_indices = filter_output.ancestor_indices

#     Xs = zeros(Float64, n_obs+1, n_X, n_smoothing)
    
#     ind_smoothing = sample_discrete(last_sampling_weight, n_smoothing)'

#     Xs[end, :, :] .= predicted_particles_swarm[end].particles_state[:, ind_smoothing]
#     smoothed_particles_swarm[end].particles_state = Xs[end, :, :]

#     @inbounds for t in (n_obs):-1:1

#         ind_smoothing = ancestor_indices[t, ind_smoothing]

#         Xs[t, :, :] .= predicted_particles_swarm[t].particles_state[:, ind_smoothing]

#         smoothed_particles_swarm[t].particles_state = Xs[t, :, :]

#     end

#     return smoothed_particles_swarm

# end