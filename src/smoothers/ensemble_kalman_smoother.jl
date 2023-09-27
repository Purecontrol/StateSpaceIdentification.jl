using LinearAlgebra


mutable struct EnsembleKalmanSmootherState

    # Filtered and predicted state
    smoothed_particles_swarm

    function EnsembleKalmanSmootherState(n_X, n_Y, n_particles)
        
        new(zeros(Float64, n_X, n_particles))

    end

end


mutable struct EnsembleKalmanSmoother <: AbstractSmoother

    n_particles::Int64
    smoother_state::EnsembleKalmanSmootherState

    function EnsembleKalmanSmoother(n_X, n_Y, n_particles)

        new(n_particles, EnsembleKalmanSmootherState(n_X, n_Y, n_particles))

    end

end


mutable struct EnsembleKalmanSmootherOutput <: SmootherOutput

    smoothed_state::TimeSeries{ParticleSwarmState}

    function EnsembleKalmanSmootherOutput(model::ForecastingModel, y_t, n_particles)

        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]

        smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, model.system.n_X, t_index; n_particles=n_particles)

        return new(smoothed_particles_swarm)

    end

end


function get_smoother_output(smoother::EnsembleKalmanSmoother, model, y_t)

    return EnsembleKalmanSmootherOutput(model, y_t, smoother.n_particles)

end


function smoother!(smoother_output::EnsembleKalmanSmootherOutput, filter_output::EnsembleKalmanFilterOutput, sys::StateSpaceSystem, smoother::EnsembleKalmanSmoother, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    initialize_smoother!(smoother_output, smoother.smoother_state, filter_output.predicted_particles_swarm[end])

    # Backward recursions
    @inbounds for t in (n_obs):-1:1

        # Get current t_step
        t_step = filter_output.predicted_particles_swarm[1].t + (t-1)*sys.dt

        Xp = filter_output.predicted_particles_swarm[t+1].particles_state
        Xf = filter_output.filtered_particles_swarm[t].particles_state

        update_smoother_state!(smoother.smoother_state, Xp, Xf)

        save_state_in_smoother_output!(smoother_output, smoother.smoother_state, t)

    end

    return smoother_output

end


function update_smoother_state!(smoother_state, Xp, Xf)

    Paf = cov(Xf, Xp, dims=2)
    Pff = cov(Xp, Xp, dims=2)
    K = Paf*pinv(Pff)

    smoother_state.smoothed_particles_swarm = Xf + K*(smoother_state.smoothed_particles_swarm - Xp)

end


function save_state_in_smoother_output!(smoother_output::EnsembleKalmanSmootherOutput, smoother_state::EnsembleKalmanSmootherState, t::Int64)

    # Save smoothed state
    smoother_output.smoothed_state[t].particles_state = smoother_state.smoothed_particles_swarm

end


function initialize_smoother!(smoother_output::EnsembleKalmanSmootherOutput, smoother_state::EnsembleKalmanSmootherState, last_predicted_state)

    # Initialize KalmanSmoother state
    smoother_state.smoothed_particles_swarm = last_predicted_state.particles_state

    # Save initial predicted state
    smoother_output.smoothed_state[end].particles_state = last_predicted_state.particles_state

end
