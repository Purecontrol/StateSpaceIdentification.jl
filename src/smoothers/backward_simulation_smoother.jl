using LinearAlgebra


mutable struct BackwardSimulationState

    # Filtered and predicted state
    smoothed_particles_swarm

    function BackwardSimulationState(n_X, n_Y, n_particles)
        
        new(zeros(Float64, n_X, n_particles))

    end

end


mutable struct BackwardSimulationSmoother <: AbstractSmoother

    n_particles::Int64
    smoother_state::BackwardSimulationState

    function BackwardSimulationSmoother(n_X, n_Y, n_particles)

        new(n_particles, BackwardSimulationState(n_X, n_Y, n_particles))

    end

    function BackwardSimulationSmoother(model::ForecastingModel; n_particles=30)

        new(n_particles, BackwardSimulationState(model.system.n_X, model.system.n_Y, n_particles))

    end

end


mutable struct BackwardSimulationSmootherOutput <: SmootherOutput

    smoothed_state::TimeSeries{ParticleSwarmState}

    function BackwardSimulationSmootherOutput(model::ForecastingModel, y_t, n_particles)

        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]

        smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, model.system.n_X, t_index; n_particles=n_particles)

        return new(smoothed_particles_swarm)

    end

end


function get_smoother_output(smoother::BackwardSimulationSmoother, model, y_t)

    return BackwardSimulationSmootherOutput(model, y_t, smoother.n_particles)

end


function smoother!(smoother_output::BackwardSimulationSmootherOutput, filter_output::ParticleFilterOutput, sys::StateSpaceSystem, smoother::BackwardSimulationSmoother, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)
    n_filtering = size(filter_output.sampling_weights, 2)

    initialize_smoother!(smoother_output, smoother.smoother_state, filter_output.predicted_particles_swarm[end], filter_output.sampling_weights[end, :], smoother.n_particles, n_filtering)

    # Backward recursions
    @inbounds for t in (n_obs):-1:1

        predicted_particle_swarm = filter_output.predicted_particles_swarm[t].particles_state
        predicted_particles_swarm_mean = filter_output.predicted_particles_swarm_mean[t+1, :, :]
        sampling_weights = filter_output.sampling_weights[t+1, :]

        σ = pinv(Matrix(sys.R_t(exogenous_variables[t, :], parameters)))

        update_smoother_state!(smoother.smoother_state, predicted_particle_swarm, predicted_particles_swarm_mean, sampling_weights, σ, smoother.n_particles, n_filtering, sys.n_X)

        save_state_in_smoother_output!(smoother_output, smoother.smoother_state, t)

    end

    return smoother_output

end


function update_smoother_state!(smoother_state::BackwardSimulationState, Xp, Xp_mean, W, σ, n_smoothing, n_filtering, n_X)

    v = smoother_state.smoothed_particles_swarm[:, :, [CartesianIndex()]] .- Xp_mean[:, [CartesianIndex()], :]

    smoothing_weights = zeros(n_smoothing, n_filtering)
    @inbounds for i in 1:n_X
        @inbounds for j in 1:n_X
            smoothing_weights += exp.((-1/2)*v[i, :, :]*σ[i, j].*v[j, :, :])
        end
    end

    smoothing_weights = smoothing_weights.*W[[CartesianIndex()], :]
    smoothing_weights ./= sum(smoothing_weights, dims=2) 

    ind_smoothing = sample_discrete(smoothing_weights', 1, n_exp=n_smoothing)[:, 1]

    # smoother_state.smoothed_particles_swarm .= Xp[:, ind_smoothing]
    smoother_state.smoothed_particles_swarm = Xp[:, ind_smoothing]


end


function save_state_in_smoother_output!(smoother_output::BackwardSimulationSmootherOutput, smoother_state::BackwardSimulationState, t::Int64)

    # Save smoothed state
    smoother_output.smoothed_state[t].particles_state = smoother_state.smoothed_particles_swarm

end


function initialize_smoother!(smoother_output::BackwardSimulationSmootherOutput, smoother_state::BackwardSimulationState, last_predicted_state, last_sampling_weights, n_smoothing, n_filtering)

    ind_smoothing = sample_discrete((1/n_filtering).*ones(n_filtering), n_smoothing)[1, :]
    # ind_smoothing = sample_discrete(last_sampling_weights, n_smoothing)

    # Initialize KalmanSmoother state
    # smoother_state.smoothed_particles_swarm .= last_predicted_state.particles_state[:, ind_smoothing']
    smoother_state.smoothed_particles_swarm = last_predicted_state.particles_state[:, ind_smoothing]

    # Save initial predicted state
    smoother_output.smoothed_state[end].particles_state = smoother_state.smoothed_particles_swarm

end

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################


# function backward_smoothing(y_t, exogenous_variables, filter_output, model, parameters; n_smoothing=30)

#     n_obs = size(y_t, 1)
#     n_X = model.system.n_X

#     # Create output structure
#     t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]
#     smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, 1, t_index; n_particles=n_smoothing)

#     # Get output filter
#     sampling_weight = filter_output.sampling_weights
#     predicted_particles_swarm = filter_output.predicted_particles_swarm
#     predicted_particles_swarm_mean = filter_output.predicted_particles_swarm_mean
#     n_filtering = size(sampling_weight, 2)

#     Xs = zeros(Float64, n_obs+1, n_X, n_smoothing)
    
#     ind_smoothing = sample_discrete((1/n_filtering).*ones(n_filtering), n_smoothing)

#     Xs[end, :, :] .= predicted_particles_swarm[end].particles_state[:, ind_smoothing']
#     smoothed_particles_swarm[end].particles_state = Xs[end, :, :]

#     @inbounds for t in (n_obs):-1:1

#         # println(t)

#         σ = pinv(Matrix(model.system.R_t(exogenous_variables[t, :], parameters)))

#         v = Xs[t+1, :, :, [CartesianIndex()]] .- predicted_particles_swarm_mean[t+1, :, [CartesianIndex()], :]

#         smoothing_weights = zeros(n_smoothing, n_filtering)
#         @inbounds for i in 1:n_X
#             @inbounds for j in 1:n_X
#                 smoothing_weights += exp.((-1/2)*v[i, :, :]*σ[i, j].*v[j, :, :])
#             end
#         end

#         smoothing_weights = smoothing_weights.*sampling_weight[t+1, [CartesianIndex()], :]
#         smoothing_weights ./= sum(smoothing_weights, dims=2) 

#         ind_smoothing = sample_discrete(smoothing_weights', 1, n_exp=n_smoothing)

#         Xs[t, :, :] .= predicted_particles_swarm[t].particles_state[:, ind_smoothing]

#         smoothed_particles_swarm[t].particles_state = Xs[t, :, :]

#     end

#     return smoothed_particles_swarm

# end




