using LinearAlgebra
using Distributions


mutable struct ParticleFilterState

    # Filtered and predicted state
    predicted_particles_swarm
    predicted_particles_swarm_mean
    filtered_particles_swarm
    observed_particles_swarm

    # Matrices and vectors
    sampling_weight
    ancestor_indice
    filtered_state_mean
    filtered_state_var

    # Likelihood
    llk

    function ParticleFilterState(init_state::GaussianStateStochasticProcess, n_X, n_Y, n_particles)
        
        predicted_particles_swarm = rand(MvNormal(init_state.μ_t, init_state.σ_t), n_particles)

        predicted_particles_swarm_mean = reshape(repeat(init_state.μ_t, n_particles), (n_X, n_particles))

        new(predicted_particles_swarm, predicted_particles_swarm_mean, zeros(Float64, n_X, n_particles), zeros(Float64, n_Y, n_particles), (1/n_particles).*ones(Float64, n_particles), zeros(Int64, n_particles), zeros(Float64, n_X), zeros(Float64, n_X, n_X), 0.0)

    end

    function ParticleFilterState(init_state::ParticleSwarmState, n_X, n_Y, n_particles)

        n_particles_init_state = size(init_state.particles_state, 2)
        if n_particles_init_state != n_particles
            @warn "The number of particles of the filter is different from the number of particles of the current state."

            selected_idx_particles = sample_discrete((1/n_particles_init_state).*ones(n_particles_init_state), n_particles)[1, :]
            init_state.particles_state = init_state.particles_state[:, selected_idx_particles]

        end
        
        predicted_particles_swarm = init_state.particles_state

        predicted_particles_swarm_mean = reshape(repeat(vcat(mean(predicted_particles_swarm, dims=2)...), n_particles), (n_X, n_particles))

        new(predicted_particles_swarm, predicted_particles_swarm_mean, zeros(Float64, n_X, n_particles), zeros(Float64, n_Y, n_particles), (1/n_particles).*ones(Float64, n_particles), zeros(Int64, n_particles), zeros(Float64, n_X), zeros(Float64, n_X, n_X), 0.0)

    end

end


mutable struct ParticleFilter <: SMCFilter

    n_particles::Int64
    init_state_x::AbstractState
    filter_state::ParticleFilterState
    positive::Bool

    function ParticleFilter(init_state::AbstractState, n_X, n_Y, n_particles, positive)

        new(n_particles, init_state, ParticleFilterState(init_state, n_X, n_Y, n_particles), positive)

    end

    function ParticleFilter(model::ForecastingModel; n_particles=30, positive=false)

        return ParticleFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles, positive)

    end

end


mutable struct ParticleFilterOutput <: SMCFilterOutput

    # Predicted, filtered and observed states
    predicted_particles_swarm::TimeSeries{ParticleSwarmState}
    filtered_particles_swarm::TimeSeries{ParticleSwarmState}
    observed_particles_swarm::TimeSeries{ParticleSwarmState}

    sampling_weights
    ancestor_indices
    predicted_particles_swarm_mean
    filtered_state_mean
    filtered_state_var

    llk::Float64

    function ParticleFilterOutput(model::ForecastingModel, y_t, n_particles)

        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]

        predicted_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, model.system.n_X, t_index; n_particles=n_particles)
        filtered_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs,  model.system.n_X, t_index[1:(end-1)]; n_particles=n_particles)
        observed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs, model.system.n_Y, t_index; n_particles=n_particles)

        sampling_weights = ones(Float64, n_obs+1, n_particles)
        ancestor_indices = zeros(Int64, n_obs, n_particles)
        predicted_particles_swarm_mean = zeros(Float64, n_obs+1, model.system.n_X, n_particles)
        filtered_state_mean = zeros(Float64, n_obs, model.system.n_X)
        filtered_state_var  = zeros(Float64, n_obs, model.system.n_X, model.system.n_X)


        return new(predicted_particles_swarm, filtered_particles_swarm, observed_particles_swarm, sampling_weights, ancestor_indices, predicted_particles_swarm_mean, filtered_state_mean, filtered_state_var, 0.0)

    end

end


function get_filter_output(filter::ParticleFilter, model, y_t)

    return ParticleFilterOutput(model, y_t, filter.n_particles)

end


function get_last_state(filter_output::ParticleFilterOutput)
    return filter_output.filtered_particles_swarm[end]
end


function filter!(filter_output::ParticleFilterOutput, sys::StateSpaceSystem, filter::ParticleFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    # Save initial state
    save_initial_state_in_filter_output!(filter_output, filter.filter_state)

    @inbounds for t in 1:n_obs

        # Get current t_step
        t_step = filter.init_state_x.t + (t-1)*sys.dt

        R = sys.R_t(exogenous_variables[t, :], parameters, t_step)
        Q = sys.Q_t(exogenous_variables[t, :], parameters, t_step)

        # Define actual transition and observation operators
        function M(x)
            return transition(sys, x, exogenous_variables[t, :], control_variables[t, :], parameters, t_step)
        end

        function H(x)
            return observation(sys, x, exogenous_variables[t, :], parameters, t_step)
        end

        update_filter_state!(filter.filter_state, y_t[t, :], M, H, R, Q, filter.n_particles, filter.positive)

        save_state_in_filter_output!(filter_output, filter.filter_state, t)

    end

    return filter_output

end

# KF for optimization
function filter!(sys::StateSpaceSystem, filter::ParticleFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    @inbounds for t in 1:n_obs

        # Get current t_step
        t_step = filter.init_state_x.t + (t-1)*sys.dt

        R = sys.R_t(exogenous_variables[t, :], parameters, t_step)
        Q = sys.Q_t(exogenous_variables[t, :], parameters, t_step)

        # Define actual transition and observation operators
        function M(x)
            return transition(sys, x, exogenous_variables[t, :], control_variables[t, :], parameters, t_step)
        end

        function H(x)
            return observation(sys, x, exogenous_variables[t, :], parameters, t_step)
        end

        update_filter_state!(filter.filter_state, y_t[t, :], M, H, R, Q, filter.n_particles, filter.positive)

    end

    return filter.filter_state.llk / n_obs

end


function update_filter_state!(filter_state::ParticleFilterState, y, M, H, R, Q, n_particles, positive)

    filter_state.sampling_weight = (1/n_particles).*ones(Float64, n_particles)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    if size(ivar_obs, 1) > 0

        #### Observation STEP ####
        filter_state.observed_particles_swarm = H(filter_state.predicted_particles_swarm)[ivar_obs, :]

        # Compute likelihood
        ṽ = y[ivar_obs] - vcat(mean(filter_state.observed_particles_swarm, dims=2)...)
        S = cov(filter_state.observed_particles_swarm, dims=2) + Q[ivar_obs, ivar_obs]
        filter_state.llk += - log(2*pi)/2 - (1/2) * (logdet(S) + ṽ' * inv(S) * ṽ)

        #### Correction STEP ####
        inv_σ = inv(Q[ivar_obs, ivar_obs])
        innov = (repeat(y[ivar_obs, :], 1, n_particles) -  filter_state.observed_particles_swarm)
        logmax = max((-0.5 * sum(innov' * inv_σ .* innov', dims=2))...)
        filter_state.sampling_weight = vcat(exp.(-0.5 * sum(innov' * inv_σ .* innov', dims=2) .- logmax)...)
        # σ = Matrix(Q[ivar_obs, ivar_obs])
        # @inbounds for ip = 1:n_particles
        #     μ = vec(filter_state.observed_particles_swarm[:, ip])
        #     d = MvNormal(μ, σ)
        #     filter_state.sampling_weight[ip] = pdf(d, y[ivar_obs])
        # end

        # Normalization of the weights
        filter_state.sampling_weight ./= sum(filter_state.sampling_weight) 

    end

    # Filtered state
    filter_state.filtered_state_mean = vec(sum(filter_state.predicted_particles_swarm .* filter_state.sampling_weight', dims = 2))
    filter_state.filtered_state_var = ((filter_state.predicted_particles_swarm .- filter_state.filtered_state_mean).*(filter_state.sampling_weight')*transpose(filter_state.predicted_particles_swarm .- filter_state.filtered_state_mean))

    #### Resampling STEP ####

    # Resampling indices according to the weights
    filter_state.ancestor_indice = rand(Categorical(filter_state.sampling_weight), n_particles)
    # resample!(filter_state.ancestor_indice, filter_state.sampling_weight)

    # Filtered particle swarm
    filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm[:, filter_state.ancestor_indice]

    #### Forecast STEP ####
    filter_state.predicted_particles_swarm_mean =  M(filter_state.predicted_particles_swarm)
    if positive
        filter_state.predicted_particles_swarm = max.(filter_state.predicted_particles_swarm_mean[:, filter_state.ancestor_indice] + rand(MvNormal(R), n_particles), 0.001)
    else
        filter_state.predicted_particles_swarm = filter_state.predicted_particles_swarm_mean[:, filter_state.ancestor_indice] + rand(MvNormal(R), n_particles)
    end

end


function save_state_in_filter_output!(filter_output::ParticleFilterOutput, filter_state::ParticleFilterState, t::Int64)

    # Save predicted state
    filter_output.predicted_particles_swarm[t+1].particles_state = filter_state.predicted_particles_swarm
    filter_output.predicted_particles_swarm_mean[t+1, :, :] = filter_state.predicted_particles_swarm_mean

    # Save filtered and observed particles swarm
    filter_output.filtered_particles_swarm[t].particles_state = filter_state.filtered_particles_swarm
    filter_output.observed_particles_swarm[t].particles_state = filter_state.observed_particles_swarm
    filter_output.filtered_state_mean[t, :] = filter_state.filtered_state_mean
    filter_output.filtered_state_var[t, :, :] = filter_state.filtered_state_var

    # Save ancestor indices
    filter_output.ancestor_indices[t, :] = filter_state.ancestor_indice

    # Save weights
    filter_output.sampling_weights[t+1, :] = filter_state.sampling_weight

    # Save likelihood
    filter_output.llk = filter_state.llk

end


function save_initial_state_in_filter_output!(filter_output::ParticleFilterOutput, filter_state::ParticleFilterState)

    # Save initial predicted state
    filter_output.predicted_particles_swarm[1].particles_state = filter_state.predicted_particles_swarm
    filter_output.predicted_particles_swarm_mean[1, :, :] = filter_state.predicted_particles_swarm_mean

    # Initialize weights
    filter_output.sampling_weights[1, :] = filter_state.sampling_weight


end