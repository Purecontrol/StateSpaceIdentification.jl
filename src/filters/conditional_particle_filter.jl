mutable struct ConditionalParticleFilterState

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

    # Added for conditional particle filter
    conditional_particle

    function ConditionalParticleFilterState(init_state::GaussianStateStochasticProcess, n_X, n_Y, n_particles, conditional_particle)
        
        predicted_particles_swarm = init_state.μ_t .+ init_state.σ_t*rand(Normal(), n_X, n_particles)

        predicted_particles_swarm_mean = reshape(repeat(init_state.μ_t, n_particles), (n_X, n_particles))

        new(predicted_particles_swarm, predicted_particles_swarm_mean, zeros(Float64, n_X, n_particles), zeros(Float64, n_Y, n_particles), (1/n_particles).*ones(Float64, n_particles), zeros(Int64, n_particles), zeros(Float64, n_X), zeros(Float64, n_X, n_X), 0.0, conditional_particle)
    end

end


mutable struct ConditionalParticleFilter <: SMCFilter

    n_particles::Int64
    init_state_x::GaussianStateStochasticProcess
    filter_state::ConditionalParticleFilterState
    positive::Bool
    ancestor_conditional_particle_method::String

    function ConditionalParticleFilter(init_state, n_X, n_Y, n_particles, positive, conditional_particle, ancestor_conditional_particle_method)

        new(n_particles, init_state, ConditionalParticleFilterState(init_state, n_X, n_Y, n_particles, conditional_particle), positive, ancestor_conditional_particle_method)

    end

    function ConditionalParticleFilter(model::ForecastingModel; n_particles=30, positive=false, conditional_particle=nothing, ancestor_conditional_particle_method="tracking")

        new(n_particles, model.current_state, ConditionalParticleFilterState(model.current_state, model.system.n_X, model.system.n_Y, n_particles, conditional_particle), positive, ancestor_conditional_particle_method)

    end

end


mutable struct ConditionalParticleFilterOutput <: SMCFilterOutput

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

    function ConditionalParticleFilterOutput(model::ForecastingModel, y_t, n_particles)

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


function get_filter_output(filter::ConditionalParticleFilter, model, y_t)

    return ConditionalParticleFilterOutput(model, y_t, filter.n_particles)

end


function filter!(filter_output::ConditionalParticleFilterOutput, sys::StateSpaceSystem, filter::ConditionalParticleFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    # Set conditional particle if not defined
    if filter.filter_state.conditional_particle === nothing
        filter.filter_state.conditional_particle = zeros(Float64, n_obs+1, sys.n_X)
    end

    # Save initial state
    save_initial_state_in_filter_output!(filter_output, filter.filter_state)

    @inbounds for t in 1:n_obs

        R = sys.R_t(exogenous_variables[t, :], parameters)
        Q = sys.Q_t(exogenous_variables[t, :], parameters)

        # Define actual transition and observation operators
        function M(x)
            return transition(sys, x, exogenous_variables[t, :], control_variables[t, :], parameters)
        end

        function H(x)
            return observation(sys, x, exogenous_variables[t, :], parameters)
        end

        update_filter_state!(filter.filter_state, y_t[t, :], M, H, R, Q, filter.filter_state.conditional_particle[t+1, :], filter.n_particles, filter.positive, filter.ancestor_conditional_particle_method)

        save_state_in_filter_output!(filter_output, filter.filter_state, t)

    end

    return filter_output

end


function update_filter_state!(filter_state::ConditionalParticleFilterState, y, M, H, R, Q, conditional_particle, n_particles, positive, ancestor_conditional_particle_method)

    filter_state.sampling_weight = (1/n_particles).*ones(Float64, n_particles)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    if size(ivar_obs, 1) > 0

        #### Observation STEP ####
        filter_state.observed_particles_swarm = H(filter_state.predicted_particles_swarm)[ivar_obs, :]

        # Compute likelihood
        ṽ = y[ivar_obs] - vcat(mean(filter_state.observed_particles_swarm, dims=2)...)
        S = cov(filter_state.observed_particles_swarm, dims=2) + Q[ivar_obs, ivar_obs]
        filter_state.llk += - log(2*pi)/2 - (1/2) * (log(det(S)) + ṽ' * inv(S) * ṽ)

        #### Correction STEP ####
        σ = Matrix(Q[ivar_obs, ivar_obs])
        for ip = 1:n_particles
            μ = vec(filter_state.observed_particles_swarm[:, ip])
            d = MvNormal(μ, σ)
            filter_state.sampling_weight[ip] = pdf(d, y[ivar_obs])
        end

        # Normalization of the weights
        filter_state.sampling_weight ./= sum(filter_state.sampling_weight) 

    end

    # Filtered state
    filter_state.filtered_state_mean = vec(sum(filter_state.predicted_particles_swarm .* filter_state.sampling_weight', dims = 2))
    filter_state.filtered_state_var = ((filter_state.predicted_particles_swarm .- filter_state.filtered_state_mean).*(filter_state.sampling_weight')*transpose(filter_state.predicted_particles_swarm .- filter_state.filtered_state_mean))

    #### Resampling STEP ####

    # Resampling indices according to the weights
    resample!(filter_state.ancestor_indice, filter_state.sampling_weight)

    # Filtered particle swarm
    filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm[:, filter_state.ancestor_indice]

    #### Forecast STEP ####
    filter_state.predicted_particles_swarm_mean =  M(filter_state.predicted_particles_swarm)
    if positive
        filter_state.predicted_particles_swarm = max.(filter_state.predicted_particles_swarm_mean[:, filter_state.ancestor_indice] + rand(MvNormal(R), n_particles), 0.001)
    else
        filter_state.predicted_particles_swarm = filter_state.predicted_particles_swarm_mean[:, filter_state.ancestor_indice] + rand(MvNormal(R), n_particles)
    end

    #### Replacing STEP ####
    # filter_state.predicted_particles_swarm_mean[:, end] = conditional_particle
    filter_state.predicted_particles_swarm[:, end] = conditional_particle

    # Resample or not weighting indices
    # Option 1
    if ancestor_conditional_particle_method == "tracking"
        filter_state.ancestor_indice[end] = n_particles 
    # Option 2
    elseif ancestor_conditional_particle_method == "sampling"
        res = conditional_particle .- filter_state.predicted_particles_swarm_mean
        w_res = exp.(-.5  * sum(transpose(transpose(res) * inv(R) .* transpose(res)), dims=1)).*filter_state.sampling_weight'
        w_res ./= sum(w_res) 
        filter_state.ancestor_indice[end] = sample_discrete(w_res,1)[1,1]
    end

end


function filter_old_old_old!(filter_output::ConditionalParticleFilter, sys::StateSpaceSystem, filter::ConditionalParticleFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    if filter.filter_state.conditional_particle === nothing
        filter.filter_state.conditional_particle = zeros(Float64, n_obs+1, sys.n_X)
    end

    # Save initial state
    save_initial_state_in_filter_output!(filter_output, filter.filter_state)

    ################################################################################
    ################################################################################
    ################################################################################

    # weights = zeros(Float64, filter.n_particles)
    filtered_state_mean = zeros(Float64, n_obs, sys.n_X)
    filtered_state_var = zeros(Float64, n_obs, sys.n_X, sys.n_X)

    ################################################################################
    ################################################################################
    ################################################################################

    @inbounds for t in 1:n_obs

        
        R = sys.R_t(exogenous_variables[t, :], parameters)
        Q = sys.Q_t(exogenous_variables[t, :], parameters)

        # Define actual transition and observation operators
        function M(x)
            return transition(sys, x, exogenous_variables[t, :], control_variables[t, :], parameters)
        end

        function H(x)
            return observation(sys, x, exogenous_variables[t, :], parameters)
        end

        ################################################################################
        ################################################################################
        ################################################################################
        
        filter_state = filter.filter_state
        n_particles = filter.n_particles

        filter_state.sampling_weight = (1/n_particles).*ones(Float64, n_particles)

        # Check the number of correct observations
        ivar_obs = findall(.!isnan.(y_t[t, :]))

        if size(ivar_obs, 1) > 0

            #### Observation STEP ####
            filter_state.observed_particles_swarm = H(filter_state.predicted_particles_swarm)[ivar_obs, :]

            # Compute likelihood
            ṽ = y_t[t, ivar_obs] - vcat(mean(filter_state.observed_particles_swarm, dims=2)...)
            S = cov(filter_state.observed_particles_swarm, dims=2) + Q[ivar_obs, ivar_obs]
            filter_state.llk += - log(2*pi)/2 - (1/2) * (log(det(S)) + ṽ' * inv(S) * ṽ)

            #### Correction STEP ####
            σ = Matrix(Q[ivar_obs, ivar_obs])
            for ip = 1:n_particles
                μ = vec(filter_state.observed_particles_swarm[:, ip])
                d = MvNormal(μ, σ)
                filter_state.sampling_weight[ip] = pdf(d, y_t[t, ivar_obs])
            end

            # Normalization of the weights
            filter_state.sampling_weight ./= sum(filter_state.sampling_weight) 

        end

        # Filtered state
        filtered_state_mean[t, :] = vec(sum(filter_state.predicted_particles_swarm .* filter_state.sampling_weight', dims = 2))
        filtered_state_var[t, :, :] = ((filter_state.predicted_particles_swarm .- filtered_state_mean[t, :]).*(filter_state.sampling_weight')*transpose(filter_state.predicted_particles_swarm .- filtered_state_mean[t, :]))

        #### Resampling STEP ####

        # Resampling indices according to the weights
        resample!(filter_state.ancestor_indice, filter_state.sampling_weight)

        # Filtered particle swarm
        filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm[:, filter_state.ancestor_indice]

        #### Forecast STEP ####
        filter_state.predicted_particles_swarm_mean =  M(filter_state.predicted_particles_swarm)
        filter_state.predicted_particles_swarm = max.(filter_state.predicted_particles_swarm_mean[:, filter_state.ancestor_indice] + rand(MvNormal(R), n_particles), 0.001)

        ### Replacing STEP ###
        # filter_state.predicted_particles_swarm_mean[:, end] = filter_state.conditional_particle[t, :]
        filter_state.predicted_particles_swarm[:, end] = filter_state.conditional_particle[t+1, :]

        # Resample or not weighting indices
        # Option 1
        # filter_state.ancestor_indice[end] = n_particles 
        # Option 2
        res = filter_state.conditional_particle[t+1, :] .- filter_state.predicted_particles_swarm_mean
        w_res = exp.(-.5  * sum(transpose(transpose(res) * inv(R) .* transpose(res)), dims=1)).*filter_state.sampling_weight'
        w_res ./= sum(w_res) 
        filter_state.ancestor_indice[end] = sample_discrete(w_res,1)[1,1]


        ################################################################################
        ################################################################################
        ################################################################################

        # update_filter_state!(filter.filter_state, y_t[t, :], M, H, R, Q, filter.n_particles)

        save_state_in_filter_output!(filter_output, filter.filter_state, t)
        

    end

    return filter_output, filtered_state_mean,  filtered_state_var

    # ca ne va pas marcher avec cette structure ... car le analysis revient en arriere de plsueirus pas de temps


end


function save_state_in_filter_output!(filter_output::ConditionalParticleFilterOutput, filter_state::ConditionalParticleFilterState, t::Int64)

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


function save_initial_state_in_filter_output!(filter_output::ConditionalParticleFilterOutput, filter_state::ConditionalParticleFilterState)

    # Replacing step => Pas de replacing step à l'initialisation ? TODO : Attente réponse Valérie
    # filter_state.predicted_particles_swarm[:, end] = filter_state.conditional_particle[1, :]
    # filter_state.predicted_particles_swarm_mean[:, end] = filter_state.conditional_particle[1, :]

    # Save initial predicted state
    filter_output.predicted_particles_swarm[1].particles_state = filter_state.predicted_particles_swarm
    filter_output.predicted_particles_swarm_mean[1, :, :] = filter_state.predicted_particles_swarm_mean

    # Initialize weights
    filter_output.sampling_weights[1, :] = filter_state.sampling_weight
end