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

    # Likelihood
    llk

    function ParticleFilterState(init_state::GaussianStateStochasticProcess, n_X, n_Y, n_particles)
        
        predicted_particles_swarm = init_state.μ_t .+ init_state.σ_t*rand(Normal(), n_X, n_particles)

        predicted_particles_swarm_mean = reshape(repeat(init_state.μ_t, n_particles), (n_X, n_particles))

        new(predicted_particles_swarm, predicted_particles_swarm_mean, zeros(Float64, n_X, n_particles), zeros(Float64, n_Y, n_particles), (1/n_particles).*ones(Float64, n_particles), zeros(Int64, n_particles), 0.0)

    end

end


mutable struct ParticleFilter <: AbstractFilter

    n_particles::Int64
    init_state_x::GaussianStateStochasticProcess
    filter_state::ParticleFilterState

    function ParticleFilter(init_state, n_X, n_Y, n_particles)

        new(n_particles, init_state, ParticleFilterState(init_state, n_X, n_Y, n_particles))

    end

end


mutable struct ParticleFilterOutput <: FilterOutput

    # Predicted, filtered and observed states
    predicted_particles_swarm::TimeSeries{ParticleSwarmState}
    filtered_particles_swarm::TimeSeries{ParticleSwarmState}
    observed_particles_swarm::TimeSeries{ParticleSwarmState}

    sampling_weights
    ancestor_indices
    predicted_particles_swarm_mean

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


        return new(predicted_particles_swarm, filtered_particles_swarm, observed_particles_swarm, sampling_weights, ancestor_indices, predicted_particles_swarm_mean, 0.0)

    end

end


function get_filter_output(filter::ParticleFilter, model, y_t)

    return ParticleFilterOutput(model, y_t, filter.n_particles)

end


function filter!(filter_output::ParticleFilterOutput, sys::StateSpaceSystem, filter::ParticleFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

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

        # println("Step $t")

        # Get current t_step
        # t_step = filter.init_state_x.t + (t-1)*sys.dt

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
        filter_state.predicted_particles_swarm = filter_state.predicted_particles_swarm_mean[:, filter_state.ancestor_indice] + rand(MvNormal(R), n_particles)

        ################################################################################
        ################################################################################
        ################################################################################

        # update_filter_state!(filter.filter_state, y_t[t, :], M, H, R, Q, filter.n_particles)

        save_state_in_filter_output!(filter_output, filter.filter_state, t)

    end

    return filter_output, filtered_state_mean,  filtered_state_var

    # ca ne va pas marcher avec cette structure ... car le analysis revient en arriere de plsueirus pas de temps


end


# function update_filter_state!(filter_state::ParticleFilterState, y, M, H, R, Q, n_particles)

# end


function save_state_in_filter_output!(filter_output::ParticleFilterOutput, filter_state::ParticleFilterState, t::Int64)

    # Save predicted state
    filter_output.predicted_particles_swarm[t+1].particles_state = filter_state.predicted_particles_swarm
    filter_output.predicted_particles_swarm_mean[t+1, :, :] = filter_state.predicted_particles_swarm_mean

    # Save filtered and observed particles swarm
    filter_output.filtered_particles_swarm[t].particles_state = filter_state.filtered_particles_swarm
    filter_output.observed_particles_swarm[t].particles_state = filter_state.observed_particles_swarm

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


function resample!(indx::Vector{Int64}, w::Vector{Float64})

    m = length(w)
    q = cumsum(w)
    i = 1
    while i <= m
        sampl = rand()
        j = 1
        while q[j] < sampl
            j = j + 1
        end
        indx[i] = j
        i = i + 1
    end
end


# function PF(da::DataAssimilation, yo::TimeSeries, np::Int64; progress = true, control = nothing)

#     # dimensions
#     nt = yo.nt        # number of observations
#     nv = yo.nv        # number of variables (dimensions of problem)

#     # initialization
#     x̂ = TimeSeries(nt, nv)

#     # special case for k=1
#     k = 1
#     m_xa_traj = Array{Float64,2}[]
#     xf = max.(da.xb .+ da.B.^(1/2)*rand(Normal(), nv, np), 10e-9)
#     ivar_obs = findall(.!isnan.(yo.u[k]))
#     nobs = length(ivar_obs)
#     weights = zeros(Float64, np)
#     indic = zeros(Int64, np)
#     part = [zeros(Float64, (nv, np)) for i = 1:nt]

#     if nobs > 0

#         for ip = 1:np
#             μ = vec(da.H[ivar_obs, :] * xf[:, ip])
#             σ = Matrix(da.R[ivar_obs, ivar_obs])
#             d = MvNormal(μ, σ)
#             weights[ip] = pdf(d, yo.u[k][ivar_obs])
#         end
#         # normalization
#         weights ./= sum(weights)
#         # resampling
#         resample!(indic, weights)
#         part[k] .= xf[:, indic]
#         weights .= weights[indic] ./ sum(weights[indic])
#         x̂.u[k] .= vec(sum(part[k] .* weights', dims = 2))

#         # find number of iterations before new observation
#         # todo: try the findnext function
#         # findnext(.!isnan.(vcat(yo.u'...)), k+1)
#         knext = 1
#         while knext + k <= nt && all(isnan.(yo.u[k+knext]))
#             knext += 1
#         end

#     else

#         weights .= 1.0 / np # weights
#         resample!(indic, weights) # resampling

#     end

#     kcount = 1

#     if progress
#         p = Progress(nt)
#     end

#     for k = 2:nt
#         if progress
#             next!(p)
#         end

#         # update step (compute forecasts) and add small Gaussian noise
#         xf = max.(da.m(part[k-1], yo.t[k-1]; control = control) .+ da.m.Σ_model.^(1/2).*rand(Normal(), nv, np), 10e-9)
#         if kcount <= length(m_xa_traj)
#             m_xa_traj[kcount] .= xf
#         else
#             push!(m_xa_traj, xf)
#         end
#         kcount += 1

#         # analysis step (correct forecasts with observations)
#         ivar_obs = findall(.!isnan.(yo.u[k]))

#         if length(ivar_obs) > 0
#             # weights
#             σ = Symmetric(da.R[ivar_obs, ivar_obs])
#             for ip = 1:np
#                 μ = vec(da.H[ivar_obs, :] * xf[:, ip])
#                 d = MvNormal(μ, σ)
#                 weights[ip] = pdf(d, yo.u[k][ivar_obs])
#             end
#             # normalization
#             weights ./= sum(weights)
#             # resampling
#             resample!(indic, weights)
#             weights .= weights[indic] ./ sum(weights[indic])
#             # stock results
#             for j = 1:knext
#                 jm = k - knext + j
#                 for ip = 1:np
#                     part[jm][:, ip] .= m_xa_traj[j][:, indic[ip]]
#                 end
#                 x̂.u[jm] .= vec(sum(part[jm] .* weights', dims = 2))
#             end
#             kcount = 1
#             # find number of iterations  before new observation
#             knext = 1
#             while knext + k <= nt && all(isnan.(yo.u[k+knext]))
#                 knext += 1
#             end
#         else
#             # stock results
#             part[k] .= xf
#             x̂.u[k] .= vec(sum(xf .* weights', dims = 2))
#         end

#     end

#     x̂ , part

# end