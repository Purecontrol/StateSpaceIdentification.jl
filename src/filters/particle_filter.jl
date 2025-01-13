mutable struct ParticleFilterState{Z <: Real} <: AbstractFilterState{Z}

    # Filtered and predicted state
    predicted_particles_swarm::Matrix{Z}
    predicted_particles_swarm_mean::Matrix{Z}
    filtered_particles_swarm::Matrix{Z}
    observed_particles_swarm::Matrix{Z}

    # Matrices and vectors
    sampling_weight::Vector{Z}
    ancestor_indice::Vector{Int64}
    filtered_state_μ::Vector{Z}
    filtered_state_Σ::Matrix{Z}

    # Likelihood
    llk::Z

    function ParticleFilterState(
            init_state::GaussianStateStochasticProcess{Z},
            n_X::Int,
            n_Y::Int,
            n_particles::Int
    ) where {Z <: Real}
        predicted_particles_swarm = rand(
            MvNormal(init_state.μ_t, init_state.Σ_t), n_particles)

        predicted_particles_swarm_mean = reshape(
            repeat(init_state.μ_t, n_particles), (n_X, n_particles))

        new{Z}(
            predicted_particles_swarm,
            predicted_particles_swarm_mean,
            zeros(Z, n_X, n_particles),
            zeros(Z, n_Y, n_particles),
            (1 / n_particles) .* ones(Z, n_particles),
            zeros(Int64, n_particles),
            zeros(Z, n_X),
            zeros(Z, n_X, n_X),
            Z(0.0)
        )
    end

    function ParticleFilterState(
            init_state::ParticleSwarmState{Z},
            n_X::Int,
            n_Y::Int,
            n_particles::Int
    ) where {Z <: Real}
        n_particles_init_state = size(init_state.particles_state, 2)
        if n_particles_init_state != n_particles
            @warn "The number of particles of the filter is different from the number of particles of the current state."

            selected_idx_particles = sample_discrete(
                (1 / n_particles_init_state) .* ones(n_particles_init_state),
                n_particles
            )[
                1,
                :
            ]
            init_state.particles_state = init_state.particles_state[
                :, selected_idx_particles]
        end

        predicted_particles_swarm = init_state.particles_state

        predicted_particles_swarm_mean = reshape(
            repeat(vcat(mean(predicted_particles_swarm, dims = 2)...), n_particles),
            (n_X, n_particles)
        )

        new{Z}(
            predicted_particles_swarm,
            predicted_particles_swarm_mean,
            zeros(Z, n_X, n_particles),
            zeros(Z, n_Y, n_particles),
            (1 / n_particles) .* ones(Z, n_particles),
            zeros(Int64, n_particles),
            zeros(Z, n_X),
            zeros(Z, n_X, n_X),
            Z(0.0)
        )
    end
end

mutable struct ParticleFilter{Z <: Real} <: AbstractStochasticMonteCarloFilter{Z}
    init_state_x::AbstractState{Z}
    state::ParticleFilterState{Z}
    n_particles::Int
    positive::Bool

    function ParticleFilter(
            init_state::AbstractState{Z}, n_X, n_Y, n_particles, positive) where {Z <: Real}
        new{Z}(init_state, ParticleFilterState(init_state, n_X, n_Y, n_particles),
            n_particles, positive)
    end

    function ParticleFilter(model::ForecastingModel{Z}; n_particles = 30,
            positive = false) where {Z <: Real}
        return ParticleFilter(
            model.current_state, model.system.n_X, model.system.n_Y, n_particles, positive)
    end
end

mutable struct ConditionalParticleFilter{Z <: Real} <: AbstractStochasticMonteCarloFilter{Z}
    init_state_x::AbstractState{Z}
    state::ParticleFilterState{Z}
    conditional_particle::Matrix{Z}
    n_particles::Int
    positive::Bool
    ancestor_conditional_particle_method::String

    function ConditionalParticleFilter(
            init_state::AbstractState{Z}, n_X, n_Y, conditional_particle, n_particles, positive,
            ancestor_conditional_particle_method) where {Z <: Real}
        new{Z}(init_state, ParticleFilterState(init_state, n_X, n_Y, n_particles),
            conditional_particle,
            n_particles, positive, ancestor_conditional_particle_method)
    end

    function ConditionalParticleFilter(
            model::ForecastingModel{Z}; conditional_particle = Array{Z}(undef, 0, 0), n_particles = 30,
            positive = false, ancestor_conditional_particle_method = "sampling") where {Z <:
                                                                                        Real}
        return ConditionalParticleFilter(
            model.current_state, model.system.n_X, model.system.n_Y, conditional_particle,
            n_particles, positive, ancestor_conditional_particle_method)
    end
end

mutable struct ParticleFilterOutput{Z <: Real} <:
               AbstractStochasticMonteCarloFilterOutput{Z}

    # Predicted, filtered and observed states
    predicted_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}
    filtered_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}
    observed_particles_swarm::TimeSeries{Z, ParticleSwarmState{Z}}

    sampling_weights::Matrix{Z}
    ancestor_indices::Matrix{Int64}
    predicted_particles_swarm_mean::Any

    weighted_gaussian_filtered_state::TimeSeries{Z, GaussianStateStochasticProcess{Z}}

    llk::Z

    function ParticleFilterOutput(
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

        weighted_gaussian_filtered_state = TimeSeries{Z, GaussianStateStochasticProcess{Z}}(
            t_index[1:(end - 1)],
            n_obs,
            model.system.n_X
        )

        sampling_weights = ones(Z, n_obs + 1, n_particles)
        ancestor_indices = zeros(Int64, n_obs, n_particles)
        predicted_particles_swarm_mean = zeros(
            Z, n_obs + 1, model.system.n_X, n_particles)
        filtered_state_mean = zeros(Z, n_obs, model.system.n_X)
        filtered_state_var = zeros(Z, n_obs, model.system.n_X, model.system.n_X)

        return new{Z}(
            predicted_particles_swarm,
            filtered_particles_swarm,
            observed_particles_swarm,
            sampling_weights,
            ancestor_indices,
            predicted_particles_swarm_mean,
            weighted_gaussian_filtered_state,
            Z(0.0)
        )
    end
end

function get_filter_output(filter::Union{ParticleFilter, ConditionalParticleFilter},
        model::ForecastingModel, y_t)
    return ParticleFilterOutput(model, y_t, filter.n_particles)
end

function get_last_state(filter_output::ParticleFilterOutput)
    return filter_output.predicted_particles_swarm[end]
end

@inline check_conditional_particle!(filter_method, n_obs, n_X) = nothing
function check_conditional_particle!(
        filter_method::ConditionalParticleFilter{Z}, n_obs, n_X) where {Z <: Real}
    if isempty(filter_method.conditional_particle)
        @warn "No conditional particle given. Initialize it with zero."
        filter_method.conditional_particle = zeros(Z, n_obs + 1, n_X)
    end
end

function filtering!(
        filter_output::ParticleFilterOutput{Z},
        sys::S,
        filter_method::F,
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        parameters::Vector{Z}
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z},
        F <: AbstractStochasticMonteCarloFilter{Z}}
    n_obs = size(observation_data, 1)

    # Only for ConditionalParticleFilter
    check_conditional_particle!(filter_method, n_obs, sys.n_X)

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

        observation_correction_resampling_forecast_step!(
            filter_method.state,
            view(observation_data, t, :),
            M,
            H,
            R,
            Q,
            filter_method.n_particles,
            filter_method.positive
        )

        replacing_step!(
            filter_method,
            R, 
            t
        )

        save_state_in_filter_output!(filter_output, filter_method.state, t)
    end

    return filter_output
end

function observation_correction_resampling_forecast_step!(
        filter_state::ParticleFilterState{Z}, y, M, H,
        R, Q, n_particles, positive) where {Z <: Real}
    filter_state.sampling_weight = (1 / n_particles) .* ones(Z, n_particles)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    if !isempty(ivar_obs)

        #### Observation STEP ####
        filter_state.observed_particles_swarm = H(filter_state.predicted_particles_swarm)[
            ivar_obs, :]

        # Compute likelihood
        ṽ = y[ivar_obs] - vcat(mean(filter_state.observed_particles_swarm, dims = 2)...)
        S = cov(filter_state.observed_particles_swarm, dims = 2) + Q[ivar_obs, ivar_obs]
        filter_state.llk += -log(2 * pi) / 2 - (1 / 2) * (logdet(S) + ṽ' * inv(S) * ṽ)

        #### Correction STEP ####
        inv_σ = inv(Q[ivar_obs, ivar_obs])
        innov = (repeat(y[ivar_obs, :], 1, n_particles) -
                 filter_state.observed_particles_swarm)
        logmax = max((-0.5 * sum(innov' * inv_σ .* innov', dims = 2))...)
        filter_state.sampling_weight = vcat(exp.(-0.5 *
                                                 sum(innov' * inv_σ .* innov', dims = 2) .-
                                                 logmax)...)
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
    filter_state.filtered_state_μ = vec(
        sum(
        filter_state.predicted_particles_swarm .* filter_state.sampling_weight',
        dims = 2
    ),
    )
    filter_state.filtered_state_Σ = (
        (filter_state.predicted_particles_swarm .- filter_state.filtered_state_μ) .*
        (filter_state.sampling_weight') * transpose(
        filter_state.predicted_particles_swarm .- filter_state.filtered_state_μ,
    )
    )

    #### Resampling STEP ####

    # Resampling indices according to the weights
    filter_state.ancestor_indice = rand(
        Categorical(filter_state.sampling_weight), n_particles)
    # resample!(filter_state.ancestor_indice, filter_state.sampling_weight)

    # Filtered particle swarm
    filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm[
        :, filter_state.ancestor_indice]

    #### Forecast STEP ####
    filter_state.predicted_particles_swarm_mean = M(filter_state.predicted_particles_swarm)
    if positive
        filter_state.predicted_particles_swarm = max.(
            filter_state.predicted_particles_swarm_mean[
                :,
                filter_state.ancestor_indice
            ] + rand(MvNormal(R), n_particles),
            POSITIVE_PRECISION
        )
    else
        filter_state.predicted_particles_swarm = filter_state.predicted_particles_swarm_mean[
            :, filter_state.ancestor_indice] +
                                                 rand(MvNormal(R), n_particles)
    end
end

@inline replacing_step!(filter_method, R, t) = nothing
function replacing_step!(
        filter_method::ConditionalParticleFilter{Z}, R, t) where {Z <: Real}
    filter_state = filter_method.state
    n_particles = filter_method.n_particles
    ancestor_conditional_particle_method = filter_method.ancestor_conditional_particle_method
    conditional_particle = filter_method.conditional_particle[t + 1, :]

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
        w_res = exp.(
            -0.5 * sum(transpose(transpose(res) * inv(R) .* transpose(res)), dims = 1)
        ) .* filter_state.sampling_weight'
        w_res ./= sum(w_res)
        filter_state.ancestor_indice[end] = sample_discrete(w_res, 1)[1, 1]
    end
end

function save_state_in_filter_output!(
        filter_output::ParticleFilterOutput{Z},
        filter_state::ParticleFilterState{Z},
        t::Int
) where {Z <: Real}

    # Save predicted state
    filter_output.predicted_particles_swarm[t + 1].particles_state = filter_state.predicted_particles_swarm
    filter_output.predicted_particles_swarm_mean[t + 1, :, :] = filter_state.predicted_particles_swarm_mean

    # Save filtered and observed particles swarm
    filter_output.filtered_particles_swarm[t].particles_state = filter_state.filtered_particles_swarm
    filter_output.observed_particles_swarm[t].particles_state = filter_state.observed_particles_swarm

    # Save weighted gaussian filtered state
    filter_output.weighted_gaussian_filtered_state[t].μ_t = filter_state.filtered_state_μ
    filter_output.weighted_gaussian_filtered_state[t].Σ_t = filter_state.filtered_state_Σ

    # Save ancestor indices
    filter_output.ancestor_indices[t, :] = filter_state.ancestor_indice

    # Save weights
    filter_output.sampling_weights[t + 1, :] = filter_state.sampling_weight

    # Save likelihood
    filter_output.llk = filter_state.llk
end

function save_initial_state_in_filter_output!(
        filter_output::ParticleFilterOutput{Z},
        filter_state::ParticleFilterState{Z}
) where {Z <: Real}

    # Save initial predicted state
    filter_output.predicted_particles_swarm[1].particles_state = filter_state.predicted_particles_swarm
    filter_output.predicted_particles_swarm_mean[1, :, :] = filter_state.predicted_particles_swarm_mean

    # Initialize weights
    filter_output.sampling_weights[1, :] = filter_state.sampling_weight
end
