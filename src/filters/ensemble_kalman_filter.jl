using LinearAlgebra
using Distributions


mutable struct EnsembleKalmanFilterState

    # Filtered and predicted state
    predicted_particles_swarm
    filtered_particles_swarm
    observed_particles_swarm

    # Matrices and vectors

    # Likelihood
    llk

    function EnsembleKalmanFilterState(init_state::GaussianStateStochasticProcess, n_X, n_Y, n_particles)
        
        predicted_particles_swarm = init_state.μ_t .+ init_state.σ_t*rand(Normal(), n_X, n_particles)

        new(predicted_particles_swarm, zeros(Float64, n_X, n_particles), zeros(Float64, n_Y, n_particles), 0.0)

    end

end


mutable struct EnsembleKalmanFilter <: SMCFilter

    n_particles::Int64
    init_state_x::GaussianStateStochasticProcess
    filter_state::EnsembleKalmanFilterState
    positive::Bool

    function EnsembleKalmanFilter(init_state, n_X, n_Y, n_particles, positive)

        new(n_particles, init_state, EnsembleKalmanFilterState(init_state, n_X, n_Y, n_particles), positive)

    end

    function EnsembleKalmanFilter(model::ForecastingModel; n_particles=30, positive=false)

        new(n_particles, model.current_state, EnsembleKalmanFilterState(model.current_state, model.system.n_X, model.system.n_Y, n_particles), positive)

    end

end


mutable struct EnsembleKalmanFilterOutput <: SMCFilterOutput

    # Predicted, filtered and observed states
    predicted_particles_swarm::TimeSeries{ParticleSwarmState}
    filtered_particles_swarm::TimeSeries{ParticleSwarmState}
    observed_particles_swarm::TimeSeries{ParticleSwarmState}

    llk::Float64

    function EnsembleKalmanFilterOutput(model::ForecastingModel, y_t, n_particles)

        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]

        predicted_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, model.system.n_X, t_index; n_particles=n_particles)
        filtered_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs,  model.system.n_X, t_index[1:(end-1)]; n_particles=n_particles)
        observed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs, model.system.n_Y, t_index; n_particles=n_particles)

        return new(predicted_particles_swarm, filtered_particles_swarm, observed_particles_swarm, 0.0)

    end

end


function get_filter_output(filter::EnsembleKalmanFilter, model, y_t)

    return EnsembleKalmanFilterOutput(model, y_t, filter.n_particles)

end


function filter!(filter_output::EnsembleKalmanFilterOutput, sys::StateSpaceSystem, filter::EnsembleKalmanFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

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

        update_filter_state!(filter.filter_state, y_t[t, :], M, H, R, Q, filter.n_particles, filter.positive)

        save_state_in_filter_output!(filter_output, filter.filter_state, t)

    end

    return filter_output

end


# KF for optimization
function filter!(sys::StateSpaceSystem, filter::EnsembleKalmanFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

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

        update_filter_state!(filter.filter_state, y_t[t, :], M, H, R, Q, filter.n_particles, filter.positive)

    end

    return filter.filter_state.llk / n_obs

end


function update_filter_state!(filter_state::EnsembleKalmanFilterState, y, M, H, R, Q, n_particles, positive)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Observation equation
    filter_state.observed_particles_swarm = H(filter_state.predicted_particles_swarm)[ivar_obs, :]

    # Approximation of Var(H(x)) and Cov(x, v)
    if length(ivar_obs) > 0

        ef_states = filter_state.predicted_particles_swarm * (Matrix(I, n_particles, n_particles) .- 1 / n_particles)
        ef_obs = filter_state.observed_particles_swarm * (Matrix(I, n_particles, n_particles) .- 1 / n_particles)
        HPHt = (ef_obs * ef_obs') ./ (n_particles - 1)
        PHt = ((ef_states * ef_obs') ./ (n_particles - 1))

        # Compute innovations and stuff for predicted and filtered states
        S = HPHt + Q[ivar_obs, ivar_obs]
        inv_S = inv(S)
        K = PHt * inv_S
        v = y[ivar_obs] .- filter_state.observed_particles_swarm + rand(MvNormal(Q), n_particles)[ivar_obs, :]

        # Update prediction
        filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm + K*v
        
    else
        filter_state.filtered_particles_swarm = filter_state.predicted_particles_swarm
    end

    # Forecast step
    if positive
        filter_state.predicted_particles_swarm =  max.(M(filter_state.filtered_particles_swarm) + rand(MvNormal(R), n_particles), 0.001)
    else
        filter_state.predicted_particles_swarm =  M(filter_state.filtered_particles_swarm) + rand(MvNormal(R), n_particles)
    end

    # Update llk
    if size(ivar_obs, 1) > 0
        ṽ = y[ivar_obs] - vcat(mean(filter_state.observed_particles_swarm, dims=2)...)
        filter_state.llk +=  - log(2*pi)/2 - (1/2) * (logdet(S) + ṽ' * inv_S * ṽ)
    end

end


function save_state_in_filter_output!(filter_output::EnsembleKalmanFilterOutput, filter_state::EnsembleKalmanFilterState, t::Int64)

    # Save predicted state
    filter_output.predicted_particles_swarm[t+1].particles_state = filter_state.predicted_particles_swarm

    # Save filtered and observed particles swarm
    filter_output.filtered_particles_swarm[t].particles_state = filter_state.filtered_particles_swarm
    filter_output.observed_particles_swarm[t].particles_state = filter_state.observed_particles_swarm

    # Save likelihood
    filter_output.llk = filter_state.llk

end


function save_initial_state_in_filter_output!(filter_output::EnsembleKalmanFilterOutput, filter_state::EnsembleKalmanFilterState)

    # Save initial predicted state
    filter_output.predicted_particles_swarm[1].particles_state = filter_state.predicted_particles_swarm

end
