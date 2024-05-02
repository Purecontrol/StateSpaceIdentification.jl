using LinearAlgebra


mutable struct ExtendedKalmanFilterState

    # Filtered and predicted state
    predicted_state_μ
    predicted_state_σ
    filtered_state_μ
    filtered_state_σ

    # Matrices and vectors
    K
    M
    L
    S
    v

    # Likelihood
    llk

    function ExtendedKalmanFilterState(init_state, n_X, n_Y)
        
        new(init_state.μ_t, init_state.σ_t, zeros(n_X), zeros(n_X, n_X), zeros(n_X, n_Y), zeros(n_X, n_Y), zeros(n_X, n_X), zeros(n_Y, n_Y), zeros(n_Y), 0.0)

    end

end


mutable struct ExtendedKalmanFilter <: AbstractFilter

    init_state_x::GaussianStateStochasticProcess
    filter_state::ExtendedKalmanFilterState

    """
    Constructor with gaussian init state.
    """
    function ExtendedKalmanFilter(init_state::GaussianStateStochasticProcess, n_X, n_Y)

        new(init_state, ExtendedKalmanFilterState(init_state, n_X, n_Y))

    end


    """
    Constructor with particle init state.
    """
    function ExtendedKalmanFilter(init_state::ParticleSwarmState, n_X, n_Y)

        μ_t = vcat(mean(init_state.particles_state, dims=2)...)
        σ_t = var(init_state.particles_state, dims=2)
        gaussian_init_state = GaussianStateStochasticProcess(init_state.t, μ_t, σ_t)

        new(gaussian_init_state, ExtendedKalmanFilterState(gaussian_init_state, n_X, n_Y))

    end

    function ExtendedKalmanFilter(model::ForecastingModel)

        return ExtendedKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y)

    end

end


mutable struct ExtendedKalmanFilterOutput <: FilterOutput

    # Predicted and filtered states
    predicted_state::TimeSeries{GaussianStateStochasticProcess}
    filtered_state::TimeSeries{GaussianStateStochasticProcess}
  
    K::Array{Array{Float64, 2}, 1}
    M::Array{Array{Float64, 2}, 1}
    L::Array{Array{Float64, 2}, 1}
    S::Array{Array{Float64, 2}, 1}
    v::Array{Array{Float64, 1}, 1}

    llk::Float64

    function ExtendedKalmanFilterOutput(model::ForecastingModel, y_t)

        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]

        predicted_state = TimeSeries{GaussianStateStochasticProcess}(n_obs+1, model.system.n_X, t_index)
        filtered_state = TimeSeries{GaussianStateStochasticProcess}(n_obs,  model.system.n_X, t_index[1:(end-1)])

        # Define matricess
        K = Array{Array{Float64, 2}, 1}(undef, n_obs)
        M = Array{Array{Float64, 2}, 1}(undef, n_obs)
        L = Array{Array{Float64, 2}, 1}(undef, n_obs)
        S = Array{Array{Float64, 2}, 1}(undef, n_obs)
        v = Array{Array{Float64, 1}, 1}(undef, n_obs)

        return new(predicted_state, filtered_state, K, M, L, S, v)

    end

end


function get_filter_output(filter::ExtendedKalmanFilter, model, y_t)

    return ExtendedKalmanFilterOutput(model, y_t)

end


function get_last_state(filter_output::ExtendedKalmanFilterOutput)
    return filter_output.filtered_state[end]
end


function filter!(filter_output::ExtendedKalmanFilterOutput, sys::StateSpaceSystem, filter::ExtendedKalmanFilter, y_t, exogenous_variables, control_variables, parameters)

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

        function dM(x)
            return sys.dM_t(x, exogenous_variables[t, :], control_variables[t, :], parameters)
        end

        function dH(x)
            return sys.dH_t(x, exogenous_variables[t, :], parameters)
        end

        update_filter_state!(filter.filter_state, y_t[t, :], control_variables[t, :], M, H, R, Q, dM, dH)

        save_state_in_filter_output!(filter_output, filter.filter_state, t)

    end

    return filter_output

end


function update_filter_state!(kalman_state::ExtendedKalmanFilterState, y, u, M, H, R, Q, dM, dH)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Compute innovations and stuff for predicted and filtered states
    jacH = dH(kalman_state.predicted_state_μ)
    kalman_state.v = y[ivar_obs] - H(kalman_state.predicted_state_μ)[ivar_obs]
    kalman_state.S = jacH[ivar_obs, :]*kalman_state.predicted_state_σ*transpose(jacH[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
    kalman_state.M = kalman_state.predicted_state_σ*transpose(jacH[ivar_obs, :])
    inv_S = inv(kalman_state.S)

    # Update states (Update step)
    kalman_state.filtered_state_μ = kalman_state.predicted_state_μ + kalman_state.M*inv_S*kalman_state.v
    kalman_state.filtered_state_σ = kalman_state.predicted_state_σ - kalman_state.M*inv_S*transpose(kalman_state.M)
    
    # Forecast step
    kalman_state.predicted_state_μ = M(kalman_state.filtered_state_μ)
    F = dM(kalman_state.filtered_state_μ)
    kalman_state.predicted_state_σ = transpose(F)*kalman_state.filtered_state_σ*F + R

    # Compute stuff for Kalman smoother
    kalman_state.K = F*kalman_state.M*inv_S #to check
    kalman_state.L = F - kalman_state.K*jacH[ivar_obs, :] #to check

    # Update likelihood
    if length(ivar_obs) > 0
        kalman_state.llk +=  - log(2*pi)/2 - (1/2) * (log(det(kalman_state.S)) + kalman_state.v' * inv_S * kalman_state.v)
    end

end


function save_state_in_filter_output!(filter_output::ExtendedKalmanFilterOutput, filter_state::ExtendedKalmanFilterState, t::Int64)

    # Save predicted state
    filter_output.predicted_state[t+1].μ_t = filter_state.predicted_state_μ
    filter_output.predicted_state[t+1].σ_t = filter_state.predicted_state_σ

    # Save filtered state
    filter_output.filtered_state[t].μ_t = filter_state.filtered_state_μ
    filter_output.filtered_state[t].σ_t = filter_state.filtered_state_σ

    # Save matrix values
    filter_output.K[t] = filter_state.K
    filter_output.M[t] = filter_state.M
    filter_output.L[t] = filter_state.L
    filter_output.S[t] = filter_state.S
    filter_output.v[t] = filter_state.v

    # Save likelihood
    filter_output.llk = filter_state.llk

end


function save_initial_state_in_filter_output!(filter_output::ExtendedKalmanFilterOutput, filter_state::ExtendedKalmanFilterState)

    # Save initial predicted state
    filter_output.predicted_state[1].μ_t = filter_state.predicted_state_μ
    filter_output.predicted_state[1].σ_t = filter_state.predicted_state_σ

end