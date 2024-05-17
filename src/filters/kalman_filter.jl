using LinearAlgebra


mutable struct KalmanFilterState

    # Filtered and predicted state
    predicted_state_μ#::Vector{Float64}
    predicted_state_σ#::Matrix{Float64}
    filtered_state_μ#::Vector{Float64}
    filtered_state_σ#::Matrix{Float64}

    # Matrices and vectors
    K#::Matrix{Float64}
    M#::Matrix{Float64}
    L#::Matrix{Float64}
    S#::Matrix{Float64}
    v#::Vector{Float64}

    # Likelihood
    llk#::Float64

    function KalmanFilterState(init_state, n_X, n_Y)
        
        new(init_state.μ_t, init_state.σ_t, zeros(n_X), zeros(n_X, n_X), zeros(n_X, n_Y), zeros(n_X, n_Y), zeros(n_X, n_X), zeros(n_Y, n_Y), zeros(n_Y), 0.0)

    end

end


mutable struct KalmanFilter <: AbstractFilter

    init_state_x::GaussianStateStochasticProcess
    filter_state::KalmanFilterState

    """
    Constructor with gaussian init state.
    """
    function KalmanFilter(init_state::GaussianStateStochasticProcess, n_X, n_Y)

        new(init_state, KalmanFilterState(init_state, n_X, n_Y))

    end


    """
    Constructor with particle init state.
    """
    function KalmanFilter(init_state::ParticleSwarmState, n_X, n_Y)

        μ_t = vcat(mean(init_state.particles_state, dims=2)...)
        σ_t = var(init_state.particles_state, dims=2)
        gaussian_init_state = GaussianStateStochasticProcess(init_state.t, μ_t, σ_t)

        new(gaussian_init_state, KalmanFilterState(gaussian_init_state, n_X, n_Y))

    end

    function KalmanFilter(model::ForecastingModel)

        return KalmanFilter(model.current_state, model.system.n_X, model.system.n_Y)

    end

end


mutable struct KalmanFilterOutput <: FilterOutput

    # Predicted and filtered states
    predicted_state::TimeSeries{GaussianStateStochasticProcess}
    filtered_state::TimeSeries{GaussianStateStochasticProcess}
  
    K::Array{Array{Float64, 2}, 1}
    M::Array{Array{Float64, 2}, 1}
    L::Array{Array{Float64, 2}, 1}
    S::Array{Array{Float64, 2}, 1}
    v::Array{Array{Float64, 1}, 1}

    llk::Float64

    function KalmanFilterOutput(model::ForecastingModel, y_t)

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


function get_filter_output(filter::KalmanFilter, model, y_t)

    return KalmanFilterOutput(model, y_t)

end


function get_last_state(filter_output::KalmanFilterOutput)
    return filter_output.predicted_state[end]
end


# function type_of_state(kf::KalmanFilter)

#     return GaussianStateStochasticProcess

# end


# General KF
function filter!(filter_output::KalmanFilterOutput, sys::GaussianLinearStateSpaceSystem, filter::KalmanFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    # Save initial state
    save_initial_state_in_filter_output!(filter_output, filter.filter_state)

    @inbounds for t in 1:n_obs

        # Get current t_step
        t_step = filter.init_state_x.t + (t-1)*sys.dt

        # Get current matrix A, B, H and Q
        A = sys.A_t(exogenous_variables[t, :], parameters, t_step)
        B = sys.B_t(exogenous_variables[t, :], parameters, t_step)
        c = sys.c_t(exogenous_variables[t, :], parameters, t_step)
        H = sys.H_t(exogenous_variables[t, :], parameters, t_step)
        d = sys.d_t(exogenous_variables[t, :], parameters, t_step)
        R = sys.R_t(exogenous_variables[t, :], parameters, t_step)
        Q = sys.Q_t(exogenous_variables[t, :], parameters, t_step)

        update_filter_state!(filter.filter_state, y_t[t, :], control_variables[t, :], A, B, c, H, d, R, Q)

        save_state_in_filter_output!(filter_output, filter.filter_state, t)

    end

    return filter_output

end


# KF for optimization
function filter!(sys::GaussianLinearStateSpaceSystem, filter_method::KalmanFilter, y_t, exogenous_variables, control_variables, parameters)

    n_obs = size(y_t, 1)

    zeta = 0.0
    @inbounds for t in 1:n_obs

        # Get current t_step
        t_step = filter_method.init_state_x.t + (t-1)*sys.dt

        # Get current matrix A, B, H and Q
        A = sys.A_t(exogenous_variables[t, :], parameters, t_step)
        B = sys.B_t(exogenous_variables[t, :], parameters, t_step)
        c = sys.c_t(exogenous_variables[t, :], parameters, t_step)
        H = sys.H_t(exogenous_variables[t, :], parameters, t_step)
        d = sys.d_t(exogenous_variables[t, :], parameters, t_step)
        R = sys.R_t(exogenous_variables[t, :], parameters, t_step)
        Q = sys.Q_t(exogenous_variables[t, :], parameters, t_step)

        update_filter_state!(filter_method.filter_state, y_t[t, :], control_variables[t, :], A, B, c, H, d, R, Q)

    end

    return filter_method.filter_state.llk / n_obs

end


function update_filter_state!(kalman_state::KalmanFilterState, y, u, A, B, c, H, d, R, Q)

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))

    # Compute innovations and stuff for predicted and filtered states
    kalman_state.v = y[ivar_obs] - (H[ivar_obs, :]*kalman_state.predicted_state_μ + d[ivar_obs])
    kalman_state.S = H[ivar_obs, :]*kalman_state.predicted_state_σ*transpose(H[ivar_obs, :]) + Q[ivar_obs, ivar_obs]
    kalman_state.M = kalman_state.predicted_state_σ*transpose(H[ivar_obs, :])
    inv_S = inv(kalman_state.S)

    # Update states (Update step)
    kalman_state.filtered_state_μ = kalman_state.predicted_state_μ + kalman_state.M*inv_S*kalman_state.v
    kalman_state.filtered_state_σ = kalman_state.predicted_state_σ - kalman_state.M*inv_S*transpose(kalman_state.M)
    
    # Forecast step
    kalman_state.predicted_state_μ = A*kalman_state.filtered_state_μ + B*u + c
    kalman_state.predicted_state_σ = A*kalman_state.filtered_state_σ*transpose(A) + R

    # Compute stuff for Kalman smoother
    kalman_state.K = A*kalman_state.M*inv_S
    kalman_state.L = A - kalman_state.K*H[ivar_obs, :]

    # Update likelihood
    if length(ivar_obs) > 0
        kalman_state.llk +=  - log(2*pi)/2 - (1/2) * (logdet(kalman_state.S) + kalman_state.v' * inv_S * kalman_state.v)
    end

end


function save_state_in_filter_output!(filter_output::KalmanFilterOutput, filter_state::KalmanFilterState, t::Int64)

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


function save_initial_state_in_filter_output!(filter_output::KalmanFilterOutput, filter_state::KalmanFilterState)

    # Save initial predicted state
    filter_output.predicted_state[1].μ_t = filter_state.predicted_state_μ
    filter_output.predicted_state[1].σ_t = filter_state.predicted_state_σ

end