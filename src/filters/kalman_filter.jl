mutable struct KalmanFilterState{Z <: Real} <: AbstractFilterState{Z}

    # Filtered and predicted state
    predicted_state_μ::Vector{Z}
    predicted_state_Σ::Matrix{Z}
    filtered_state_μ::Vector{Z}
    filtered_state_Σ::Matrix{Z}

    # Matrices and vectors
    K::Matrix{Z}
    M::Matrix{Z}
    L::Matrix{Z}
    S::Matrix{Z}
    v::Vector{Z}

    # Likelihood
    llk::Z

    function KalmanFilterState(init_state::GaussianStateStochasticProcess{Z},
            n_X::Int, n_Y::Int) where {Z <: Real}
        new{Z}(
            init_state.μ_t,
            init_state.Σ_t,
            zeros(Z, n_X),
            zeros(Z, n_X, n_X),
            zeros(Z, n_X, n_Y),
            zeros(Z, n_X, n_Y),
            zeros(Z, n_X, n_X),
            zeros(Z, n_Y, n_Y),
            zeros(Z, n_Y),
            0.0
        )
    end

    function KalmanFilterState{K}(init_state::GaussianStateStochasticProcess,
        n_X::Int, n_Y::Int) where {K <: Real}
        new{K}(
            init_state.μ_t,
            init_state.Σ_t,
            zeros(K, n_X),
            zeros(K, n_X, n_X),
            zeros(K, n_X, n_Y),
            zeros(K, n_X, n_Y),
            zeros(K, n_X, n_X),
            zeros(K, n_Y, n_Y),
            zeros(K, n_Y),
            0.0
        )
    end
end

mutable struct KalmanFilter{Z <: Real} <: AbstractGaussianDeterministicFilter{Z}
    init_state_x::GaussianStateStochasticProcess{Z}
    state::KalmanFilterState{Z}

    """
    Constructor with gaussian init state.
    """
    function KalmanFilter(init_state::GaussianStateStochasticProcess{Z}, n_X, n_Y) where {Z <: Real}
        new{Z}(init_state, KalmanFilterState(init_state, n_X, n_Y))
    end

    """Constructor with given type K."""
    function KalmanFilter{K}(init_state::GaussianStateStochasticProcess, n_X, n_Y) where {K <: Real}
        new{K}(init_state, KalmanFilterState{K}(init_state, n_X, n_Y))
    end

    """
    Constructor with particle init state.
    """
    function KalmanFilter(init_state::ParticleSwarmState{Z}, n_X, n_Y) where {Z <: Real}
        μ_t = vcat(mean(init_state.particles_state, dims = 2)...)
        Σ_t = var(init_state.particles_state, dims = 2)
        gaussian_init_state = GaussianStateStochasticProcess(init_state.t, μ_t, Σ_t)

        new{Z}(gaussian_init_state, KalmanFilterState(gaussian_init_state, n_X, n_Y))
    end

    function KalmanFilter(model::ForecastingModel{Z, T, S}; type=Z) where {Z <: Real, T <: AbstractStateSpaceSystem, S <: AbstractState{Z}}
        return KalmanFilter{type}(deepcopy(model.current_state), model.system.n_X, model.system.n_Y)
    end
end

mutable struct KalmanFilterOutput{Z <: Real} <: AbstractGaussianFilterOutput{Z}

    # Predicted and filtered states
    predicted_state::TimeSeries{Z, GaussianStateStochasticProcess{Z}}
    filtered_state::TimeSeries{Z, GaussianStateStochasticProcess{Z}}

    K::Vector{Matrix{Z}}
    M::Vector{Matrix{Z}}
    L::Vector{Matrix{Z}}
    S::Vector{Matrix{Z}}
    v::Vector{Vector{Z}}

    llk::Z

    function KalmanFilterOutput(model::ForecastingModel{Z, T, St},
            y_t) where {Z <: Real, T <: AbstractStateSpaceSystem, St <: AbstractState{Z}}
        n_obs = size(y_t, 1)

        t_index = [model.current_state.t + (model.system.dt) * (t - 1)
                   for t in 1:(n_obs + 1)]

        predicted_state = TimeSeries{Z, GaussianStateStochasticProcess{Z}}(
            t_index, n_obs + 1, model.system.n_X)
        filtered_state = TimeSeries{Z, GaussianStateStochasticProcess{Z}}(
            t_index[1:(end - 1)],
            n_obs,
            model.system.n_X
        )

        # Define matrices
        K = Array{Array{Z, 2}, 1}(undef, n_obs)
        M = Array{Array{Z, 2}, 1}(undef, n_obs)
        L = Array{Array{Z, 2}, 1}(undef, n_obs)
        S = Array{Array{Z, 2}, 1}(undef, n_obs)
        v = Array{Array{Z, 1}, 1}(undef, n_obs)

        return new{Z}(predicted_state, filtered_state, K, M, L, S, v, 0.0)
    end
end

function get_filter_output(
        filter_method::KalmanFilter, model::ForecastingModel, observation_data)
    return KalmanFilterOutput(model, observation_data)
end

function get_last_state(filter_output::KalmanFilterOutput)
    return filter_output.predicted_state[end]
end

# General KF
function filtering!(
        filter_output::KalmanFilterOutput{Z},
        sys::GaussianLinearStateSpaceSystem{Z},
        filter_method::KalmanFilter{Z},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        parameters::Vector{Z}
) where {Z <: Real}
    n_obs = size(observation_data, 1)

    # Save initial state
    save_initial_state_in_filter_output!(filter_output, filter_method.state)

    t_step_table = collect(range(
        filter_method.init_state_x.t, length = n_obs, step = sys.dt))

    # n_X = sys.n_X
    # n_Y = sys.n_Y
    # A = zeros(eltype(parameters), n_X, n_X)
    # B = zeros(eltype(parameters), n_X, 1)
    # c = zeros(eltype(parameters), n_X, 1)
    # H = zeros(eltype(parameters), n_Y, n_X)
    # d = zeros(eltype(parameters), n_Y, 1)
    # R = zeros(eltype(parameters), n_X, n_X)
    # Q = zeros(eltype(parameters), n_Y, n_Y)

    @inbounds for (t, t_step) in enumerate(t_step_table)

        # Get current matrix A, B, H and Q
        ex = view(exogenous_data, t, :)
        A = sys.A_t(ex, parameters, t_step)#::AbstractMatrix{Z}
        B = sys.B_t(ex, parameters, t_step)#::AbstractMatrix{Z}
        c = sys.c_t(ex, parameters, t_step)#::AbstractVector{Z}
        H = sys.H_t(ex, parameters, t_step)#::AbstractMatrix{Z}
        d = sys.d_t(ex, parameters, t_step)#::AbstractVector{Z}
        R = sys.R_t(ex, parameters, t_step)#::AbstractMatrix{Z}
        Q = sys.Q_t(ex, parameters, t_step)#::AbstractMatrix{Z}

        update_filter_state!(
            filter_method.state,
            view(observation_data, t, :),
            view(control_data, t, :),
            A,
            B,
            c,
            H,
            d,
            R,
            Q
        )

        save_state_in_filter_output!(filter_output, filter_method.state, t)
    end

    return filter_output
end

# KF for optimization
function filtering!(
        sys::GaussianLinearStateSpaceSystem{Z},
        filter_method::KalmanFilter{D},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        parameters::Vector{D}
) where {Z <: Real, D <: Real}
    n_obs = size(observation_data, 1)

    t_step_table = collect(range(
        filter_method.init_state_x.t, length = n_obs, step = sys.dt))

    @inbounds for (t, t_step) in enumerate(t_step_table)

        # Get current matrix A, B, H and Q
        ex = exogenous_data[t, :]
        A = sys.A_t(ex, parameters, t_step)
        B = sys.B_t(ex, parameters, t_step)
        c = sys.c_t(ex, parameters, t_step)
        H = sys.H_t(ex, parameters, t_step)
        d = sys.d_t(ex, parameters, t_step)
        R = sys.R_t(ex, parameters, t_step)
        Q = sys.Q_t(ex, parameters, t_step)

        update_filter_state!(
            filter_method.state,
            view(observation_data, t, :),
            view(control_data, t, :),
            A,
            B,
            c,
            H,
            d,
            R,
            Q
        )
    end

    return filter_method.state.llk / n_obs
end

function update_filter_state!(kalman_state::KalmanFilterState{Z}, y, u, A, B, c , H, d, R, Q) where {Z <: Real}

    # Check the number of correct observations
    ivar_obs = findall(.!isnan.(y))
    Hivar = @view H[ivar_obs, :]

    # Compute innovations and stuff for predicted and filtered states
    kalman_state.v = view(y, ivar_obs) - (Hivar * kalman_state.predicted_state_μ + view(d, ivar_obs))
    kalman_state.M = kalman_state.predicted_state_Σ * transpose(Hivar)
    kalman_state.S = Hivar * kalman_state.M + view(Q, ivar_obs, ivar_obs)

    inv_S = inv(kalman_state.S)
    M_invS = kalman_state.M * inv_S

    # Update states (Update step)
    kalman_state.filtered_state_μ .= kalman_state.predicted_state_μ +
                                    M_invS * kalman_state.v
    kalman_state.filtered_state_Σ .= kalman_state.predicted_state_Σ -
                                    M_invS * transpose(kalman_state.M)

    # Forecast step
    kalman_state.predicted_state_μ .= A * kalman_state.filtered_state_μ + B * u + c
    kalman_state.predicted_state_Σ .= A * kalman_state.filtered_state_Σ * transpose(A) + R

    # Compute stuff for Kalman smoother
    kalman_state.K = A * M_invS
    kalman_state.L = A - kalman_state.K * Hivar

    # Update likelihood
    if !isempty(ivar_obs)
        kalman_state.llk += -log(2 * pi) / 2 -
                            (1 / 2) * (logdet(kalman_state.S) +
                             kalman_state.v' * inv_S * kalman_state.v)
    end
end

function save_state_in_filter_output!(
        filter_output::KalmanFilterOutput{Z},
        filter_state::KalmanFilterState{Z},
        t::Int
) where {Z <: Real}

    # Save predicted state
    filter_output.predicted_state[t + 1].μ_t .= filter_state.predicted_state_μ
    filter_output.predicted_state[t + 1].Σ_t .= filter_state.predicted_state_Σ

    # Save filtered state
    filter_output.filtered_state[t].μ_t .= filter_state.filtered_state_μ
    filter_output.filtered_state[t].Σ_t .= filter_state.filtered_state_Σ

    # Save matrix values
    filter_output.K[t] = filter_state.K
    filter_output.M[t] = filter_state.M
    filter_output.L[t] = filter_state.L
    filter_output.S[t] = filter_state.S
    filter_output.v[t] = filter_state.v

    # Save likelihood
    filter_output.llk = filter_state.llk
end

function save_initial_state_in_filter_output!(
        filter_output::KalmanFilterOutput{Z},
        filter_state::KalmanFilterState{Z}
) where {Z <: Real}

    # Save initial predicted state
    filter_output.predicted_state[1].μ_t .= filter_state.predicted_state_μ
    filter_output.predicted_state[1].Σ_t .= filter_state.predicted_state_Σ
end
