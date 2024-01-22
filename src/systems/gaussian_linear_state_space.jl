# export GaussianLinearStateSpaceSystem

@doc raw"""

`GaussianLinearStateSpaceSystem(A_t::Function, B_t::Function, c_t::Function, H_t::Function, d_t::Function, R_t::Function, Q_t::Function, dt::Float64)`

Definition of the system matrices ``A_t, B_t, c_t, H_t, d_t, R_t, Q_t`` for linear gaussian state space models with a fixed timestep of dt.

```math
\begin{gather*}
        \begin{aligned}
            \x{t+dt} &= A_t \x{t} + B_t u(t) + c_t + \eta_{t} \quad &\eta_{t} \sim \mathcal{N}(0, R_t)\\
            y_{t} &=  H_t \x{t} + d_t + \epsilon_{t} \quad &\epsilon_{t} \sim \mathcal{N}(0, Q_t)\\
        \end{aligned}
    \end{gather*}
```

where:

* ``x_t`` is a ``n_X \times 1`` vector
* ``y_t`` is a ``n_Y \times 1`` vector
* ``u_t`` is a ``n_U \times 1`` vector
* ``A_t`` is a ``n_X \times n_X`` matrix
* ``B_t`` is a ``n_X \times n_U`` matrix
* ``c_t`` is a ``n_X \times 1`` vector
* ``H_t`` is a ``n_Y \times n_X`` matrix
* ``d_t`` is a ``n_Y \times 1`` vector
* ``R_t`` is a ``n_X \times n_X`` matrix
* ``Q_t`` is a ``n_Y \times n_Y`` matrix
"""
mutable struct GaussianLinearStateSpaceSystem <: StateSpaceSystem

    # General components of gaussian linear state space systems 
    A_t::Function
    B_t::Function
    c_t::Function
    H_t::Function
    d_t::Function    
    R_t::Function
    Q_t::Function

    #Size of observation and latent space
    n_X::Int64
    n_Y::Int64

    # Time between two states
    dt::Float64

    function GaussianLinearStateSpaceSystem(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt)

        return new(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt)
    end
end


function forecast(system::GaussianLinearStateSpaceSystem, current_state::AbstractState, exogenous_variables, control_variables, parameters; n_steps_ahead=1)

    if isa(current_state, ParticleSwarmState)
        μ_t = vcat(mean(current_state.particles_state, dims=2)...)
        σ_t = var(current_state.particles_state, dims=2)
        gaussian_init_state = GaussianStateStochasticProcess(current_state.t, μ_t, σ_t)
    else
        gaussian_init_state = current_state
    end

    # Set up output vectors
    predicted_state = TimeSeries{GaussianStateStochasticProcess}(n_steps_ahead+1, system.n_X, [current_state.t + (step-1)*system.dt for step  in 1:(n_steps_ahead+1)])
    predicted_obs = TimeSeries{GaussianStateStochasticProcess}(n_steps_ahead+1, system.n_Y, [current_state.t + (step-1)*system.dt for step  in 1:(n_steps_ahead+1)])

    # Define init conditions
    current_H = system.H_t(exogenous_variables[1, :], parameters, gaussian_init_state.t)
    predicted_state[1].μ_t = gaussian_init_state.μ_t
    predicted_state[1].σ_t = gaussian_init_state.σ_t
    predicted_obs[1].μ_t = current_H*gaussian_init_state.μ_t
    predicted_obs[1].σ_t = transpose(current_H)*gaussian_init_state.σ_t*current_H + system.Q_t(exogenous_variables[1, :], parameters, current_state.t)

    @inbounds for step in 2:(n_steps_ahead+1)

        # Define current t_step
        t_step = gaussian_init_state.t + (step-1)*system.dt

        # Get current matrix A and B
        A = system.A_t(exogenous_variables[step, :], parameters, t_step)
        B = system.B_t(exogenous_variables[step, :], parameters, t_step)
        H = system.H_t(exogenous_variables[step, :], parameters, t_step)
        
        # Update predicted state and covariance
        predicted_state[step].μ_t = A*predicted_state[step-1].μ_t + B*control_variables[step, :] + system.c_t(exogenous_variables[step, :], parameters, t_step)
        predicted_state[step].σ_t = transpose(A)*predicted_state[step-1].σ_t*A + system.R_t(exogenous_variables[step, :], parameters, t_step)
        
        # Update observed state and covariance
        predicted_obs[step].μ_t = H*predicted_state[step].μ_t + system.d_t(exogenous_variables[step, :], parameters, t_step)
        predicted_obs[step].σ_t = transpose(H)*predicted_state[step].σ_t*H + system.Q_t(exogenous_variables[step, :], parameters, t_step)


    end

    return predicted_state, predicted_obs

end


function default_filter(model::ForecastingModel{GaussianLinearStateSpaceSystem})

    return KalmanFilter(model)

end


function default_smoother(model::ForecastingModel{GaussianLinearStateSpaceSystem})

    return KalmanSmoother(model.system.n_X, model.system.n_Y)

end


function transition(ssm::GaussianLinearStateSpaceSystem, current_μ, exogenous_variables, control_variables, parameters, t) 

    return ssm.A_t(exogenous_variables, parameters, t)*current_μ .+ ssm.B_t(exogenous_variables, parameters, t)*control_variables .+ ssm.c_t(exogenous_variables, parameters, t)

end


function observation(ssm::GaussianLinearStateSpaceSystem, current_μ, exogenous_variables, parameters, t) 

    return ssm.H_t(exogenous_variables, parameters, t)*current_μ .+ ssm.d_t(exogenous_variables, parameters, t)

end