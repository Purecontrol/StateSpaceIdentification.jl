# export GaussianNonLinearStateSpaceSystem

@doc raw"""

`GaussianNonLinearStateSpaceSystem(M_t::Function, H_t::Function, R_t::Function, Q_t::Function, dt::Float64)`

Definition of the system fonctions ``M_t, H_t, R_t, Q_t`` for nonlinear gaussian state space models with a fixed timestep of dt.

```math
\begin{gather*}
    \begin{aligned}
        \x{t+1} &= M_t (\x{t} , u(t)) + \eta_{t} \quad &\eta_{t} \sim \mathcal{N}(0, R_t)\\
        y_{t}   &=  H_t (\x{t}) + \epsilon_{t} \quad &\epsilon_{t} \sim \mathcal{N}(0, Q_t)\\
    \end{aligned}
\end{gather*}
```

where:

* ``x_t`` is a ``n_X \times 1`` vector
* ``y_t`` is a ``n_Y \times 1`` vector
* ``u_t`` is a ``n_U \times 1`` vector
* ``M_t`` is a ``n_X -> n_X`` function
* ``H_t`` is a ``n_X -> n_Y`` function
* ``R_t`` is a ``n_X \times n_X`` matrix
* ``Q_t`` is a ``n_Y \times n_Y`` matrix
"""
mutable struct GaussianNonLinearStateSpaceSystem <: StateSpaceSystem

    # General components of gaussian non linear state space systems 
    M_t::Function
    H_t::Function
    R_t::Function
    Q_t::Function

    #Size of observation and latent space
    n_X::Int64
    n_Y::Int64

    # Time between two states
    dt::Float64

    function GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt)

        return new(M_t, H_t, R_t, Q_t, n_X, n_Y, dt)
    end

end

function transition(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, control_variables, parameters) 

    return ssm.M_t(current_x, exogenous_variables, control_variables, parameters)

end

function observation(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, parameters) 

    return ssm.H_t(current_x, exogenous_variables, parameters)

end

function forecast(system::GaussianNonLinearStateSpaceSystem , current_state::GaussianStateStochasticProcess, exogenous_variables, control_variables, parameters; n_steps_ahead=1)

    predicted_state = TimeSeries{GaussianStateStochasticProcess}(n_steps_ahead+1, system.n_X)
    predicted_obs = TimeSeries{GaussianStateStochasticProcess}(n_steps_ahead+1, system.n_Y)

    # Define init conditions
    current_obs = system.H_t(current_state.μ_t, exogenous_variables[1, :], parameters)
    predicted_state[1].t = current_state.t
    predicted_state[1].μ_t = current_state.μ_t
    predicted_state[1].σ_t = current_state.σ_t
    predicted_obs[1].t = current_state.t
    predicted_obs[1].μ_t = current_obs
    predicted_obs[1].σ_t = transpose(current_H)*current_state.σ_t*current_H + system.Q_t(exogenous_variables[1, :], parameters)

    @inbounds for step in 2:(n_steps_ahead+1)

        # Define current t_step
        t_step = current_state.t + (step-1)*system.dt

        # Get current matrix A and B
        A = system.A_t(exogenous_variables[step, :], parameters)
        B = system.B_t(exogenous_variables[step, :], parameters)
        H = system.H_t(exogenous_variables[step, :], parameters)
        
        # Update predicted state and covariance
        predicted_state[step].t = t_step
        predicted_state[step].μ_t = A*predicted_state[step-1].μ_t + B*control_variables[step-1, :] + system.c_t(exogenous_variables[step, :], parameters)
        predicted_state[step].σ_t = transpose(A)*predicted_state[step-1].σ_t*A + system.R_t(exogenous_variables[step, :], parameters)
        
        # Update observed state and covariance
        predicted_obs[step].t = t_step
        predicted_obs[step].μ_t = H*predicted_state[step].μ_t + system.d_t(exogenous_variables[step, :], parameters)
        predicted_obs[step].σ_t = transpose(H)*predicted_state[step].σ_t*H + system.Q_t(exogenous_variables[step, :], parameters)


    end

    return predicted_state, predicted_obs

end

function default_filter(model::ForecastingModel{GaussianNonLinearStateSpaceSystem})

    return ParticleFilter(model.current_state, model.system.n_X, model.system.n_Y, 50)

end

