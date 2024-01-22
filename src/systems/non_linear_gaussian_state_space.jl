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


function transition(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, control_variables, parameters, t) 

    return ssm.M_t(current_x, exogenous_variables, control_variables, parameters, t)

end


function observation(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, parameters, t) 

    return ssm.H_t(current_x, exogenous_variables, parameters, t)

end


function forecast(system::GaussianNonLinearStateSpaceSystem, current_state::AbstractState, exogenous_variables, control_variables, parameters; n_steps_ahead=1, n_particles=nothing)

    if isnothing(n_particles)
        @warn "n_particles not specified. Default is the n_particles of the current state or 100."
        n_particles = isa(current_state, ParticleSwarmState) ? size(current_state.particles_state, 2) : 100
    end

    if isa(current_state, GaussianStateStochasticProcess)
        predicted_particles_swarm = current_state.μ_t .+ sqrt.(current_state.σ_t)*rand(Normal(), system.n_X, n_particles)
        current_state = ParticleSwarmState(n_particles, current_state.t, predicted_particles_swarm)
    else
        n_particles_init_state = size(current_state.particles_state, 2)
        if n_particles_init_state != n_particles

            @warn "The number of particles of the filter is different from the number of particles of the current state."
            selected_idx_particles = sample_discrete((1/n_particles_init_state).*ones(n_particles_init_state), n_particles)[1, :]
            predicted_particles_swarm = current_state.particles_state[:, selected_idx_particles]
            current_state = ParticleSwarmState(n_particles, current_state.t, predicted_particles_swarm)

        end
    end

    # Set up output vectors
    predicted_state = TimeSeries{ParticleSwarmState}(n_steps_ahead+1, system.n_X, [current_state.t + (step-1)*system.dt for step  in 1:(n_steps_ahead+1)])
    predicted_obs = TimeSeries{ParticleSwarmState}(n_steps_ahead+1, system.n_Y, [current_state.t + (step-1)*system.dt for step  in 1:(n_steps_ahead+1)])

    # Define init conditions
    predicted_state[1].particles_state = current_state.particles_state
    predicted_obs[1].particles_state = observation(system, current_state.particles_state, exogenous_variables[1, :], parameters, current_state.t)

    @inbounds for step in 2:(n_steps_ahead+1)

        # Define current t_step
        t_step = current_state.t + (step-1)*system.dt

        R = system.R_t(exogenous_variables[step, :], parameters, t_step)
        Q = system.Q_t(exogenous_variables[step, :], parameters, t_step)
        
        # Update predicted state and covariance
        predicted_state[step].particles_state = transition(system, predicted_state[step-1].particles_state, exogenous_variables[step, :], control_variables[step, :], parameters, t_step)  + rand(MvNormal(R), n_particles)
        
        # Update observed state and covariance
        predicted_obs[step].particles_state = observation(system, predicted_state[step].particles_state, exogenous_variables[step, :], parameters, t_step) + rand(MvNormal(Q), n_particles)

    end

    return predicted_state, predicted_obs

end


function default_filter(model::ForecastingModel{GaussianNonLinearStateSpaceSystem}; kwargs...)

    return ParticleFilter(model; kwargs...)

end

import Base:convert

function convert(::Type{ForecastingModel{GaussianNonLinearStateSpaceSystem}}, model::ForecastingModel{GaussianLinearStateSpaceSystem})

    dt = model.system.dt
    n_Y = model.system.n_Y
    n_X = model.system.n_X

    @inline M_t(x, exogenous, u, params, t) = model.system.A_t(exogenous, params, t)*x .+ model.system.B_t(exogenous, params, t)*u .+ model.system.c_t(exogenous, params, t)
    @inline H_t(x, exogenous, params, t) = model.system.H_t(exogenous, params, t)*x .+ model.system.d_t(exogenous, params, t)

    gnlss = GaussianNonLinearStateSpaceSystem(M_t, H_t, model.system.R_t, model.system.Q_t, n_X, n_Y, dt)

    new_model = ForecastingModel{GaussianNonLinearStateSpaceSystem}(gnlss, model.current_state, model.parameters)

    return new_model

end