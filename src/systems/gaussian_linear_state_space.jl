"""
Definition of the system matrices ``A_t, B_t, c_t, H_t, d_t, R_t, Q_t`` for linear gaussian state space models with a fixed timestep of dt.

```math
\\begin{gather*}
        \\begin{aligned}
            \\x{t+dt} &= A_t \\x{t} + B_t u(t) + c_t + \\eta_{t} \\quad &\\eta_{t} \\sim \\mathcal{N}(0, R_t)\\
            y_{t} &=  H_t \\x{t} + d_t + \\epsilon_{t} \\quad &\\epsilon_{t} \\sim \\mathcal{N}(0, Q_t)\\
        \\end{aligned}
    \\end{gather*}
```
$(TYPEDEF)

$(TYPEDFIELDS)

"""
mutable struct GaussianLinearStateSpaceSystem <: AbstractLinearStateSpaceSystem

    # General components of gaussian linear state space systems 
    """Function ``A_t`` returning a ``n_X \\times n_X`` matrix."""
    A_t::Function
    """Function ``B_t`` returning a ``n_X \\times n_U`` matrix."""
    B_t::Function
    """Function ``c_t`` returning a ``n_X \\times 1`` vector."""
    c_t::Function
    """Function ``H_t`` returning a ``n_Y \\times n_X`` matrix."""
    H_t::Function
    """Function ``d_t`` returning a ``n_Y \\times 1`` vector."""
    d_t::Function
    """Function ``R_t`` returning a ``n_X \\times n_X`` matrix."""
    R_t::Function
    """Function ``Q_t`` returning a ``n_Y \\times n_Y`` matrix."""
    Q_t::Function

    """Number of state variables."""
    n_X::Int
    """Number of observations."""
    n_Y::Int
    """Time between two timesteps in seconds."""
    dt::Real

    """Constructor with full arguments."""
    function GaussianLinearStateSpaceSystem(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt)
        return new(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt)
    end
end

"""
$(TYPEDSIGNATURES)

The ``default_filter`` for ``GaussianLinearStateSpaceSystem`` is the ``KalmanFilter``.
"""
function default_filter(model::ForecastingModel{Z, GaussianLinearStateSpaceSystem, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return KalmanFilter(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``default_smoother`` for ``GaussianLinearStateSpaceSystem`` is the ``KalmanSmoother``.
"""
function default_smoother(model::ForecastingModel{Z, GaussianLinearStateSpaceSystem, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return KalmanSmoother(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``transition`` function for ``GaussianLinearStateSpaceSystem`` which is ``\\x{t+dt} &= A_t \\x{t} + B_t u(t) + c_t``.
"""
function transition(
        ssm::GaussianLinearStateSpaceSystem,
        state_variables::Vector{Z},
        exogenous_variables::Vector{Z},
        control_variables::Vector{Z},
        parameters::Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.A_t(exogenous_variables, parameters, t) * state_variables .+
           ssm.B_t(exogenous_variables, parameters, t) * control_variables .+
           ssm.c_t(exogenous_variables, parameters, t)
end

"""
$(TYPEDSIGNATURES)

The ``observation`` function for ``GaussianLinearStateSpaceSystem`` which is ``y_{t} &=  H_t \\x{t} + d_t``
"""
function observation(
        ssm::GaussianLinearStateSpaceSystem,
        state_variables::Vector{Z},
        exogenous_variables::Vector{Z},
        parameters::Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.H_t(exogenous_variables, parameters, t) * state_variables .+
           ssm.d_t(exogenous_variables, parameters, t)
end
