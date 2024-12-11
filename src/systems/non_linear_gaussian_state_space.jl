import Base: convert

"""
Definition of the system fonctions ``M_t, H_t, R_t, Q_t`` for nonlinear gaussian state space models with a fixed timestep of dt.

```math
\\begin{gather*}
    \\begin{aligned}
        \\x{t+1} &= M_t (\\x{t} , u(t)) + \\eta_{t} \\quad &\\eta_{t} \\sim \\mathcal{N}(0, R_t)\\
        y_{t}   &=  H_t (\\x{t}) + \\epsilon_{t} \\quad &\\epsilon_{t} \\sim \\mathcal{N}(0, Q_t)\\
    \\end{aligned}
\\end{gather*}
```
.
$(TYPEDEF)

$(TYPEDFIELDS)
"""
mutable struct GaussianNonLinearStateSpaceSystem <: AbstractNonLinearStateSpaceSystem

    # General components of gaussian non linear state space systems
    """Function ``M_t`` which is a ``n_X -> n_X`` function."""
    M_t::Function
    """Function ``H_t`` which is a ``n_X -> n_Y`` function."""
    H_t::Function
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

    function GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt)
        return new(M_t, H_t, R_t, Q_t, n_X, n_Y, dt)
    end
end

"""
$(TYPEDSIGNATURES)

The ``default_filter`` for ``GaussianLinearStateSpaceSystem`` is the ``ParticleFilter``.
"""
function default_filter(model::ForecastingModel{Z, GaussianNonLinearStateSpaceSystem, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return ParticleFilter(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``default_smoother`` for ``GaussianLinearStateSpaceSystem`` is the ``BackwardSimulationSmoother``.
"""
function default_smoother(model::ForecastingModel{Z, GaussianNonLinearStateSpaceSystem, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return BackwardSimulationSmoother(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``transition`` function for ``GaussianNonLinearStateSpaceSystem`` which is ``y_{t}   &=  H_t (\\x{t})``.
"""
function transition(
        ssm::GaussianNonLinearStateSpaceSystem,
        state_variables::Vector{Z},
        exogenous_variables::Vector{Z},
        control_variables::Vector{Z},
        parameters::Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.M_t(state_variables, exogenous_variables, control_variables, parameters, t)
end

"""
$(TYPEDSIGNATURES)

The ``observation`` function for ``GaussianNonLinearStateSpaceSystem`` which is ``y_{t} &=  H_t \\x{t} + d_t + \\epsilon_{t} \\quad &\\epsilon_{t} \\sim \\mathcal{N}(0, Q_t)``
"""
function observation(
        ssm::GaussianNonLinearStateSpaceSystem,
        state_variables::Vector{Z},
        exogenous_variables::Vector{Z},
        parameters::Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.H_t(state_variables, exogenous_variables, parameters, t)
end

"""
Convert ForecastingModel{GaussianLinearStateSpaceSystem} into ForecastingModel{GaussianNonLinearStateSpaceSystem}.
"""
function Base.convert(
        ::Type{ForecastingModel{Z, GaussianNonLinearStateSpaceSystem, S}},
        model::ForecastingModel{Z, GaussianLinearStateSpaceSystem, S}
) where {Z <: Real, S <: AbstractState{Z}}
    @inline M_t(x, exogenous, u, params, t) = model.system.A_t(exogenous, params, t) * x .+
                                              model.system.B_t(exogenous, params, t) * u .+
                                              model.system.c_t(exogenous, params, t)
    @inline H_t(x, exogenous, params, t) = model.system.H_t(exogenous, params, t) * x .+
                                           model.system.d_t(exogenous, params, t)

    gnlss = GaussianNonLinearStateSpaceSystem(
        M_t,
        H_t,
        model.system.R_t,
        model.system.Q_t,
        model.system.n_X,
        model.system.n_Y,
        model.system.dt
    )

    new_model = ForecastingModel(
        gnlss,
        model.current_state,
        model.parameters
    )

    return new_model
end
