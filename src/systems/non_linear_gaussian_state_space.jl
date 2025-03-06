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
mutable struct GaussianNonLinearStateSpaceSystem{Z <: Real} <: AbstractNonLinearStateSpaceSystem{Z}

    # General components of gaussian non linear state space systems
    """Provider ``M_t`` which is a ``n_X -> n_X`` function."""
    M_t::TransitionNonLinearProvider{Z}
    """Provider ``H_t`` which is a ``n_X -> n_Y`` function."""
    H_t::ObservationNonLinearProvider{Z}
    """Provider ``R_t`` returning a ``n_X \\times n_X`` matrix."""
    R_t::AbstractMatrixProvider{Z}
    """Provider ``Q_t`` returning a ``n_Y \\times n_Y`` matrix."""
    Q_t::AbstractMatrixProvider{Z}

    """Number of state variables."""
    n_X::Int
    """Number of observations."""
    n_Y::Int
    """Time between two timesteps in seconds."""
    dt::Z

    """Constructor with full arguments."""
    function GaussianNonLinearStateSpaceSystem{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt) where {Z <: Real}
        return new{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt)
    end

    """Constructor with Type conversion."""
    function GaussianNonLinearStateSpaceSystem{Z}(M_t::Union{Function, TransitionNonLinearProvider{Z}}, H_t::Union{Function, ObservationNonLinearProvider{Z}}, R_t::Union{MatOrFun, AbstractMatrixProvider{Z}}, Q_t::Union{MatOrFun, AbstractMatrixProvider{Z}}, n_X, n_Y, dt) where {Z <: Real}

            # Convert types
            M_t = isa(M_t, TransitionNonLinearProvider) ? M_t : TransitionNonLinearProvider{Z}(M_t)
            H_t = isa(H_t, ObservationNonLinearProvider) ? H_t : ObservationNonLinearProvider{Z}(H_t)
            R_t = isa(R_t, AbstractMatrixProvider) ? R_t : (isa(R_t, Matrix) ? StaticMatrix{Z}(R_t) : DynamicMatrix{Z}(R_t))
            Q_t = isa(Q_t, AbstractMatrixProvider) ? Q_t : (isa(Q_t, Matrix) ? StaticMatrix{Z}(Q_t) : DynamicMatrix{Z}(Q_t))

        return new{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt)
    end
end

"""
$(TYPEDSIGNATURES)

The ``default_filter`` for ``GaussianNonLinearStateSpaceSystem`` is the ``ParticleFilter``.
"""
function default_filter(model::ForecastingModel{Z, GaussianNonLinearStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return ParticleFilter(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``default_smoother`` for ``GaussianNonLinearStateSpaceSystem`` is the ``BackwardSimulationSmoother``.
"""
function default_smoother(model::ForecastingModel{Z, GaussianNonLinearStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return BackwardSimulationSmoother(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``transition`` function for ``GaussianNonLinearStateSpaceSystem`` which is ``y_{t}   &=  H_t (\\x{t})``.
"""
function transition(
        ssm::GaussianNonLinearStateSpaceSystem{Z},
        state_variables::AbstractVecOrMat{Z},
        exogenous_variables::AbstractVector{Z},
        control_variables::AbstractVector{Z},
        parameters::AbstractArray,#Vector{Z},
        t::Z
) where {Z <: Real}#, A <: AbstractArray{Z}}
    return ssm.M_t(state_variables, exogenous_variables, control_variables, parameters, t)
end

"""
$(TYPEDSIGNATURES)

The ``observation`` function for ``GaussianNonLinearStateSpaceSystem`` which is ``y_{t} &=  H_t \\x{t} + d_t + \\epsilon_{t} \\quad &\\epsilon_{t} \\sim \\mathcal{N}(0, Q_t)``
"""
function observation(
        ssm::GaussianNonLinearStateSpaceSystem{Z},
        state_variables::AbstractVecOrMat{Z},
        exogenous_variables::AbstractVector{Z},
        parameters::AbstractArray,#Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.H_t(state_variables, exogenous_variables, parameters, t)
end

"""
Convert ForecastingModel{GaussianLinearStateSpaceSystem} into ForecastingModel{GaussianNonLinearStateSpaceSystem}.
"""
function Base.convert(
        ::Type{ForecastingModel{Z, GaussianNonLinearStateSpaceSystem{Z}, S}},
        model::ForecastingModel{Z, GaussianLinearStateSpaceSystem{Z}, S}
) where {Z <: Real, S <: AbstractState{Z}}
    @inline M_t(x, exogenous, u, params, t) = model.system.A_t(exogenous, params, t) * x .+
                                              model.system.B_t(exogenous, params, t) * u .+
                                              model.system.c_t(exogenous, params, t)
                                              
    @inline H_t(x, exogenous, params, t) = model.system.H_t(exogenous, params, t) * x .+
                                           model.system.d_t(exogenous, params, t)

    gnlss = GaussianNonLinearStateSpaceSystem{Z}(
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
