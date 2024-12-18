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
mutable struct GaussianLinearStateSpaceSystem{Z <: Real} <:
               AbstractLinearStateSpaceSystem{Z}

    # General components of gaussian linear state space systems 
    """Provider ``A_t`` returning a ``n_X \\times n_X`` matrix."""
    A_t::AbstractMatrixProvider{Z}#Union{MatFunction{Z}, Matrix{Z}}
    """Provider ``B_t`` returning a ``n_X \\times n_U`` matrix."""
    B_t::AbstractMatrixProvider{Z}#Union{MatFunction{Z}, Matrix{Z}}
    """Provider ``c_t`` returning a ``n_X \\times 1`` vector."""
    c_t::AbstractVectorProvider{Z}#Union{MatFunction{Z}, Vector{Z}}
    """Provider ``H_t`` returning a ``n_Y \\times n_X`` matrix."""
    H_t::AbstractMatrixProvider{Z}#Union{MatFunction{Z}, Matrix{Z}}
    """Provider ``d_t`` returning a ``n_Y \\times 1`` vector."""
    d_t::AbstractVectorProvider{Z}#Union{MatFunction{Z}, Vector{Z}}
    """Provider ``R_t`` returning a ``n_X \\times n_X`` matrix."""
    R_t::AbstractMatrixProvider{Z}#Union{MatFunction{Z}, Matrix{Z}}
    """Provider ``Q_t`` returning a ``n_Y \\times n_Y`` matrix."""
    Q_t::AbstractMatrixProvider{Z}#Union{MatFunction{Z}, Matrix{Z}}

    """Number of state variables."""
    n_X::Int
    """Number of observations."""
    n_Y::Int
    """Time between two timesteps in seconds."""
    dt::Z

    """Constructor with full arguments."""
    function GaussianLinearStateSpaceSystem{T}(
            A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt) where {T <: Real}
        return new{T}(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt)
    end

    """Constructor with Type conversion."""
    function GaussianLinearStateSpaceSystem{T}(
            A_t::MatOrFun, B_t::MatOrFun, c_t::VecOrFun, H_t::MatOrFun,
            d_t::VecOrFun, R_t::MatOrFun, Q_t::MatOrFun, n_X, n_Y, dt) where {T <: Real}

            # Convert types
            A_t = isa(A_t, Matrix) ? StaticMatrix{T}(A_t) : DynamicMatrix{T}(A_t)
            B_t = isa(B_t, Matrix) ? StaticMatrix{T}(B_t) : DynamicMatrix{T}(B_t)
            c_t = isa(c_t, Vector) ? StaticVector{T}(c_t) : DynamicVector{T}(c_t)
            H_t = isa(H_t, Matrix) ? StaticMatrix{T}(H_t) : DynamicMatrix{T}(H_t)
            d_t = isa(d_t, Vector) ? StaticVector{T}(d_t) : DynamicVector{T}(d_t)
            R_t = isa(R_t, Matrix) ? StaticMatrix{T}(R_t) : DynamicMatrix{T}(R_t)
            Q_t = isa(Q_t, Matrix) ? StaticMatrix{T}(Q_t) : DynamicMatrix{T}(Q_t)

        return new{T}(A_t, B_t, c_t, H_t, d_t, R_t, Q_t, n_X, n_Y, dt)
    end
end

"""
$(TYPEDSIGNATURES)

The ``default_filter`` for ``GaussianLinearStateSpaceSystem`` is the ``KalmanFilter``.
"""
function default_filter(model::ForecastingModel{Z, GaussianLinearStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return KalmanFilter(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``default_smoother`` for ``GaussianLinearStateSpaceSystem`` is the ``KalmanSmoother``.
"""
function default_smoother(model::ForecastingModel{Z, GaussianLinearStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return KalmanSmoother(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``transition`` function for ``GaussianLinearStateSpaceSystem`` which is ``\\x{t+dt} &= A_t \\x{t} + B_t u(t) + c_t``.
"""
function transition(
        ssm::GaussianLinearStateSpaceSystem{Z},
        state_variables::VecOrMat{Z},
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
        ssm::GaussianLinearStateSpaceSystem{Z},
        state_variables::VecOrMat{Z},
        exogenous_variables::Vector{Z},
        parameters::Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.H_t(exogenous_variables, parameters, t) * state_variables .+
           ssm.d_t(exogenous_variables, parameters, t)
end
