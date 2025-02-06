MatOrFun = Union{AbstractMatrix, Function}
VecOrFun = Union{AbstractVector, Function}
"""
$(TYPEDEF)

AbstractStateSpaceSystem is an abstract type describing a state-space system, which is a global class for models having a transition
equation relating hidden states at time `t` (`x_t`), to the states at time `t+1` (`x_{t+1}`), and an observation equation
relating hidden states `x_t` to observations `y_t`.
"""
abstract type AbstractStateSpaceSystem{Z<:Real} end
"""
$(TYPEDEF)

Linear version of AbstractStateSpaceSystem.
"""
abstract type AbstractLinearStateSpaceSystem{Z<:Real} <: AbstractStateSpaceSystem{Z} end
"""
$(TYPEDEF)

Non-Linear version of AbstractStateSpaceSystem.
"""
abstract type AbstractNonLinearStateSpaceSystem{Z<:Real} <: AbstractStateSpaceSystem{Z} end

"""
$(TYPEDSIGNATURES)

Abstract ``transition`` function that has to be defined for all subtypes of AbstractStateSpaceSystem.
"""
# function transition(
#         ssm::S,
#         state_variables::AbstractArray,
#         exogenous_variables::AbstractArray,
#         control_variables::AbstractArray,
#         parameters::AbstractArray,
#         t::Real
# ) where {S <: AbstractStateSpaceSystem}
#     error("The function ``transition``` has to be defined for subtype of AbstractStateSpaceSystem.")
# end

"""
$(TYPEDSIGNATURES)

Abstract ``observation`` function that has to be defined for all subtypes of AbstractStateSpaceSystem.
"""
# function observation(
#         ssm::S,
#         state_variables::AbstractArray,
#         exogenous_variables::AbstractArray,
#         parameters::AbstractArray,
#         t::Real
# ) where {S <: AbstractStateSpaceSystem}
#     error("The function ``observation``` has to be defined for subtype of AbstractStateSpaceSystem.")
# end