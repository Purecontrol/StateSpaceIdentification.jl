"""

$(TYPEDEF)

$(TYPEDFIELDS)

``ForecastingModel`` is composed of a system (<:``AbstractStateSpaceSystem), a current_state (<:``AbstractState) and 
some parameters. The model uses the current state and the parameters to compute the transition and observation equations
given by the system. 
"""
mutable struct ForecastingModel{
    Z <: Real, T <: AbstractStateSpaceSystem{Z}, S <: AbstractState{Z}}
    """System's structure"""
    system::T
    """Current state of the system"""
    current_state::S
    """Model's parameters"""
    parameters::Vector{Z}

    """Constructor with full arguments."""
    function ForecastingModel(system::T,
            current_state::S,
            parameters::Vector{Z}) where {
            Z <: Real, T <: AbstractStateSpaceSystem, S <: AbstractState{Z}}
        new{Z, T, S}(system, current_state, parameters)
    end
end

"""
$(TYPEDSIGNATURES)

Abstract ``default_filter`` function that has to be defined for all subtypes of AbstractStateSpaceSystem.
"""
@inline default_filter(model::ForecastingModel{Z, T, S}; kwargs...) where {Z <: Real, T <: AbstractStateSpaceSystem{Z}, S <: AbstractStateSpaceSystem{Z}} = error("The function ``default_filter``` has to be defined for subtype of AbstractStateSpaceSystem.")

"""
$(TYPEDSIGNATURES)

Abstract ``default_smoother`` function that has to be defined for all subtypes of AbstractStateSpaceSystem.
"""
@inline default_smoother(model::ForecastingModel{Z, T, S}; kwargs...) where {Z <: Real, T <: AbstractStateSpaceSystem{Z}, S <: AbstractStateSpaceSystem{Z}} = error("The function ``default_smoother``` has to be defined for subtype of AbstractStateSpaceSystem.")