import Base: getindex, iterate, lastindex, length, size, convert, eltype

using Distributions
using LinearAlgebra: diag
using RecipesBase
using Statistics
using DocStringExtensions

export TimeSeries

DEFAULT_REAL_TYPE = Float64

"""
The `AbstractState` type serves as the parent type for all state implementations within this package.

$(TYPEDEF)

A state is characterized by two key elements:
  - A time representing its temporal context.
  - A description of the state, whose specifics depend on the particular state type.
"""
abstract type AbstractState{T <: Real} end

######################################################################################
# GaussianStateStochasticProcess
######################################################################################

"""
A subtype of `AbstractState` defined for Multivariate Gaussian Distribution representation of the state.
In other words, the state is defined by its mean ``μ_t`` and its covariance matrix ``Σ_t``.

$(TYPEDEF)

$(TYPEDFIELDS)

# Examples
```jldoctest
julia> t = 0.0;
julia> μ_t = [1.0];
julia> Σ_t = [2.0;;];
julia> state = GaussianStateStochasticProcess(t, μ_t, Σ_t)
GaussianStateStochasticProcess(0.0, [1.0], [2.0;;])
```

$(METHODLIST)
"""
mutable struct GaussianStateStochasticProcess{T <: Real} <: AbstractState{T}
    """Time of the state."""
    t::T
    """Mean of the stochastic process."""
    μ_t::Vector{T}
    """Variance of the stochastic process."""
    Σ_t::Matrix{T}

    """Constructor with full arguments."""
    function GaussianStateStochasticProcess(
            t::T,
            μ_t::Vector{T},
            Σ_t::Matrix{T}
    ) where {T <: Real}
        return new{T}(t, μ_t, Σ_t)
    end

    """Constructor with type conversion"""
    function GaussianStateStochasticProcess{Z}(
            t::T,
            μ_t::Vector{<:Real},
            Σ_t::Matrix{<:Real}
    ) where {Z <: Real, T <: Real}
        return new{Z}(t, μ_t, Σ_t)
    end

    """Constructor with different input real type."""
    function GaussianStateStochasticProcess(
            t::T,
            μ_t::Vector{<:Real},
            Σ_t::Matrix{<:Real}
    ) where {T <: Real}
        return new{Z}(t, μ_t, Σ_t)
    end

    """Constructor with number of states."""
    function GaussianStateStochasticProcess{Z}(k::Integer; kwargs...) where {Z <: Real}
        return new{Z}(
            0.0, zeros(Z, k), zeros(Z, k, k))
    end

    """Constructor with number of states and timestep value."""
    function GaussianStateStochasticProcess{Z}(
            t::Real, k::Integer; kwargs...) where {Z <: Real}
        return new{Z}(
            t, zeros(Z, k), zeros(Z, k, k))
    end
end

# Simplified constructors
@inline GaussianStateStochasticProcess(k::Integer; kwargs...) = GaussianStateStochasticProcess{DEFAULT_REAL_TYPE}(
    k; kwargs...)
@inline GaussianStateStochasticProcess(t::Real, k::Integer; kwargs...) = GaussianStateStochasticProcess{DEFAULT_REAL_TYPE}(
    t, k; kwargs...)

"""
Parametric type conversion of GaussianStateStochasticProcess.
"""
function Base.convert(::Type{GaussianStateStochasticProcess{Z1}},
        gssp::GaussianStateStochasticProcess{Z2}) where {Z1 <: Real, Z2 <: Real}
    return GaussianStateStochasticProcess{Z1}(gssp.t, gssp.μ_t, gssp.Σ_t)
end

"""
Indexing GaussianStateStochasticProcess with Int.
"""
function Base.getindex(state::GaussianStateStochasticProcess, i::Int)
    new_state = deepcopy(state)
    new_state.μ_t = new_state.μ_t[i:i]
    new_state.Σ_t = new_state.Σ_t[i:i, i:i]
    return new_state
end

"""
Indexing GaussianStateStochasticProcess with Supported Index.
"""
function Base.getindex(state::GaussianStateStochasticProcess, I...)
    new_state = deepcopy(state)
    new_state.μ_t = new_state.μ_t[I...]
    new_state.Σ_t = new_state.Σ_t[I..., I...]
    return new_state
end

######################################################################################
# ParticleSwarmState
######################################################################################

"""
A subtype of `AbstractState` defined for non-parametric Distribution representation of the state.
In other words, the state is defined by a swarm of particles ``particles_state``.

$(TYPEDEF)

$(TYPEDFIELDS)
"""
mutable struct ParticleSwarmState{T <: Real} <: AbstractState{T}
    """Time of the state."""
    t::T
    """Particle's swarm representing the distribution of the state."""
    particles_state::Matrix{T}

    """Constructor with full arguments."""
    function ParticleSwarmState(
            t::T,
            particles_state::Matrix{T}
    ) where {T <: Real}
        return new{T}(t, particles_state)
    end

    """Constructor with type conversion."""
    function ParticleSwarmState{Z}(
            t::T,
            particles_state::Matrix{T}
    ) where {Z <: Real, T <: Real}
        return new{Z}(t, particles_state)
    end

    """Constructor with number of states."""
    function ParticleSwarmState{Z}(
            k::Integer; n_particles::Integer = 10, kwargs...) where {Z <: Real}
        return new{Z}(0.0, zeros(Z, k, n_particles))
    end

    """Constructor with number of states and timestep value."""
    function ParticleSwarmState{Z}(
        t::Real, k::Integer; n_particles::Integer = 10, kwargs...) where {Z <: Real}
        return new{Z}(t, zeros(Z, k, n_particles))
    end
end

# Simplified constructors
@inline ParticleSwarmState(k::Integer; kwargs...) = ParticleSwarmState{DEFAULT_REAL_TYPE}(
    k; kwargs...)
@inline ParticleSwarmState(t::Real, k::Integer; kwargs...) = ParticleSwarmState{DEFAULT_REAL_TYPE}(
    t, k; kwargs...)

"""
Parametric type conversion of ParticleSwarmState.
"""
function Base.convert(::Type{ParticleSwarmState{Z1}},
        p::ParticleSwarmState{Z2}) where {Z1 <: Real, Z2 <: Real}
    return ParticleSwarmState{Z1}(p.t, p.particles_state)
end

"""
Indexing ParticleSwarmState with Int.
"""
function Base.getindex(state::ParticleSwarmState, i::Int)
    new_state = deepcopy(state)
    new_state.particles_state = new_state.particles_state[i:i, :]
    return new_state
end

"""
Indexing ParticleSwarmState with Supported Index.
"""
function Base.getindex(state::ParticleSwarmState, I...)
    new_state = deepcopy(state)
    new_state.particles_state = new_state.particles_state[I..., :]
    return new_state
end

function Base.size(state::ParticleSwarmState; I...)
    return size(state.particles_state; I...)
end
Base.size(state::ParticleSwarmState{T}, d) where {T} = d::Integer <= 2 ? size(state)[d] : 1

"""
Get the number of particles in the swarm with length
"""
function Base.length(state::ParticleSwarmState)
    return size(state.particles_state, 2)
end

######################################################################################
# TimeSeries
######################################################################################
"""
A collection of subtypes of `AbstractState` stored in a Vector. This stucture represents
a timeseries.

$(TYPEDEF)

$(TYPEDFIELDS)
"""
struct TimeSeries{Z <: Real, T <: AbstractState{Z}}
    """Vector of states."""
    values::Vector{T}

    """Constructor with full arguments."""
    function TimeSeries(
            values::Vector{T}
    ) where {Z <: Real, T <: AbstractState{Z}}
        return new{Z, T}(values)
    end

    """Constructor with type conversion."""
    function TimeSeries{Z1}(values::Vector{T}) where {Z1 <: Real, T <: AbstractState}
        converted_values = convert.(Base.typename(eltype(values)).wrapper{Z1}, values)
        return new{Z1, eltype(converted_values)}(converted_values)
    end

    """Constructor with number of states and number of timesteps."""
    function TimeSeries{Z, T}(
            n_t::Integer,
            n_state::Integer;
            kwargs...
    ) where {Z <: Real, T <: AbstractState{Z}}
        return new{Z, T}(repeat([T(n_state; kwargs...)], n_t))
    end

    """Constructor with number of states and value of timesteps."""
    function TimeSeries{Z2, T}(
            t_index::Vector{<:Real},
            n_t::Integer,
            n_state::Integer;
            kwargs...
    ) where {Z2 <: Real, T <: AbstractState{Z2}}
        return new{Z2, T}([T(t_index[i], n_state; kwargs...) for i in 1:n_t])
    end
end

# Simplified constructors
@inline TimeSeries(T::DataType, n_t::Integer, n_state::Integer; kwargs...) = TimeSeries{
    T.parameters[1], T}(n_t, n_state; kwargs...)
@inline TimeSeries(T::DataType, t_index::Vector{<:Real}, n_t::Integer, n_state::Integer; kwargs...) = TimeSeries{
    T.parameters[1], T}(t_index, n_t, n_state; kwargs...)
@inline TimeSeries(T::UnionAll, n_t::Integer, n_state::Integer; kwargs...) = TimeSeries{
    DEFAULT_REAL_TYPE, T{DEFAULT_REAL_TYPE}}(n_t, n_state; kwargs...)
@inline TimeSeries(T::UnionAll, t_index::Vector{<:Real}, n_t::Integer, n_state::Integer; kwargs...) = TimeSeries{
    DEFAULT_REAL_TYPE, T{DEFAULT_REAL_TYPE}}(t_index, n_t, n_state; kwargs...)

"""
Parametric type conversion of TimeSeries.
"""
@inline Base.convert(::Type{TimeSeries{Z1, T1}}, ts::TimeSeries{Z2, T2}) where {Z1 <: Real, T1 <: AbstractState{Z1}, Z2 <: Real, T2 <: AbstractState{Z2}} = TimeSeries{Z1}(ts.values)

"""
Indexing TimeSeries with Int.
"""
@inline getindex(ts::TimeSeries, i::Int) = return ts.values[i]

"""
Indexing TimeSeries with Supported Index.
"""
@inline Base.getindex(ts::TimeSeries, I...) = TimeSeries(ts.values[I...])

"""
Define length of TimeSeries.
"""
@inline Base.length(ts::TimeSeries) = length(ts.values)

"""
Define iterators of TimeSeries
"""
@inline Base.iterate(ts::TimeSeries) = (ts[1], 2)
@inline Base.iterate(ts::TimeSeries, i) = i <= length(ts) ? (ts[i], i + 1) : nothing
@inline Base.eltype(ts::TimeSeries) = typeof(ts).parameters[2]
@inline Base.lastindex(ts::TimeSeries) = length(ts)

######################################################################################
# Plot's recipe
######################################################################################

"""
Plot's recipe for TimeSeries of GaussianStateStochasticProcess.
"""
@recipe function plot(ts::TimeSeries{Z, GaussianStateStochasticProcess{Z}};
        label = "", ic = 0.95, x_date=nothing) where {Z <: Real}
    dist = Normal(0, 1)
    mean_process = stack(map(t -> t.μ_t, ts), dims = 1)
    var_process = stack(map(t -> diag(t.Σ_t), ts), dims = 1)
    if isnothing(x_date)
        t_index = stack(map(t -> t.t, ts), dims = 1)
    else
        t_index = x_date
    end

    mean_label = isempty(label) ? "" : hcat(vec(["Mean "] .* label)...)
    ci_label = isempty(label) ? "" : hcat(vec(["CI $(Int(ic * 100))% "] .* label)...)

    @series begin
        alpha --> 0.1
        label := ci_label
        fillrange := mean_process + quantile(dist, ic + (1 - ic) / 2) * sqrt.(var_process)
        t_index, mean_process + quantile(dist, (1 - ic) / 2) * sqrt.(var_process)
    end
    @series begin
        label := mean_label
        t_index, mean_process
    end
end

"""
Plot's recipe for TimeSeries of ParticleSwarmState.
"""
@recipe function plot(
        ts::TimeSeries{Z, ParticleSwarmState{Z}};
        label = "",
        index = 1:(size(ts[1], 1)),
        ic = 0.95,
        quantile_tab = nothing, 
        x_date=nothing
) where {Z <: Real}
    mean_process = vcat(map(s -> mean(s.particles_state', dims = 1), ts)...)
    q_low = hcat(map(
        s -> [quantile(i, (1 - ic) / 2) for i in eachrow(s.particles_state)], ts)...)'
    q_high = hcat(map(
        s -> [quantile(i, ic + (1 - ic) / 2) for i in eachrow(s.particles_state)], ts)...)'
    
    if isnothing(x_date)
        t_index = stack(map(t -> t.t, ts), dims = 1)
    else
        t_index = x_date
    end

    mean_label = isempty(label) ? "" : hcat(vec(["Mean "] .* label)...)
    ci_label = isempty(label) ? "" : hcat(vec(["CI $(Int(ic * 100))% "] .* label)...)

    @series begin
        alpha --> 0.1
        label := ci_label
        fillrange := q_high[:, index]
        t_index, q_low[:, index]
    end
    @series begin
        label := mean_label
        t_index, mean_process[:, index]
    end

    if !isnothing(quantile_tab)
        for q in quantile_tab
            q_values = hcat(map(
                s -> [quantile(i, q) for i in eachrow(s.particles_state)], ts)...)'
            q_label = isempty(label) ? "" : hcat(vec(["Q$q "] .* label)...)
            @series begin
                label := q_label
                t_index, q_values[:, index]
            end
        end
    end
end

######################################################################################
# TimeSeries metrics
######################################################################################

"""
Return the Root Mean Squared Error (RMSE) between two vector ``x_true`` and ``x_pred``.
"""
function _base_rmse(x_true, x_pred)
    return vec(mean((x_true - x_pred) .^ 2, dims = 2))
end

"""
Return the Root Mean Squared Error (RMSE) between the vector ``x_true`` and the TimeSeries{GaussianStateStochasticProcess} prediction ``x_pred``.
"""
function rmse(
        x_true, x_pred::TimeSeries{Z, GaussianStateStochasticProcess{Z}}) where {Z <: Real}
    return _base_rmse(x_true, hcat(map(x -> x.μ_t, x_pred)...))
end

"""
Return the Root Mean Squared Error (RMSE) between the vector ``x_true`` and the TimeSeries{ParticleSwarmState} prediction ``x_pred``.
"""
function rmse(x_true, x_pred::TimeSeries{Z, ParticleSwarmState{Z}}) where {Z <: Real}
    return _base_rmse(
        x_true,
        hcat(map(x -> mean(x.particles_state, dims = 2), x_pred)...)
    )
end

"""
Return the confidence interval of a given confidence value ``1-α`` for a TimeSeries{GaussianStateStochasticProcess} prediction ``x_pred``.
"""
function _get_confidence_interval(
        x_pred::TimeSeries{Z, GaussianStateStochasticProcess{Z}};
        α = 0.05
) where {Z <: Real}
    x_mean = hcat(map(x -> x.μ_t, x_pred)...)
    x_std = hcat(map(x -> sqrt.(diag(x.Σ_t)), x_pred)...)
    q_α = quantile(Normal(0, 1), 1 - (α / 2))
    return x_mean - q_α .* x_std, x_mean + q_α .* x_std
end

"""
Return the confidence interval of a given confidence value ``1-α`` for a TimeSeries{ParticleSwarmState} prediction ``x_pred``.
"""
function _get_confidence_interval(
        x_pred::TimeSeries{Z, ParticleSwarmState{Z}}; α = 0.05) where {Z <: Real}
    x_low_pred = hcat(
        map(
        x -> mapslices(y -> quantile(y, α / 2), x.particles_state, dims = 2),
        x_pred
    )...,
    )
    x_high_pred = hcat(
        map(
        x -> mapslices(y -> quantile(y, 1 - (α / 2)), x.particles_state, dims = 2),
        x_pred
    )...,
    )
    return x_low_pred, x_high_pred
end

"""
Return the coverage probability of a given confidence interval of a given confidence value ``1-α`` between the vector ``x_true`` and the TimeSeries prediction ``x_pred``.
"""
function coverage_probability(
        x_true,
        x_pred::TimeSeries;
        α = 0.05
)
    x_low_pred, x_high_pred = _get_confidence_interval(x_pred; α = α)
    return vec(mean(x_low_pred .< x_true .< x_high_pred, dims = 2))
end

"""
Return the average width of a given confidence interval of a given confidence value ``1-α`` of the TimeSeries prediction ``x_pred``.
"""
function average_width(
        x_pred::TimeSeries;
        α = 0.05
)
    x_low_pred, x_high_pred = _get_confidence_interval(x_pred; α = α)
    return vec(mean(x_high_pred - x_low_pred, dims = 2))
end
