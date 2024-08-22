import Base: getindex
import Base: lastindex
import Base:length, iterate
import Plots.plot
import Plots.plot!
using LinearAlgebra:diag

export TimeSeries
export StateStochasticProcess

using Statistics

abstract type AbstractState end


mutable struct GaussianStateStochasticProcess <: AbstractState
    
    t::Float64
    μ_t::Vector{Float64}
    σ_t::Matrix{Float64}

    function GaussianStateStochasticProcess(t::Real, μ_t::Vector{Float64}, σ_t::Matrix{Float64})

        new(t, μ_t, σ_t)

    end

    function GaussianStateStochasticProcess(k::Integer)

        new(0.0, zeros(Float64, k), zeros(Float64, k, k))

    end

    function GaussianStateStochasticProcess(k::Integer, t::Float64)

        new(t, zeros(Float64, k), zeros(Float64, k, k))

    end

end

function getindex(state::GaussianStateStochasticProcess, i::Int)
    new_state = deepcopy(state)
    new_state.μ_t = new_state.μ_t[i:i]
    new_state.σ_t = new_state.σ_t[i:i, i:i]
    return new_state
end

function getindex(state::GaussianStateStochasticProcess, u::UnitRange{Int64})
    new_state = deepcopy(state)
    new_state.μ_t = new_state.μ_t[u]
    new_state.σ_t = new_state.σ_t[u, u]
    return new_state
end

function getindex(state::GaussianStateStochasticProcess, u::Vector{Int64})
    u = unique(u)
    new_state = deepcopy(state)
    new_state.μ_t = new_state.μ_t[u]
    new_state.σ_t = new_state.σ_t[u, u]
    return new_state
end

mutable struct ParticleSwarmState <: AbstractState
    
    n_particles::Int64
    t::Float64
    particles_state::Array{Float64, 2}

    function ParticleSwarmState(k::Integer; n_particles::Int64=10)

        new(n_particles, 0.0, zeros(Float64, k, n_particles))

    end

    function ParticleSwarmState(k::Integer, t::Float64; n_particles::Int64=10)

        new(n_particles, t, zeros(Float64, k, n_particles))

    end

    function ParticleSwarmState(n_particles::Int64, t::Float64, particles_state::Array{Float64, 2})

        new(n_particles, t, particles_state)

    end

end

function getindex(state::ParticleSwarmState, i::Int)
    new_state = deepcopy(state)
    new_state.particles_state = new_state.particles_state[i:i, :]
    return new_state
end

function getindex(state::ParticleSwarmState, u::UnitRange{Int64})
    new_state = deepcopy(state)
    new_state.particles_state = new_state.particles_state[u, :]
    return new_state
end

function getindex(state::ParticleSwarmState, u::Vector{Int64})
    u = unique(u)
    new_state = deepcopy(state)
    new_state.particles_state = new_state.particles_state[u, :]
    return new_state
end

struct TimeSeries{T <: AbstractState}

    n_t::Integer
    n_state::Integer

    state::Vector{T}

    function TimeSeries{T}(n_t::Integer, n_state::Integer; kwargs...) where {T <: AbstractState}

        time = zeros(Float64, n_t)
        state = [T(n_state; kwargs...) for i = 1:n_t]

        new{T}(n_t, n_state, state)

    end

    function TimeSeries{T}(n_t::Integer, n_state::Integer, t_index::Array{Float64, 1}; kwargs...) where {T <: AbstractState}

        time = zeros(Float64, n_t)
        state = [T(n_state, t_index[i]; kwargs...) for i = 1:n_t]

        new{T}(n_t, n_state, state)

    end

    function TimeSeries{T}(n_t::Integer, n_state::Integer, state::Vector{T}) where {T <: AbstractState}

        new{T}(n_t, n_state, state)

    end

end


function getindex(t::TimeSeries, i::Int)
    t.state[i]
end

function getindex(t::TimeSeries{T}, u::UnitRange{Int64}) where {T <: AbstractState}
    TimeSeries{T}(length(u), t.n_state, t.state[u])
end

function getindex(t::TimeSeries{T}, u::Vector{Int64}) where {T <: AbstractState}
    TimeSeries{T}(length(u), t.n_state, t.state[u])
end

function length(t::TimeSeries)
    return t.n_t
end

function iterate(t::TimeSeries)
        return t[1], 2
end

function iterate(t::TimeSeries, i)

    if i <= length(t)
        return t[i], i+1
    else
        return nothing
    end
end

function lastindex(t::TimeSeries)
    t.n_t
end


function plot(t::TimeSeries{GaussianStateStochasticProcess}; label="", kwargs...)
    mean_process = vcat([t[i].μ_t' for i in 1:t.n_t]...)
    var_process = vcat([diag(t[i].σ_t)' for i in 1:t.n_t]...)
    t_index = vcat([t[i].t for i in 1:t.n_t]...)
    if label != ""
        plot(t_index, mean_process - 1.96*sqrt.(var_process), fillrange = mean_process + 1.96*sqrt.(var_process), alpha=0.3, label=hcat("CI 95 % ".*label...); kwargs...)
        plot!(t_index, mean_process, label=hcat("Mean ".*label...); kwargs...)
    else
        plot(t_index, mean_process - 1.96*sqrt.(var_process), fillrange = mean_process + 1.96*sqrt.(var_process), alpha=0.3; kwargs...)
        plot!(t_index, mean_process; kwargs...)
    end
end


function plot!(t::TimeSeries{GaussianStateStochasticProcess}; label="", kwargs...)
    mean_process = vcat([t[i].μ_t' for i in 1:t.n_t]...)
    var_process = vcat([diag(t[i].σ_t)' for i in 1:t.n_t]...)
    t_index = vcat([t[i].t for i in 1:t.n_t]...)
    if label != ""
        plot!(t_index, mean_process - 1.96*sqrt.(var_process), fillrange = mean_process + 1.96*sqrt.(var_process), alpha=0.3, label=hcat("CI 95 % ".*label...); kwargs...)
        plot!(t_index, mean_process, label=hcat("Mean ".*label...); kwargs...)
    else
        plot!(t_index, mean_process - 1.96*sqrt.(var_process), fillrange = mean_process + 1.96*sqrt.(var_process), alpha=0.3; kwargs...)
        plot!(t_index, mean_process; kwargs...)
    end
end


function plot(t::TimeSeries{ParticleSwarmState}; label="", index = 1:t.n_state, kwargs...)
    mean_process = hcat([[quantile(t[i].particles_state[j, :], 0.5) for j in 1:t.n_state] for i in 1:t.n_t]...)'
    q_low = hcat([[quantile(t[i].particles_state[j, :], 0.025) for j in 1:t.n_state] for i in 1:t.n_t]...)'
    q_high = hcat([[quantile(t[i].particles_state[j, :], 0.975) for j in 1:t.n_state] for i in 1:t.n_t]...)'
    t_index = vcat([t[i].t for i in 1:t.n_t]...)

    if label != ""
        plot(t_index, q_low[:, index], fillrange = q_high[:, index], alpha=0.3, label = hcat("CI 95 % ".*label...); kwargs...)
        plot!(t_index, mean_process[:, index], label = hcat("Mean ".*label...); kwargs...)
    else
        plot(t_index, q_low[:, index], fillrange = q_high[:, index], alpha=0.3; kwargs...)
        plot!(t_index, mean_process[:, index]; kwargs...)
    end
end


function plot!(t::TimeSeries{ParticleSwarmState}; label="", index = 1:t.n_state, kwargs...)
    mean_process = hcat([[quantile(t[i].particles_state[j, :], 0.5) for j in 1:t.n_state] for i in 1:t.n_t]...)'
    q_low = hcat([[quantile(t[i].particles_state[j, :], 0.025) for j in 1:t.n_state] for i in 1:t.n_t]...)'
    q_high = hcat([[quantile(t[i].particles_state[j, :], 0.975) for j in 1:t.n_state] for i in 1:t.n_t]...)'
    t_index = vcat([t[i].t for i in 1:t.n_t]...)

    if label != ""
        plot!(t_index, q_low[:, index], fillrange = q_high[:, index], alpha=0.3, label = hcat("CI 95 % ".*label...); kwargs...)
        plot!(t_index, mean_process[:, index], label = hcat("Mean ".*label...); kwargs...)
    else
        plot!(t_index, q_low[:, index], fillrange = q_high[:, index], alpha=0.3; kwargs...)
        plot!(t_index, mean_process[:, index]; kwargs...)
    end
end


function _base_rmse(x_true, x_pred)

    return vec(mean((x_true - x_pred).^2, dims=2))

end


function rmse(x_true, x_pred::TimeSeries{GaussianStateStochasticProcess})

    return _base_rmse(x_true, hcat(map(x-> x.μ_t, x_pred)...) )

end


function rmse(x_true, x_pred::TimeSeries{ParticleSwarmState})

    return _base_rmse(x_true, hcat(map(x-> median(x.particles_state, dims=2), x_pred)...) )

end


function _get_confidence_interval(x_pred::TimeSeries{GaussianStateStochasticProcess}; α=0.05)
    x_mean = hcat(map(x-> x.μ_t, x_pred)...)
    x_std = hcat(map(x-> sqrt.(diag(x.σ_t)), x_pred)...)
    q_α = quantile(Normal(0, 1), 1-(α/2))
    return x_mean - q_α.*x_std, x_mean + q_α.*x_std
end


function _get_confidence_interval(x_pred::TimeSeries{ParticleSwarmState}; α=0.05)
    x_low_pred = hcat(map(x-> mapslices(y-> quantile(y, α/2), x.particles_state, dims=2), x_pred)...)
    x_high_pred = hcat(map(x-> mapslices(y-> quantile(y, 1-(α/2)), x.particles_state, dims=2), x_pred)...)
    return x_low_pred, x_high_pred
end


function coverage_probability(x_true, x_pred::Union{TimeSeries{GaussianStateStochasticProcess}, TimeSeries{ParticleSwarmState}}; α=0.05)
    x_low_pred, x_high_pred = _get_confidence_interval(x_pred; α=α)
    return vec(mean(x_low_pred .< x_true .< x_high_pred, dims=2))
end


function average_width(x_pred::Union{TimeSeries{GaussianStateStochasticProcess}, TimeSeries{ParticleSwarmState}}; α=0.05)
    x_low_pred, x_high_pred = _get_confidence_interval(x_pred; α=α)
    return vec(mean(x_high_pred - x_low_pred, dims=2))
end
