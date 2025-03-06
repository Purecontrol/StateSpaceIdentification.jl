using NearestNeighbors
import Base: convert

#####################################################################################################################
############################################# Local Linear Regression ###############################################
#####################################################################################################################
# TODO : clean a lot the stuff again because typing is not good again !

"""
Local linear regression structure. 
"""
mutable struct LLR
    index_analogs::Any
    analogs::Any
    successors::Any
    tree::Any

    ignored_nodes::Any

    # Hyperparameters
    k::Any
    lag_x::Any
    kernel::Any
    lag_time::Any

    function LLR(
            index_analogs,
            analogs,
            successors,
            tree,
            ignored_nodes,
            k,
            lag_x,
            kernel,
            lag_time
    )
        new(
            index_analogs,
            analogs,
            successors,
            tree,
            ignored_nodes,
            k,
            lag_x,
            kernel,
            lag_time
        )
    end

    function LLR(
            index_analogs,
            analogs,
            successors,
            tree,
            ignored_nodes;
            k = 10,
            lag_x = 5,
            kernel = "rectangular",
            lag_time = 0
    )
        new(
            index_analogs,
            analogs,
            successors,
            tree,
            ignored_nodes,
            k,
            lag_x,
            kernel,
            lag_time
        )
    end

    function LLR(index_analogs, analogs, successors, tree, ignored_nodes, k, lag_x, kernel)
        new(index_analogs, analogs, successors, tree, ignored_nodes, k, lag_x, kernel, 0)
    end
end

@inline function skip(i::Int; visited = Set())::Bool
    i ∈ visited
end

function (llr::LLR)(x, t)

    # Get parameters llr
    k = llr.k
    lag_x = llr.lag_x
    kernel = llr.kernel
    analogs = llr.analogs
    succesors = llr.successors
    tree = llr.tree
    lag_time = llr.lag_time

    n_particules = size(x, 2)
    nb_point_tree = size(tree.data)[1]

    # Set skip nodes
    # if idx != 0 && lag_x != 0
    #     llr.ignored_nodes = Set(max(1, idx-lag_x):min(nb_point_tree, idx+lag_x))
    # else
    #     llr.ignored_nodes = Set([])
    # end
    # llr.ignored_nodes = Set([])
    # if t != 0 && lag_x != 0
    #     for i in 1:nb_point_tree
    #         if (t-lag_x <= llr.index_analogs[i] <= t+lag_x)
    #             push!(llr.ignored_nodes, i)
    #         elseif (lag_time != 0) && (((t%1 <= llr.index_analogs[i]%1 - lag_time) && (t%1 >= (llr.index_analogs[i] + lag_time)%1)) || ((t%1 >= llr.index_analogs[i]%1 + lag_time) && (t%1 <= (1 + llr.index_analogs[i]%1 - lag_time))))
    #             push!(llr.ignored_nodes, i)
    #         end
    #     end
    # end
    # llr.ignored_nodes = Set{Int64}()
    # if t != 0 && lag_x != 0
    #     indexes = llr.index_analogs
    #     ignored_nodes = llr.ignored_nodes
    #     for i in eachindex(indexes)
    #         index = indexes[i]
    #         if (t-lag_x <= index <= t+lag_x) || ((lag_time != 0) && ((index%1 <= t%1 - lag_time && index%1 >= max(t +lag_time, 1)%1) || (index%1 >= t%1 + lag_time && index%1 <= 1 + t - lag_time)))
    #             push!(ignored_nodes, i)
    #         end
    #     end
    # end
    llr.ignored_nodes = Set{Int64}()
    if t != 0 && lag_x != 0
        indexes = llr.index_analogs
        n = length(indexes)
        ignored_nodes = BitVector(undef, n)  # Pré-allocations d'un vecteur booléen
        t_mod = t % 1

        # Condition 1: Vérifier les index dans l'intervalle [t-lag_x, t+lag_x]
        in_range = (t - lag_x .<= indexes) .& (indexes .<= t + lag_x)

        # Condition 2: Vérifier les index modulo dans l'intervalle des temps
        mod_indexes = indexes .% 1
        out_of_mod_range = (
            (mod_indexes .<= t_mod - lag_time) .&
            (mod_indexes .>= max(t + lag_time, 1) % 1)
        ) .| ((mod_indexes .>= t_mod + lag_time) .&
                            (mod_indexes .<= 1 + t - lag_time))

        # Combiner les conditions
        ignored_nodes .= in_range .| (lag_time != 0 && out_of_mod_range)

        # Collecter les indices ignorés
        llr.ignored_nodes = Set(findall(ignored_nodes))
    end

    if length(llr.ignored_nodes) >= nb_point_tree - 2
        error("Lag_x is too high. Decreases it.")
        exit()
    end

    # Check if k is correct
    k = min(nb_point_tree - length(llr.ignored_nodes), k)

    knn_sol = knn(tree, x, k, false, (x) -> skip(x, visited = llr.ignored_nodes))
    knn_x_new = hcat(knn_sol[1]...)
    distance_x_new = hcat(knn_sol[2]...)

    # Reset skip nodes
    llr.ignored_nodes = Set([])

    weights = ones(n_particules, k) / k
    if kernel == "tricube"
        h_m = maximum(distance_x_new, dims = 1)
        weights = transpose((1 .- (distance_x_new ./ h_m) .^ 3) .^ 3)
    end
    weights = weights ./ sum(weights, dims = 2)

    A = vcat([ones(1, k, n_particules), analogs[:, knn_x_new]]...) .*
        permutedims(cat(weights, dims = 3), (3, 2, 1))
    B = succesors[knn_x_new, :] .* cat(weights', dims = 3)
    if size(B, 3) > 1 #TODO : improve after Multiple MISO here but maybe in the futur interesting to do MIMO
        M = cat(map((x, y) -> x' \ y, eachslice(A, dims=3), eachslice(B, dims=2))..., dims=3)
    else
        M = Array{Float64}(undef, size(A, 1), size(B, 3), n_particules)
        @inbounds for i in 1:n_particules
            @inbounds @views M[:, :, i] = A[:, :, i]' \ B[:, i]
        end
    end
    x_new = vcat([ones(1, n_particules), x]...)

    mean_xf = zeros(Float64, size(M, 2), n_particules)
    for i in 1:n_particules
        mean_xf[:, i] = x_new[:, i]' * M[:, :, i]
    end

    return mean_xf
end


#####################################################################################################################
####################################### GaussianNonParametricStateSpaceSystem #######################################
#####################################################################################################################


mutable struct GaussianNonParametricStateSpaceSystem{Z <: Real} <: AbstractNonLinearStateSpaceSystem{Z}

    # General components of gaussian non parametric state space systems
    """Provider ``M_t`` which is a ``n_X -> n_X`` function."""
    M_t::TransitionNonParametricProvider{Z}
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

    # Additional attributes for local linear regression structure
    llrs::Vector{LLR}
    μ::Any
    Σ::Any

    """Constructor with full arguments."""
    function GaussianNonParametricStateSpaceSystem{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs; μ = 0.0, Σ = 1.0) where {Z <: Real}
        return new{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs, μ, Σ)
    end

    """Constructor with Type conversion."""
    function GaussianNonParametricStateSpaceSystem{Z}(M_t::Union{Function, TransitionNonParametricProvider{Z}}, H_t::Union{Function, ObservationNonLinearProvider{Z}}, R_t::Union{MatOrFun, AbstractMatrixProvider{Z}}, Q_t::Union{MatOrFun, AbstractMatrixProvider{Z}}, n_X, n_Y, dt, llrs; μ = 0.0, Σ = 1.0) where {Z <: Real}

            # Convert types
            M_t = isa(M_t, TransitionNonParametricProvider) ? M_t : TransitionNonParametricProvider{Z}(M_t)
            H_t = isa(H_t, ObservationNonLinearProvider) ? H_t : ObservationNonLinearProvider{Z}(H_t)
            R_t = isa(R_t, AbstractMatrixProvider) ? R_t : (isa(R_t, Matrix) ? StaticMatrix{Z}(R_t) : DynamicMatrix{Z}(R_t))
            Q_t = isa(Q_t, AbstractMatrixProvider) ? Q_t : (isa(Q_t, Matrix) ? StaticMatrix{Z}(Q_t) : DynamicMatrix{Z}(Q_t))

        return new{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs, μ, Σ)
    end

end

"""
$(TYPEDSIGNATURES)

The ``default_filter`` for ``GaussianNonParametricStateSpaceSystem`` is the ``ParticleFilter``.
"""
function default_filter(model::ForecastingModel{Z, GaussianNonParametricStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return ParticleFilter(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``default_smoother`` for ``GaussianNonParametricStateSpaceSystem`` is the ``BackwardSimulationSmoother``.
"""
function default_smoother(model::ForecastingModel{Z, GaussianNonParametricStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return BackwardSimulationSmoother(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``transition`` function for ``GaussianNonParametricStateSpaceSystem`` which is ``y_{t}   &=  H_t (\\x{t})``.
"""
function transition(
        ssm::GaussianNonParametricStateSpaceSystem{Z},
        state_variables::AbstractVecOrMat{Z},
        exogenous_variables::AbstractVector{Z},
        control_variables::AbstractVector{Z},
        parameters::AbstractArray,#Vector{Z},
        t::Z
) where {Z <: Real}#, A <: AbstractArray{Z}}
    return ssm.M_t(state_variables, _scale(state_variables, ssm.μ, ssm.Σ), exogenous_variables, control_variables, parameters, ssm.llrs, t)
end


"""
$(TYPEDSIGNATURES)

The ``observation`` function for ``GaussianNonParametricStateSpaceSystem`` which is ``y_{t} &=  H_t \\x{t} + d_t + \\epsilon_{t} \\quad &\\epsilon_{t} \\sim \\mathcal{N}(0, Q_t)``
"""
function observation(
        ssm::GaussianNonParametricStateSpaceSystem{Z},
        state_variables::AbstractVecOrMat{Z},
        exogenous_variables::AbstractVector{Z},
        parameters::AbstractArray,#Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.H_t(state_variables, exogenous_variables, parameters, t)
end


#####################################################################################################################
################################################### Utils for ML ####################################################
#####################################################################################################################

"""

"""
function k_choice(
        ssm::GaussianNonParametricStateSpaceSystem,
        x_t,
        y_t,
        exogenous_variables,
        control_variables,
        time_variables;
        k_list = [5, 10, 15, 20, 25]
)
    n_t = size(x_t, 1)

    L = zeros(length(k_list))
    E = zeros(length(k_list))
    for (index_k, k) in enumerate(k_list)
        for i in size(ssm.llrs, 1)
            ssm.llrs[i].k = k
        end

        try
            err = 0
            for idx in 1:(n_t)
                mean_xf = transition(
                    ssm,
                    x_t[idx, :],
                    exogenous_variables[idx, :],
                    control_variables[idx, :],
                    [],
                    time_variables[idx]
                )
                innov = y_t[idx, :] .- mean_xf
                err += mean((innov) .^ 2)
                E[index_k] = sqrt(err) / n_t
            end
        catch
            E[index_k] = 10^10
        end
    end

    ind_min = argmin(E)

    return k_list[ind_min], E
end
