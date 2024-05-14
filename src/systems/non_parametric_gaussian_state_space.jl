using NearestNeighbors


mutable struct LLR

    index_analogs
    analogs
    successors
    tree

    ignored_nodes

    # Hyperparameters
    k
    lag_x
    kernel

    function LLR(index_analogs, analogs, successors, tree, ignored_nodes, k, lag_x, kernel)

        new(index_analogs, analogs, successors, tree, ignored_nodes, k, lag_x, kernel)

    end

    function LLR(index_analogs, analogs, successors, tree, ignored_nodes; k=10, lag_x=5, kernel="rectangular")

        new(index_analogs, analogs, successors, tree, ignored_nodes, k, lag_x, kernel)

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

    n_particules = size(x, 2)
    nb_point_tree = size(tree.data)[1]

    # Set skip nodes
    # if idx != 0 && lag_x != 0
    #     llr.ignored_nodes = Set(max(1, idx-lag_x):min(nb_point_tree, idx+lag_x))
    # else
    #     llr.ignored_nodes = Set([])
    # end
    llr.ignored_nodes = Set([])
    if t != 0 && lag_x != 0
        for i in 1:nb_point_tree
            if (t-lag_x <= llr.index_analogs[i] <= t+lag_x)
                push!(llr.ignored_nodes, i)
            end
        end
    end
    # println(llr.ignored_nodes)
    # @error "help me !!!"

    if length(llr.ignored_nodes) >= nb_point_tree - 2
        error("Lag_x is too high. Decreases it.")
        exit()
    end

    # Check if k is correct
    k = min(nb_point_tree - length(llr.ignored_nodes), k)

    knn_sol = knn(tree, x, k, false, (x) -> skip(x, visited=llr.ignored_nodes))
    knn_x_new = hcat(knn_sol[1]...)
    distance_x_new = hcat(knn_sol[2]...)

    # Reset skip nodes
    llr.ignored_nodes = Set([])

    weights = ones(n_particules,k)/k
    if kernel == "tricube"
        h_m = maximum(distance_x_new, dims=1)
        weights = transpose((1 .- (distance_x_new./h_m).^3).^3)
    end
    weights = weights./sum(weights, dims = 2)

    A = vcat([ones(1, k, n_particules), analogs[:, knn_x_new]]...) .* permutedims(cat(weights, dims=3), (3, 2, 1))
    B = succesors[knn_x_new, :] .* cat(weights', dims=3)

    M = hcat(map((x, y) -> x' \ y, eachslice(A, dims=3), eachslice(B, dims=2))...)

    x_new = vcat([ones(1, n_particules), x]...)

    mean_xf = sum(M .* x_new, dims = 1)
    
    return mean_xf

end


mutable struct GaussianNonParametricStateSpaceSystem <: StateSpaceSystem

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

    # LLR
    llrs::Array{LLR}
    μ
    σ

    function GaussianNonParametricStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs; μ=0.0, σ=1.0)

        return new(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs, μ, σ)
    end

end


function transition(ssm::GaussianNonParametricStateSpaceSystem, current_x, exogenous_variables, control_variables, parameters, t) 

    return ssm.M_t(current_x, scale(current_x, ssm.μ, ssm.σ), exogenous_variables, control_variables, ssm.llrs, t)

end


function observation(ssm::GaussianNonParametricStateSpaceSystem, current_x, exogenous_variables, parameters, t) 

    return ssm.H_t(current_x, exogenous_variables, parameters, t)

end


function forecast(system::GaussianNonParametricStateSpaceSystem, current_state::AbstractState, exogenous_variables, control_variables, parameters; n_steps_ahead=1)

    @error "Not implemented yet !"

end

function default_filter(model::ForecastingModel{GaussianNonParametricStateSpaceSystem}; kwargs...)

    return ParticleFilter(model; kwargs...)

end


function k_choice(ssm::GaussianNonParametricStateSpaceSystem, x_t, y_t, exogenous_variables, control_variables, time_variables; k_list = [5, 10, 15, 20, 25])

    n_t = size(x_t, 1)

    L = zeros(length(k_list))
    E = zeros(length(k_list))
    for (index_k, k) in enumerate(k_list)

        for i in size(ssm.llrs, 1)
            ssm.llrs[i].k = k
        end

        err = 0
        for idx in 1:(n_t)

            mean_xf = transition(ssm, x_t[idx, :], exogenous_variables[idx, :], control_variables[idx, :], [], time_variables[idx])
            innov = y_t[idx, :] .- mean_xf
            err += mean((innov)^2)
        end
        E[index_k] = sqrt(err)/n_t

    end

    ind_min = argmin(E)

    return k_list[ind_min], E

end

