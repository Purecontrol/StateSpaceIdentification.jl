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


function (llr::LLR)(x, idx)

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
    if idx != 0 && lag_x != 0
        for i in 1:nb_point_tree
            if (max(1, idx-lag_x) <= llr.index_analogs[i] <= min(nb_point_tree, idx+lag_x))
                push!(llr.ignored_nodes, i)
            end
        end
    end

    if llr.ignored_nodes == Set(1:nb_point_tree)
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

    function GaussianNonParametricStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs)

        return new(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs)
    end

end


function transition(ssm::GaussianNonParametricStateSpaceSystem, current_x, exogenous_variables, control_variables, parameters, t) 

    return ssm.M_t(current_x, exogenous_variables, control_variables, ssm.llrs, t)

end


function observation(ssm::GaussianNonParametricStateSpaceSystem, current_x, exogenous_variables, parameters, t) 

    return ssm.H_t(current_x, exogenous_variables, parameters, t)

end


function forecast(system::GaussianNonParametricStateSpaceSystem , current_state::GaussianStateStochasticProcess, exogenous_variables, control_variables, parameters; n_steps_ahead=1)

    predicted_state = TimeSeries{GaussianStateStochasticProcess}(n_steps_ahead+1, system.n_X)
    predicted_obs = TimeSeries{GaussianStateStochasticProcess}(n_steps_ahead+1, system.n_Y)

    # Define init conditions
    current_obs = system.H_t(current_state.μ_t, exogenous_variables[1, :], parameters)
    predicted_state[1].t = current_state.t
    predicted_state[1].μ_t = current_state.μ_t
    predicted_state[1].σ_t = current_state.σ_t
    predicted_obs[1].t = current_state.t
    predicted_obs[1].μ_t = current_obs
    predicted_obs[1].σ_t = transpose(current_H)*current_state.σ_t*current_H + system.Q_t(exogenous_variables[1, :], parameters)

    @inbounds for step in 2:(n_steps_ahead+1)

        # Define current t_step
        t_step = current_state.t + (step-1)*system.dt

        # Get current matrix A and B
        A = system.A_t(exogenous_variables[step, :], parameters)
        B = system.B_t(exogenous_variables[step, :], parameters)
        H = system.H_t(exogenous_variables[step, :], parameters)
        
        # Update predicted state and covariance
        predicted_state[step].t = t_step
        predicted_state[step].μ_t = A*predicted_state[step-1].μ_t + B*control_variables[step-1, :] + system.c_t(exogenous_variables[step, :], parameters)
        predicted_state[step].σ_t = transpose(A)*predicted_state[step-1].σ_t*A + system.R_t(exogenous_variables[step, :], parameters)
        
        # Update observed state and covariance
        predicted_obs[step].t = t_step
        predicted_obs[step].μ_t = H*predicted_state[step].μ_t + system.d_t(exogenous_variables[step, :], parameters)
        predicted_obs[step].σ_t = transpose(H)*predicted_state[step].σ_t*H + system.Q_t(exogenous_variables[step, :], parameters)


    end

    return predicted_state, predicted_obs

end


function default_filter(model::ForecastingModel{GaussianNonParametricStateSpaceSystem})

    return ParticleFilter(model.current_state, model.system.n_X, model.system.n_Y, 50)

end


function k_choice(ssm::GaussianNonParametricStateSpaceSystem, x_t, y_t, exogenous_variables, control_variables; k_list = [5, 10, 15, 20, 25])

    n_t = size(x_t, 1)

    L = zeros(length(k_list))
    E = zeros(length(k_list))
    for (index_k, k) in enumerate(k_list)

        llr_exp = deepcopy(ssm.llrs)
        for i in size(llr_exp, 1)
            llr_exp[i].k = k
        end

        err = 0
        for idx in 1:(n_t)

            t_step = 0

            mean_xf = ssm.M_t(x_t[idx, :], exogenous_variables[idx, :], control_variables[idx, :], llr_exp, t_step)
            innov = y_t[idx, :] .- mean_xf
            err += mean((innov)^2)
        end
        E[index_k] = sqrt(err)/n_t

    end

    println(E)

    ind_min = argmin(E)

    return k_list[ind_min], E

end

