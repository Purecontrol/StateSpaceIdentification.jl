using Optimization, OptimizationOptimJL, OptimizationOptimisers
using SparseArrays
using FiniteDiff
using NaNMath; nm=NaNMath


function numerical_MLE(model::ForecastingModel, y_t, exogenous_variables, control_variables; optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), verbose=false, kwargs...)

    function inverse_llk(params_vec, unused)

        return - loglike(model, y_t, exogenous_variables, control_variables; parameters = params_vec)
    
    end

    if verbose == true
        function callback_verbose(p, loss)
            println("Log Likelihood : ", - loss)
            return false
        end
    end

    optprob = OptimizationFunction(inverse_llk, diff_method)
    prob = Optimization.OptimizationProblem(optprob, model.parameters, [])
    if verbose == true
        sol = solve(prob, optim_method; callback = callback_verbose, kwargs...)
    else
        sol = solve(prob, optim_method; kwargs...)
    end

    return sol

end


function EM(model::ForecastingModel{GaussianLinearStateSpaceSystem}, y_t, exogenous_variables, control_variables; n_iter=100, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

    # Fixed values
    n_obs = size(y_t, 1)

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    # Q function
    function Q(parameters, smoothed_values)

        L = 0
        @inbounds for t in 1:n_obs

            ivar_obs = ivar_obs_vec[t]

            R_i = model.system.R_t(exogenous_variables[t, :], parameters)
            A_i = model.system.A_t(exogenous_variables[t, :], parameters)
            B_i = model.system.B_t(exogenous_variables[t, :], parameters)
            c_i = model.system.c_t(exogenous_variables[t, :], parameters)

            η_i = smoothed_values.smoothed_state[t+1].μ_t - (A_i*smoothed_values.smoothed_state[t].μ_t + B_i*control_variables[t, :] + c_i)
            V_η_i = smoothed_values.smoothed_state[t+1].σ_t - smoothed_values.autocov_state[t]*transpose(A_i) - A_i*transpose(smoothed_values.autocov_state[t]) + A_i*smoothed_values.smoothed_state[t].σ_t*transpose(A_i)
            
            if valid_obs_vec[t]

                H_i = model.system.H_t(exogenous_variables[t, :], parameters)
                d_i = model.system.d_t(exogenous_variables[t, :], parameters)
                Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)

                ϵ_i = y_t[t, ivar_obs] - (H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].μ_t + d_i[ivar_obs])
                V_ϵ_i = H_i[ivar_obs, :]*smoothed_values.smoothed_state[t].σ_t*transpose(H_i[ivar_obs, :])

                L -= (sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L -= (sum(log(det(R_i))) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    parameters = model.parameters
    @inbounds for i in 1:n_iter

        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters)

        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_ouput)
        sol = solve(prob, optim_method; kwargs...)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters)

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end


function EM_EnKS(model::ForecastingModel, y_t, exogenous_variables, control_variables; n_particles=30, n_iter=100, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    # Q function
    function Q(parameters, smoothed_values)

        L = 0
        @inbounds for t in 1:n_obs

            # Get current t_step
            # t_step = t_start + (t-1)*dt

            ivar_obs = ivar_obs_vec[t]

            R_i = model.system.R_t(exogenous_variables[t, :], parameters)

            M_i = transition(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables[t, :], control_variables[t, :], parameters)
    
            η_i = smoothed_values.smoothed_state[t+1].particles_state - M_i
            Ω = (η_i*η_i') ./ (n_particles - 1)

            if valid_obs_vec[t]

                H_i = observation(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables[t, :], parameters)[ivar_obs, :]
                Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)
                ϵ_i = y_t[t, ivar_obs] .- H_i
                Σ = (ϵ_i*ϵ_i') ./ (n_particles - 1)

                L -= (sum(logdet(Q_i[ivar_obs, ivar_obs])) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L -= (sum(logdet(R_i)) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    mean_solving_time = 0.0
    parameters = model.parameters
    @inbounds for i in 1:n_iter
        
        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters, smoother_method=EnsembleKalmanSmoother(model.system.n_X, model.system.n_Y, n_particles))

        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_ouput)
        t1 = time()
        sol = solve(prob, optim_method; kwargs...)
        t2 = time()
        if i != 1
            mean_solving_time += (t2 - t1)/(n_iter-1)
        end
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])
    println("Mean solving time : ", mean_solving_time, " s.")

    return parameters

end


function EM_EnKS2(model::ForecastingModel, y_t, exogenous_variables, control_variables; n_particles=30, n_iter=100, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    # Q function
    function Q(parameters, smoothing_states)

        L = 0
        @inbounds for t in 1:n_obs

            # Get current t_step
            # t_step = t_start + (t-1)*dt

            ivar_obs = ivar_obs_vec[t]

            R_i = model.system.R_t(exogenous_variables[t, :], parameters)

            M_i = transition(model.system, smoothing_states[t], exogenous_variables[t, :], control_variables[t, :], parameters)
    
            η_i = smoothing_states[t+1] - M_i
            Ω = (η_i*η_i') ./ (n_particles - 1)

            if valid_obs_vec[t]

                H_i = observation(model.system, smoothing_states[t], exogenous_variables[t, :], parameters)[ivar_obs, :]
                Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)
                ϵ_i = y_t[t, ivar_obs] .- H_i
                Σ = (ϵ_i*ϵ_i') ./ (n_particles - 1)

                L -= (sum(logdet(Q_i[ivar_obs, ivar_obs])) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L -= (sum(logdet(R_i)) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    parameters = model.parameters
    @inbounds for i in 1:n_iter
        
        @showtime filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        @showtime smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters, smoother_method=EnsembleKalmanSmoother(model.system.n_X, model.system.n_Y, n_particles))
        
        all_particles_states = [i.particles_state for i in smoother_ouput.smoothed_state.state]

        prob = Optimization.OptimizationProblem(optprob, parameters, all_particles_states)
        @showtime sol = solve(prob, optim_method; kwargs...)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model.current_state, model.system.n_X, model.system.n_Y, n_particles))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters

end