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


function EM(model::ForecastingModel{GaussianLinearStateSpaceSystem}, y_t, exogenous_variables, control_variables; maxiters_em=100, abstol_em=1e-8, reltol_em=1e-8, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

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

                L -= (sum(logdet(Q_i[ivar_obs, ivar_obs])) +  tr((ϵ_i*transpose(ϵ_i) + V_ϵ_i)*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L -= (sum(logdet(R_i)) + tr((η_i*transpose(η_i) + V_η_i)*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    parameters = model.parameters
    @inbounds for i in 1:maxiters_em

        if i > 2 && (abs(llk_array[end-1] - llk_array[end])  <  abstol_em || abs((llk_array[end-1] - llk_array[end])/(llk_array[end-1]))  <  reltol_em)
            break
        end

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


function EM_EnKS(model::ForecastingModel, y_t, exogenous_variables, control_variables; lb = nothing, ub = nothing, n_particles=30, maxiters_em=100, abstol_em=1e-8, reltol_em=1e-8, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    # Q function
    function Q(parameters, smoothed_values)

        L = 0.0
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
    parameters = model.parameters
    @inbounds for i in 1:maxiters_em

        if i > 2 && (abs(llk_array[end-1] - llk_array[end])  <  abstol_em || abs((llk_array[end-1] - llk_array[end])/(llk_array[end-1]))  <  reltol_em)
            break
        end
        
        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model, n_particles=n_particles))

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters, smoother_method=EnsembleKalmanSmoother(model, n_particles=n_particles))

        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_ouput, lb = lb, ub = ub)
        sol = solve(prob, optim_method; kwargs...)
        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model, n_particles=n_particles))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters, (llk = llk_array, )

end


function EM_EnKS2(model::ForecastingModel, y_t, exogenous_variables, control_variables; n_particles=30, maxiters_em=100, abstol_em=1e-8, reltol_em=1e-8, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

    # Fixed values
    n_obs = size(y_t, 1)

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    view_exo_row = eachrow(exogenous_variables)
    view_control_row = eachrow(control_variables)

    # Q function
    function Q(parameters, smoothed_values)

        R_arr = model.system.R_t.(view_exo_row, eachcol(parameters))
        Q_arr = model.system.Q_t.(view_exo_row, eachcol(parameters))
        M_arr = transition((model.system, ), todo, view_exo_row, view_control_row, eachcol(parameters))

        L = 0.0
        @inbounds for t in 1:n_obs

            ivar_obs = ivar_obs_vec[t]

            R_i = R_arr[t]

            M_i = transition(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables[t, :], control_variables[t, :], parameters)
    
            η_i = smoothed_values.smoothed_state[t+1].particles_state - M_i
            Ω = (η_i*η_i') ./ (n_particles - 1)

            if valid_obs_vec[t]

                H_i = observation(model.system, smoothed_values.smoothed_state[t].particles_state, exogenous_variables[t, :], parameters)[ivar_obs, :]
                Q_i = Q_arr[t]
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
    @inbounds for i in 1:maxiters_em

        if i > 2 && (abs(llk_array[end-1] - llk_array[end])  <  abstol_em || abs((llk_array[end-1] - llk_array[end])/(llk_array[end-1]))  <  reltol_em)
            break
        end
        
        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model, n_particles=n_particles))

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoother_ouput = smoother(model, y_t, exogenous_variables, control_variables, filter_output; parameters=parameters, smoother_method=EnsembleKalmanSmoother(model, n_particles=n_particles))

        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_ouput)
        sol = solve(prob, optim_method; kwargs...)

        parameters = sol.minimizer
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=EnsembleKalmanFilter(model, n_particles=n_particles))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters, (llk = llk_array, )

end


function EM_EnKS_old2(model::ForecastingModel, y_t, exogenous_variables, control_variables; n_particles=30, n_iter=100, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

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


function SEM(model::ForecastingModel, y_t, exogenous_variables, control_variables; lb = nothing, ub = nothing, n_filtering=30, n_smoothing=30, maxiters_em=100, abstol_em=1e-8, reltol_em=1e-8, optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), kwargs...)

    # Fixed values
    n_obs = size(y_t, 1)
    t_start = model.current_state.t
    dt = model.system.dt

    ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

    # Q function
    function Q(parameters, smoothed_values)

        L = 0.0
        @inbounds for t in 1:n_obs

            # Get current t_step
            # t_step = t_start + (t-1)*dt

            ivar_obs = ivar_obs_vec[t]

            R_i = model.system.R_t(exogenous_variables[t, :], parameters)

            M_i = transition(model.system, smoothed_values[t].particles_state, exogenous_variables[t, :], control_variables[t, :], parameters)
    
            η_i = smoothed_values[t+1].particles_state - M_i
            Ω = (η_i*η_i') ./ (n_smoothing - 1)

            if valid_obs_vec[t]

                H_i = observation(model.system, smoothed_values[t].particles_state, exogenous_variables[t, :], parameters)[ivar_obs, :]
                Q_i = model.system.Q_t(exogenous_variables[t, :], parameters)
                ϵ_i = y_t[t, ivar_obs] .- H_i
                Σ = (ϵ_i*ϵ_i') ./ (n_smoothing - 1)

                L += - (sum(log(det(Q_i[ivar_obs, ivar_obs]))) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += - (log(2*pi) + sum(log(det(R_i))) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    parameters = model.parameters
    @inbounds for i in 1:maxiters_em

        if i > 2 && (abs(llk_array[end-1] - llk_array[end])  <  abstol_em || abs((llk_array[end-1] - llk_array[end])/(llk_array[end-1]))  <  reltol_em)
            break
        end

        filter_output, filtered_state, filtered_state_var= filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ParticleFilter(model, n_particles = n_filtering))

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        smoothed_particles_swarm = backward_smoothing(y_t, exogenous_variables, filter_output, model, parameters; n_smoothing=n_smoothing)

        prob = Optimization.OptimizationProblem(optprob, parameters, smoothed_particles_swarm, lb = lb, ub = ub)
        sol = solve(prob, optim_method; kwargs...)
        parameters = sol.minimizer
    
    end

    filter_output , filtered_state, filtered_state_var = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ParticleFilter(model, n_particles = n_filtering))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters, (llk = llk_array, )

end