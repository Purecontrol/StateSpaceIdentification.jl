using Optimization, OptimizationOptimJL, OptimizationOptimisers
using SparseArrays
using FiniteDiff
using NaNMath; nm=NaNMath


function numerical_MLE(model::ForecastingModel, y_t, exogenous_variables, control_variables; optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), verbose=false, filter_method::AbstractFilter=default_filter(model), kwargs...)

    function inverse_llk(params_vec, unused)

        return - loglike(model, y_t, exogenous_variables, control_variables; parameters = params_vec, filter = filter_method)
    
    end

    if verbose == true
        function callback_verbose(p, loss)
            println("Log Likelihood : ", - loss)
            return false
        end
    end

    optprob = OptimizationFunction(inverse_llk, diff_method)
    prob = Optimization.OptimizationProblem(optprob, model.parameters, [])
    sol = []
    try
        if verbose == true
            sol = solve(prob, optim_method; callback = callback_verbose, kwargs...)
        else
            sol = solve(prob, optim_method; kwargs...)
        end
    catch e
        println("Singular Value")
        return []
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


function SEM(model::ForecastingModel, y_t, exogenous_variables, control_variables; optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), maxiters=100, abstol=1e-8, reltol=1e-8, p_filter = Dict(:n_particles => 30), p_smoothing = Dict(:n_particles => 30), p_opt_problem = Dict(), p_optim_method = Dict())

    # Choice of filter and smoother
    n_smoothing = p_smoothing[:n_particles]

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

                L += - (sum(logdet(Q_i[ivar_obs, ivar_obs])) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += - (sum(logdet(R_i)) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    parameters = model.parameters
    time_filtering = 0.0
    time_smoothing = 0.0
    time_M = 0.0
    @inbounds for i in 1:maxiters

        if i > 2 && (abs(llk_array[end-1] - llk_array[end])  <  abstol || abs((llk_array[end-1] - llk_array[end])/(llk_array[end-1]))  <  reltol)
            break
        end

        t1 = time()
        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ParticleFilter(model; p_filter...))
        t2 = time()
        time_filtering += t2-t1

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        t1 = time()
        smoother_output = smoother(model, y_t, exogenous_variables, control_variables, filter_output; smoother_method=BackwardSimulationSmoother(model; p_smoothing...))
        t2 = time()
        time_smoothing += t2-t1

        t1 = time()
        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_output.smoothed_state; p_opt_problem...)
        sol = solve(prob, optim_method; p_optim_method...)
        parameters = sol.minimizer
        t2 = time()
        time_M += t2-t1
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ParticleFilter(model; p_filter...))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters, (llk = llk_array, time_filtering=time_filtering/maxiters, time_smoothing=time_smoothing/maxiters, time_M=time_M/maxiters)

end


function SEM_CPF(model::ForecastingModel, y_t, exogenous_variables, control_variables; optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), maxiters=100, abstol=1e-8, reltol=1e-8, p_filter = Dict(:n_particles => 30), p_smoothing = Dict(:n_particles => 30), p_opt_problem = Dict(), p_optim_method = Dict())

    # Choice of filter and smoother
    n_smoothing = p_smoothing[:n_particles]

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

                L += - (sum(logdet(Q_i[ivar_obs, ivar_obs])) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += - (sum(logdet(R_i)) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    parameters = model.parameters
    time_filtering = 0.0
    time_smoothing = 0.0
    time_M = 0.0
    conditional_particle = nothing
    @inbounds for i in 1:maxiters

        if i > 2 && (abs(llk_array[end-1] - llk_array[end])  <  abstol || abs((llk_array[end-1] - llk_array[end])/(llk_array[end-1]))  <  reltol)
            break
        end

        t1 = time()
        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ConditionalParticleFilter(model; conditional_particle = conditional_particle, p_filter...))
        t2 = time()
        time_filtering += t2-t1

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        t1 = time()
        smoother_output = smoother(model, y_t, exogenous_variables, control_variables, filter_output; smoother_method=BackwardSimulationSmoother(model; p_smoothing...))
        t2 = time()
        time_smoothing += t2-t1

        # Set conditional particle
        conditional_particle = hcat(map((x) -> x.particles_state[:, end], smoother_output.smoothed_state)...)'

        t1 = time()
        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_output.smoothed_state; p_opt_problem...)
        sol = solve(prob, optim_method; p_optim_method...)
        parameters = sol.minimizer
        t2 = time()
        time_M += t2-t1
    
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ConditionalParticleFilter(model; conditional_particle = conditional_particle, p_filter...))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters, (llk = llk_array, time_filtering=time_filtering/maxiters, time_smoothing=time_smoothing/maxiters, time_M=time_M/maxiters)

end


function npSEM_CPF(model::ForecastingModel, y_t, exogenous_variables, control_variables, update_M; optim_method=Optim.Newton(), diff_method=Optimization.AutoForwardDiff(), maxiters=100, abstol=1e-8, reltol=1e-8, p_filter = Dict(:n_particles => 30), p_smoothing = Dict(:n_particles => 30), p_opt_problem = Dict(), p_optim_method = Dict())

    # Choice of filter and smoother
    n_smoothing = p_smoothing[:n_particles]

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

                L += - (sum(logdet(Q_i[ivar_obs, ivar_obs])) +  tr(Σ*pinv(Q_i[ivar_obs, ivar_obs])) )
            end
            L += - (sum(logdet(R_i)) + tr(Ω*pinv(R_i)) )

        end

        return - L/n_obs

    end

    # M function
    optprob = OptimizationFunction(Q, diff_method)

    llk_array = []
    parameters = model.parameters
    time_filtering = 0.0
    time_smoothing = 0.0
    time_M = 0.0
    conditional_particle = nothing
    @inbounds for i in 1:maxiters

        if i > 2 && (abs(llk_array[end-1] - llk_array[end])  <  abstol || abs((llk_array[end-1] - llk_array[end])/(llk_array[end-1]))  <  reltol)
            break
        end

        t1 = time()
        filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ConditionalParticleFilter(model; conditional_particle = conditional_particle, p_filter...))
        t2 = time()
        time_filtering += t2-t1

        push!(llk_array, filter_output.llk / n_obs)
        println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

        t1 = time()
        smoother_output = smoother(model, y_t, exogenous_variables, control_variables, filter_output; smoother_method=BackwardSimulationSmoother(model; p_smoothing...))
        t2 = time()
        time_smoothing += t2-t1

        # Set conditional particle
        conditional_particle = hcat(map((x) -> x.particles_state[:, end], smoother_output.smoothed_state)...)'

        t1 = time()
        prob = Optimization.OptimizationProblem(optprob, parameters, smoother_output.smoothed_state; p_opt_problem...)
        sol = solve(prob, optim_method; p_optim_method...)
        parameters = sol.minimizer
        t2 = time()
        time_M += t2-t1

        # Update non parametric estimate of m
        ns = 2
        if ns == 1
            idx_selected_particules = sample(collect(1:(n_smoothing-1)))
            new_x = hcat(map((x) -> x.particles_state[:, idx_selected_particules], smoother_output.smoothed_state)...)'
            idx = Int.(1:(size(new_x, 1)-2))
        else
            new_x = map((x) -> x.particles_state, smoother_output.smoothed_state)
            println(size(new_x))
        end
        update_M(idx, new_x, exogenous_variables, control_variables, model.system.llrs)

    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; parameters=parameters, filter=ConditionalParticleFilter(model; conditional_particle = conditional_particle, p_filter...))

    push!(llk_array, filter_output.llk / n_obs)
    println("Final | Log Likelihood : ", llk_array[end])

    return parameters, (llk = llk_array, time_filtering=time_filtering/maxiters, time_smoothing=time_smoothing/maxiters, time_M=time_M/maxiters)

end