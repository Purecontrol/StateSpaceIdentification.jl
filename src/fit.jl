using Optimization, OptimizationOptimJL
using StaticArrays
using ForwardDiff

function numerical_MLE(
        model::ForecastingModel{Z},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z};
        optim_method = Optim.Newton(),
        diff_method = Optimization.AutoForwardDiff(),
        verbose = false,
        kwargs...
) where {Z <: Real}
    all_parameters = isa(model.parameters, Vector{Z}) ? ComponentArray(to_optimize=deepcopy(model.parameters), fixed=[]) : deepcopy(model.parameters)
    free_parameters = all_parameters.to_optimize
    fixed_parameters = all_parameters.fixed
    model_opt = deepcopy(model)

    # Log likelihood function
    function inverse_llk(parameters, hyperparameters)
        return -loglike(
            model_opt,
            observation_data,
            exogenous_data,
            control_data;
            parameters = vcat(parameters, fixed_parameters)
        )
    end

    # Callback function for logging if verbose
    function callback_verbose(p, loss)
        verbose ? (println("Log Likelihood: ", -loss); false) : false
    end

    # Optimization problem setup
    optprob = OptimizationFunction(inverse_llk, diff_method)
    prob = Optimization.OptimizationProblem(optprob, copy(free_parameters))

    # Solve the problem
    sol = solve(prob, optim_method; kwargs..., callback = callback_verbose)

    return sol
end

function E_step(parameters, model, observation_data, exogenous_data,
        control_data, filter_method, smoother_method)
    # Filtering
    filter_output = filtering(model, observation_data, exogenous_data, control_data;
        parameters = parameters, filter_method = deepcopy(filter_method))

    # Smoothing
    smoother_output = smoothing(
        model, observation_data, exogenous_data, control_data, filter_output;
        parameters = parameters, smoother_method = deepcopy(smoother_method))

    return filter_output, smoother_output
end

function M_step(parameters, optprob, optim_method, inputs_Q, p_optim_method, p_opt_problem)

    # Parameter update
    prob = Optimization.OptimizationProblem(optprob, parameters, inputs_Q; p_opt_problem...)
    sol = solve(prob, optim_method; p_optim_method...)
    return sol.minimizer

end

"""
Updates the Local Linear Regressors (LLRs) within a GaussianNonParametricStateSpaceSystem based on smoothed particle data.

$(TYPEDSIGNATURES)

This function samples particles from the smoothed particle swarm, constructs analog data, 
and updates the LLRs using either a provided callback function or the default update function.
"""
function M_nonparametric_step!(model, parameters, exogenous_data, control_data, smoothed_particles, n_particles_smoothing, n_obs, n_X, n_iteration, custom_llrs_update_callback!)
    
    # Sample choosen particles
    n_selected_particles_nonparametrics = 5
    idxes_selected_particules = sample(collect(1:(n_particles_smoothing - 1)), n_selected_particles_nonparametrics)

    # Get selected particles
    analog_states = permutedims(smoothed_particles[1:(end-1), :, idxes_selected_particules], (1, 3, 2))
    analog_indexes = repeat(Int.(1:(n_obs - 1)), n_selected_particles_nonparametrics)
    analog_times_in_days = repeat([model.current_state.t + (t - 1) * model.system.dt for t in 1:(n_obs - 1)], n_selected_particles_nonparametrics)

    # Temporary not update mean and variance (it will be like some kind of normalization layer)
    # model.system.μ = mean(reshape(new_x[1:end, :], (:)), dims=1)
    # model.system.σ = std(reshape(new_x[1:end, :], (:)), dims=1)

    # Update catalog
    update_llrs!(model.system, analog_indexes, analog_times_in_days, analog_states, exogenous_data, control_data, update_function! = custom_llrs_update_callback!)
    
    # Search for good number of neighbors
    if n_iteration % 3 == 1
        k_list = collect(50:5:450)
        opt_k, _ = find_optimal_number_neighbors(
            model.system,
            parameters,
            analog_states[1:(end - 1), 1, :],
            analog_states[2:(end), 1, :],
            exogenous_data,
            control_data,
            analog_times_in_days,
            k_list
        )
        @info "Update n_neighbors=$(opt_k) to all LocalLinearRegressor."
        for llr in model.system.llrs
            llr.n_neighbors = opt_k
        end
    end
end

function ExpectationMaximization(
        model::ForecastingModel{Z, S},
        observation_data::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z};
        filter_method::AbstractFilter{Z} = default_filter(model),
        smoother_method::AbstractSmoother{Z} = default_smoother(model),
        maxiters_em::Int = 100,
        abstol_em::Float64 = 1e-8,
        reltol_em::Float64 = 1e-8,
        optim_method = Optim.Newton(),
        diff_method = Optimization.AutoForwardDiff(),
        p_optim_method = Dict(),
        p_opt_problem = Dict(),
        verbose = false,
        iter_saem = 0,
        alpha = 1,
        fit_initial_conditions=false, 
        custom_llrs_update_callback!::Union{Function, Nothing} = nothing
) where {Z <: Real, S <: AbstractStateSpaceSystem{Z}}

    # Fixed values
    n_obs = size(observation_data, 1)

    ivar_obs_vec = [findall(.!isnan.(observation_data[t, :])) for t in 1:n_obs]
    valid_obs_vec = [length(ivar_obs_vec[t]) > 0 for t in 1:n_obs]
    ssm = model.system
    t_index_table = [model.current_state.t + (ssm.dt) * (t - 1) for t in 1:n_obs]
    n_X = ssm.n_X
    n_Y = ssm.n_Y

    Q = get_Q_function(filter_method, smoother_method, ivar_obs_vec, valid_obs_vec, t_index_table, ssm,
        n_X, n_Y, n_obs, observation_data, exogenous_data, control_data)

    # Parameter setup
    all_parameters = isa(model.parameters, Vector{Z}) ? ComponentArray(to_optimize=deepcopy(model.parameters), fixed=[]) : deepcopy(model.parameters)
    free_parameters = all_parameters.to_optimize
    fixed_parameters = all_parameters.fixed
    function Q_restricted(parameters, p)
        return Q(vcat(parameters, fixed_parameters), p)
    end

    # Optimization setup
    optprob = OptimizationFunction(Q_restricted, diff_method)

    # Convergence setup
    llk_array = []
    rstd_llk_array = []
    rs = RollingStd(10)
    termination_flag = false
    if iter_saem > 0
        @warn "By default alpha is fixed to 1, so fixing iter_saem > 0 whithout changing alpha has no effect."
        maxiters_saem = maxiters_em
    end

    # Main EM iterations
    for i in 1:maxiters_em
        if i > 2 &&
            !termination_flag &&
            (
                abs(rstd_llk_array[end - 1] - rstd_llk_array[end]) < abstol_em ||
                abs((rstd_llk_array[end - 1] - rstd_llk_array[end]) / (rstd_llk_array[end - 1])) < reltol_em
            )
            if verbose && iter_saem > 0
                @info "Algorithm has converged. Continued for $iter_saem iterations to stabilize solution."
            end
            termination_flag = true
            maxiters_saem = i + iter_saem
        elseif termination_flag && i > maxiters_saem
            @info "SAEM iterations are finished."
            break
        elseif !termination_flag && i > maxiters_em - iter_saem
            termination_flag = true
            @info "Algorithm has not converged yet. However, due to the value of maxiters_em switch to SAEM with $iter_saem iterations to stabilize solution."
        end

        filter_output, smoother_output = E_step(collect(all_parameters), model, observation_data, exogenous_data, control_data, filter_method, smoother_method)
        push!(llk_array, filter_output.llk / n_obs)
        update!(rs, llk_array[end])
        push!(rstd_llk_array, get_value(rs))
        if verbose
             @info "Iter n° $(i-1) | Log Likelihood: $(round(llk_array[end], digits=5))."
        end
        inputs_Q = postprocessing_smoother_output(smoother_output, n_obs, n_X)

        # Update conditional particle
        if isa(filter_method, ConditionalParticleFilter)
            filter_method.conditional_particle = hcat(map((x) -> x.particles_state[:, end], smoother_output.smoothed_particles_swarm)...)'
        end

        # Optimization with smoothed data
        found_parameters = M_step(copy(free_parameters), optprob, optim_method, inputs_Q, p_optim_method, p_opt_problem)
        if termination_flag == false
            free_parameters = found_parameters
        else
            free_parameters = alpha * found_parameters + (1 - alpha) * free_parameters
        end
        all_parameters.to_optimize = free_parameters

        if fit_initial_conditions == true
            model.current_state = smoother_output.smoothed_state[1]
        end

        # Update catalog
        if isa(ssm, GaussianNonParametricStateSpaceSystem)
            M_nonparametric_step!(model, collect(all_parameters), exogenous_data, control_data, inputs_Q, smoother_method.n_particles, n_obs, n_X, i, custom_llrs_update_callback!)
        end

    end

    # Final filtering
    filter_output = filtering(model, observation_data, exogenous_data, control_data;
        parameters = collect(all_parameters), filter_method = deepcopy(filter_method))
    push!(llk_array, filter_output.llk / n_obs)
    if verbose
        println("Final | Log Likelihood: ", llk_array[end])
    end

    return isa(model.parameters, Vector{Z}) ? collect(all_parameters) : all_parameters
end

############################################################################################
########################################## KALMAN ##########################################
############################################################################################

function postprocessing_smoother_output(
        smoother_output::KalmanSmootherOutput{Z}, n_obs, n_X) where {Z <: Real}
    smoothed_state_μ = SizedArray{Tuple{n_obs + 1, n_X}}(stack(
        map(t -> t.μ_t, smoother_output.smoothed_state), dims = 1))
    smoothed_state_Σ = SizedArray{Tuple{n_obs + 1, n_X, n_X}}(stack(
        map(t -> Symmetric(t.Σ_t), smoother_output.smoothed_state), dims = 1))
    return (smoothed_state_μ, smoothed_state_Σ, smoother_output.autocov_state)
end

function get_Q_function(filter_method, smoother_method::S, ivar_obs_vec, valid_obs_vec, t_index_table,
        ssm, n_X, n_Y, n_obs, observation_data, exogenous_data,
        control_data) where {Z <: Real, S <: AbstractGaussianDeterministicSmoother{Z}}
    function QKalman(parameters, p)
        smoothed_state_μ, smoothed_state_Σ, smoothed_autocov = p

        L = eltype(parameters)(0.0)
        η_i = zeros(eltype(parameters), n_X)
        V_η_i = zeros(eltype(parameters), n_X, n_X)
        ϵ_i = zeros(eltype(parameters), n_Y)
        V_ϵ_i = zeros(eltype(parameters), n_Y, n_Y)

        @inbounds for (t, t_step) in enumerate(t_index_table)
            ivar_obs = ivar_obs_vec[t]

            ex = view(exogenous_data, t, :)
            R_i = Symmetric(ssm.R_t(ex, parameters, t_step))
            A_i = ssm.A_t(ex, parameters, t_step)
            B_i = ssm.B_t(ex, parameters, t_step)
            c_i = ssm.c_t(ex, parameters, t_step)

            η_i .= view(smoothed_state_μ, t + 1, :) - (
                A_i * view(smoothed_state_μ, t, :) +
                B_i * view(control_data, t, :) +
                c_i
            )
            V_η_i .= view(smoothed_state_Σ, t + 1, :, :) -
                     smoothed_autocov[t] * transpose(A_i) -
                     A_i * transpose(smoothed_autocov[t]) +
                     A_i * view(smoothed_state_Σ, t, :, :) * transpose(A_i)

            if valid_obs_vec[t]
                H_i = view(ssm.H_t(ex, parameters, t_step), ivar_obs, :)
                d_i = view(ssm.d_t(ex, parameters, t_step), ivar_obs)
                Q_i = Symmetric(ssm.Q_t(ex, parameters, t_step)[ivar_obs, ivar_obs])

                ϵ_i .= view(observation_data, t, ivar_obs) - (
                    H_i * view(smoothed_state_μ, t, :) +
                    d_i
                )
                V_ϵ_i .= H_i *
                         view(smoothed_state_Σ, t, :, :) *
                         transpose(H_i)

                L -= (
                    logdet(Q_i) +
                    tr((ϵ_i * transpose(ϵ_i) + V_ϵ_i) * inv(Q_i))
                )
            end
            L -= (logdet(R_i) + tr((η_i * transpose(η_i) + V_η_i) * inv(R_i)))
        end

        return -L / n_obs
    end

    return QKalman
end

function get_Q_function(filter_method, smoother_method::ExtendedKalmanSmoother{Z}, ivar_obs_vec, valid_obs_vec, t_index_table,
    ssm, n_X, n_Y, n_obs, observation_data, exogenous_data,
    control_data) where {Z <: Real}

    function QExtendedKalman(parameters, p)
        smoothed_state_μ, smoothed_state_Σ, smoothed_autocov = p

        L = eltype(parameters)(0.0)
        η_i = zeros(eltype(parameters), n_X)
        V_η_i = zeros(eltype(parameters), n_X, n_X)
        ϵ_i = zeros(eltype(parameters), n_Y)
        V_ϵ_i = zeros(eltype(parameters), n_Y, n_Y)

        @inbounds for (t, t_step) in enumerate(t_index_table)
            ivar_obs = ivar_obs_vec[t]

            ex = view(exogenous_data, t, :)
            R_i = Symmetric(ssm.R_t(ex, parameters, t_step))
            M_i = transition(
                ssm,
                view(smoothed_state_μ, t, :),
                ex,
                view(control_data, t, :),
                parameters,
                t_step
            )
            dM_i = filter_method.dM_t(
                view(smoothed_state_μ, t, :),
                ex,
                view(control_data, t, :),
                parameters, 
                t_step
            )

            η_i .= view(smoothed_state_μ, t + 1, :) - M_i
            
            V_η_i .= view(smoothed_state_Σ, t + 1, :, :) -
                    smoothed_autocov[t] * transpose(dM_i) -
                    dM_i * transpose(smoothed_autocov[t]) +
                    dM_i * view(smoothed_state_Σ, t, :, :) * transpose(dM_i)

            if valid_obs_vec[t]
                H_i = view(observation(
                    ssm,
                    view(smoothed_state_μ, t, :),
                    ex,
                    parameters,
                    t_step
                ), ivar_obs, :)
                dH_i = view(filter_method.dH_t(
                    view(smoothed_state_μ, t, :),
                    ex,
                    parameters, 
                    t_step
                ), ivar_obs, :)
                Q_i = Symmetric(ssm.Q_t(ex, parameters, t_step)[ivar_obs, ivar_obs])

                ϵ_i .= view(observation_data, t, ivar_obs) - H_i
                V_ϵ_i .= dH_i *
                        view(smoothed_state_Σ, t, :, :) *
                        transpose(dH_i)

                L -= (
                    logdet(Q_i) +
                    tr((ϵ_i * transpose(ϵ_i) + V_ϵ_i) * inv(Q_i))
                )
            end
            L -= (logdet(R_i) + tr((η_i * transpose(η_i) + V_η_i) * inv(R_i)))
        end

        return -L / n_obs
    end

    return QExtendedKalman
end

############################################################################################
############################################ SMC ###########################################
############################################################################################

function postprocessing_smoother_output(
        smoother_output::AbstractStochasticMonteCarloSmootherOutput{Z},
        n_obs, n_X) where {Z <: Real}
    n_particles = size(smoother_output.smoothed_particles_swarm[1], 2)
    smoothed_particles = SizedArray{Tuple{n_obs + 1, n_X, n_particles}}(stack(
        map(t -> t.particles_state, smoother_output.smoothed_particles_swarm), dims = 1))
    return (smoothed_particles)
end

function get_Q_function(filter_method, smoother_method::S, ivar_obs_vec, valid_obs_vec, t_index_table,
        ssm, n_X, n_Y, n_obs, observation_data, exogenous_data,
        control_data) where {Z <: Real, S <: AbstractStochasticMonteCarloSmoother{Z}}
    n_smoothing = smoother_method.n_particles
    function Q_SMC(parameters, p)
        smoothed_particles = p

        L = eltype(parameters)(0.0)
        Ω = zeros(eltype(parameters), n_X, n_X)
        Σ = zeros(eltype(parameters), n_Y, n_Y)

        @inbounds for (t, t_step) in enumerate(t_index_table)
            ivar_obs = ivar_obs_vec[t]
            ex = view(exogenous_data, t, :)

            R_i = Symmetric(ssm.R_t(ex, parameters, t_step))

            M_i = transition(
                ssm,
                view(smoothed_particles, t, :, :),
                ex,
                view(control_data, t, :),
                parameters,
                t_step
            )

            η_i = @views view(smoothed_particles, t + 1, :, :) .- M_i
            mul!(Ω, η_i, η_i')#Ω .= (η_i * η_i') ./ (n_smoothing - 1)
            Ω .*= (1.0 / (n_smoothing - 1))
            if valid_obs_vec[t]
                H_i = view(
                    observation(
                        ssm,
                        view(smoothed_particles, t, :, :),
                        ex,
                        parameters,
                        t_step), ivar_obs, :)
                Q_i = @views view(ssm.Q_t(ex, parameters, t_step), ivar_obs, ivar_obs)
                ϵ_i = @views view(observation_data, t, ivar_obs) .- H_i
                mul!(Σ, ϵ_i, ϵ_i') #Σ .= (ϵ_i * ϵ_i') ./ (n_smoothing - 1)
                Σ .*= (1.0 / (n_smoothing - 1))
                L -= (
                    sum(logdet(Q_i)) +
                    tr(Σ * pinv(Q_i))
                )
            end
            L -= (sum(logdet(R_i)) + tr(Ω * pinv(R_i)))
        end

        return -L / n_obs
    end

    return Q_SMC
end

#########################################################################################################"
#########################################################################################################"
#########################################################################################################

# TODO : clean nonparametric version
# function npSEM_CPF(
#         model::ForecastingModel,
#         y_t,
#         exogenous_variables,
#         control_variables,
#         update_M;
#         optim_method = Optim.Newton(),
#         diff_method = Optimization.AutoForwardDiff(),
#         maxiters = 100,
#         abstol = 1e-8,
#         reltol = 1e-8,
#         p_filter = Dict(:n_particles => 30),
#         p_smoothing = Dict(:n_particles => 30),
#         p_opt_problem = Dict(),
#         p_optim_method = Dict(),
#         conditional_particle = nothing
# )

#     # Choice of filter and smoother
#     n_smoothing = p_smoothing[:n_particles]

#     # Fixed values
#     n_obs = size(y_t, 1)
#     t_start = model.current_state.t
#     dt = model.system.dt

#     ivar_obs_vec = [findall(.!isnan.(y_t[t, :])) for t in 1:n_obs]
#     valid_obs_vec = [length(ivar_obs_vec) > 0 for t in 1:n_obs]

#     # Q function
#     function Q(parameters, smoothed_values)
#         L = 0.0
#         @inbounds for t in 1:n_obs

#             # Get current t_step
#             t_step = model.current_state.t + (t - 1) * model.system.dt

#             ivar_obs = ivar_obs_vec[t]

#             R_i = model.system.R_t(exogenous_variables[t, :], parameters, t_step)

#             M_i = transition(
#                 model.system,
#                 smoothed_values[t].particles_state,
#                 exogenous_variables[t, :],
#                 control_variables[t, :],
#                 parameters,
#                 t_step
#             )

#             η_i = smoothed_values[t + 1].particles_state - M_i
#             Ω = (η_i * η_i') ./ (n_smoothing - 1)

#             if valid_obs_vec[t]
#                 H_i = observation(
#                     model.system,
#                     smoothed_values[t].particles_state,
#                     exogenous_variables[t, :],
#                     parameters,
#                     t_step
#                 )[
#                     ivar_obs,
#                     :
#                 ]
#                 Q_i = model.system.Q_t(exogenous_variables[t, :], parameters, t_step)
#                 ϵ_i = y_t[t, ivar_obs] .- H_i
#                 Σ = (ϵ_i * ϵ_i') ./ (n_smoothing - 1)

#                 L += -(
#                     sum(logdet(Q_i[ivar_obs, ivar_obs])) +
#                     tr(Σ * pinv(Q_i[ivar_obs, ivar_obs]))
#                 )
#             end
#             L += -(sum(logdet(R_i)) + tr(Ω * pinv(R_i)))
#         end

#         return -L / n_obs
#     end

#     # M function
#     optprob = OptimizationFunction(Q, diff_method)

#     rs = RollingStd(10)
#     llk_array = []
#     std_array = []
#     parameters_array = []
#     parameters = model.parameters
#     time_filtering = 0.0
#     time_smoothing = 0.0
#     time_M = 0.0
#     alpha = 0.1
#     termination_flag = false
#     iter_saem = 10
#     maxiters_saem = maxiters
#     @inbounds for i in 1:maxiters
#         if i > 2 &&
#            !termination_flag &&
#            (
#                abs(std_array[end - 1] - std_array[end]) < abstol ||
#                abs((std_array[end - 1] - std_array[end]) / (std_array[end - 1])) < reltol
#            )
#             @info "Algorithm has converged. Continued for $iter_saem iterations to stabilize solution."
#             termination_flag = true
#             maxiters_saem = i + iter_saem
#             parameters = mean(parameters_array[(end - 5):end])
#         end

#         if i > maxiters_saem
#             break
#         end

#         if i >= maxiters - iter_saem
#             termination_flag = true
#         end

#         t1 = time()
#         filter_output = filter(
#             model,
#             y_t,
#             exogenous_variables,
#             control_variables;
#             parameters = parameters,
#             filter = ConditionalParticleFilter(
#                 model;
#                 conditional_particle = conditional_particle,
#                 p_filter...
#             )
#         )
#         t2 = time()
#         time_filtering += t2 - t1

#         update!(rs, filter_output.llk / n_obs)
#         push!(llk_array, filter_output.llk / n_obs)
#         push!(std_array, get_value(rs))
#         println("Iter n° $(i-1) | Log Likelihood : ", llk_array[end])

#         t1 = time()
#         smoother_output = smoother(
#             model,
#             y_t,
#             exogenous_variables,
#             control_variables,
#             filter_output;
#             parameters = parameters,
#             smoother_method = BackwardSimulationSmoother(model; p_smoothing...)
#         )
#         t2 = time()
#         time_smoothing += t2 - t1

#         # Set conditional particle
#         conditional_particle = hcat(map(
#             (x) -> x.particles_state[:, end], smoother_output.smoothed_state)...)'

#         t1 = time()
#         prob = Optimization.OptimizationProblem(
#             optprob,
#             parameters,
#             smoother_output.smoothed_state;
#             p_opt_problem...
#         )
#         sol = solve(prob, optim_method; p_optim_method...)
#         push!(parameters_array, sol.minimizer)
#         if termination_flag == false
#             parameters = sol.minimizer
#         else
#             parameters = alpha * sol.minimizer + (1 - alpha) * parameters
#         end
#         t2 = time()
#         time_M += t2 - t1

#         # Update non parametric estimate of m
#         ns = 5 #n_smoothing-1
#         # if ns == 1
#         #     idx_selected_particules = sample(collect(1:(n_smoothing-1)))
#         #     new_x = hcat(map((x) -> x.particles_state[:, idx_selected_particules], smoother_output.smoothed_state)...)'
#         #     idx = Int.(1:(size(new_x, 1)-2))
#         #     t_idx = [model.current_state.t + (t-1)*model.system.dt for t in 1:(n_obs-1)]
#         # else
#         #     idxes_selected_particules = sample(collect(1:(n_smoothing-1)), ns)
#         #     new_x = hcat([hcat(map((x) -> x.particles_state[:, i], smoother_output.smoothed_state)...)'[1:(end-1)] for i in idxes_selected_particules]...)
#         #     # new_x2 = hcat(map((x) -> x.particles_state[:, 2], smoother_output.smoothed_state)...)'
#         #     # new_x = hcat([new_x1[1:(end-1)], new_x2[1:(end-1)]]...)
#         #     idx = repeat(Int.(1:(n_obs - 1)), ns)
#         #     t_idx = repeat([model.current_state.t + (t-1)*model.system.dt for t in 1:(n_obs-1)], ns)
#         # end

#         idxes_selected_particules = sample(collect(1:(n_smoothing - 1)), ns)
#         new_x = permutedims(
#             cat(
#                 [cat(
#                      map(
#                          (x) -> x.particles_state[:, i],
#                          smoother_output.smoothed_state
#                      )...,
#                      dims = 2
#                  )'[
#                      1:(end - 1),
#                      :
#                  ] for i in idxes_selected_particules]...,
#                 dims = 3
#             ),
#             (1, 3, 2)
#         )
#         # new_x2 = hcat(map((x) -> x.particles_state[:, 2], smoother_output.smoothed_state)...)'
#         # new_x = hcat([new_x1[1:(end-1)], new_x2[1:(end-1)]]...)
#         idx = repeat(Int.(1:(n_obs - 1)), ns)
#         t_idx = repeat(
#             [model.current_state.t + (t - 1) * model.system.dt for t in 1:(n_obs - 1)],
#             ns
#         )

#         # model.system.μ = mean(reshape(new_x[1:end, :], (:)), dims=1)
#         # model.system.σ = std(reshape(new_x[1:end, :], (:)), dims=1)
#         update_M(
#             idx,
#             t_idx,
#             new_x,
#             exogenous_variables,
#             control_variables,
#             model.system.llrs,
#             model.system.μ,
#             model.system.σ
#         )
#         if i % 3 == 1
#             k_list = collect(50:5:450)
#             # k_list = collect(5:5:min(maximum(map(x->size(x.analogs, 2), model.system.llrs)), 1500))
#             opt_k, _ = k_choice(
#                 model.system,
#                 new_x[1:(end - 1), 1, :],
#                 new_x[2:(end), 1, :],
#                 exogenous_variables,
#                 control_variables,
#                 t_idx;
#                 k_list = k_list
#             )
#             for llr in model.system.llrs
#                 llr.k = opt_k
#             end
#         end
#     end

#     filter_output = filter(
#         model,
#         y_t,
#         exogenous_variables,
#         control_variables;
#         parameters = parameters,
#         filter = ConditionalParticleFilter(
#             model;
#             conditional_particle = conditional_particle,
#             p_filter...
#         )
#     )

#     push!(llk_array, filter_output.llk / n_obs)
#     println("Final | Log Likelihood : ", llk_array[end])

#     return parameters, (parameters = parameters_array, llk = llk_array, std = std_array)
# end
