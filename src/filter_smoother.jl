abstract type FilterOutput end

abstract type SMCFilterOutput <: FilterOutput end


function filter(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter=nothing, parameters=model.parameters, kwargs...)

    if isnothing(filter)
        filter = default_filter(model; kwargs...)
    end

    if !(isa(filter, AbstractFilter))
        @error "Le filtre doit être de type AbstractFilter."
    end

    filter_output = get_filter_output(filter, model, y_t)

    return filter!(filter_output, model.system, filter, y_t, exogenous_variables, control_variables, parameters)

end


function update(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter_method=nothing, parameters=model.parameters, kwargs...)

    if isnothing(filter_method)
        filter_method = default_filter(model; kwargs...)
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; filter=filter_method, parameters=parameters)

    new_model = deepcopy(model)
    new_model.current_state = get_last_state(filter_output)

    return new_model, filter_output

end


function update!(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter_method=nothing, parameters=model.parameters, kwargs...)

    if isnothing(filter_method)
        filter_method = default_filter(model; kwargs...)
    end

    filter_output = filter(model, y_t, exogenous_variables, control_variables; filter=filter_method, parameters=parameters)

    model.current_state = get_last_state(filter_output)

    return filter_output

end


function loglike(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter::AbstractFilter=default_filter(model), parameters=model.parameters)

    return filter!(model.system, deepcopy(filter), y_t, exogenous_variables, control_variables, parameters)

end


abstract type SmootherOutput end


# Smoother + Filter
function smoother(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter_method::AbstractFilter=default_filter(model), smoother_method::AbstractSmoother=default_smoother(model), parameters=model.parameters)

    # Apply filtering
    filter_output = filter(model, y_t, exogenous_variables, control_variables; filter=filter_method, parameters=parameters)
    
    # Smoothing step
    return smoother(model, y_t, exogenous_variables, control_variables, filter_output; smoother_method=smoother_method, parameters=parameters)

end


function smoother(model::ForecastingModel, y_t, exogenous_variables, control_variables, filter_output::FilterOutput; smoother_method::AbstractSmoother=default_smoother(model), parameters=model.parameters)

    smoother_output = get_smoother_output(smoother_method, model, y_t)

    return smoother!(smoother_output, filter_output, model.system, smoother_method, y_t, exogenous_variables, control_variables, parameters)

end