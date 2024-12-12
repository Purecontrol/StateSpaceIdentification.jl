"""
$(TYPEDSIGNATURES)

Call the filtering process of the choosen method on the provided data (``observation_data``, ``exogenous_data`` 
and ``control_data``) and the ``model``
"""
function filtering(
        model::ForecastingModel,
        observation_data::Matrix,
        exogenous_data::Matrix,
        control_data::Matrix;
        filtering_method = nothing,
        parameters = model.parameters,
        kwargs...
)
    if isnothing(filtering_method)
        filtering_method = default_filter(model; kwargs...)
    end

    if !(isa(filtering_method, AbstractFilter))
        @error "The type of the filtering method need a subtype of AbstractFilter."
    end

    filter_output = get_filter_output(filtering_method, model, observation_data)

    return filtering!(
        filter_output,
        model.system,
        filtering_method,
        observation_data,
        exogenous_variables,
        control_variables,
        parameters
    )
end

"""
$(TYPEDSIGNATURES)

Call the filtering process of the choosen method on the provided data (``observation_data``, ``exogenous_data`` 
and ``control_data``) and the ``model`` and returns a new model with updated ``current_state``.
"""
function update(
        model::ForecastingModel,
        observation_data,
        exogenous_data,
        control_data;
        filter_method = nothing,
        parameters = model.parameters,
        kwargs...
)
    if isnothing(filter_method)
        filter_method = default_filter(model; kwargs...)
    end

    filter_output = filtering(
        model,
        observation_data,
        exogenous_data,
        control_data;
        filter = filter_method,
        parameters = parameters
    )

    new_model = deepcopy(model)
    new_model.current_state = get_last_state(filter_output)

    return new_model, filter_output
end

"""
$(TYPEDSIGNATURES)

Call the filtering process of the choosen method on the provided data (``observation_data``, ``exogenous_data`` 
and ``control_data``) and the ``model`` and update the ``current_state`` of the model.
"""
function update!(
        model::ForecastingModel,
        observation_data,
        exogenous_data,
        control_data;
        filter_method = nothing,
        parameters = model.parameters,
        kwargs...
)
    if isnothing(filter_method)
        filter_method = default_filter(model; kwargs...)
    end

    filter_output = filtering(
        model,
        observation_data,
        exogenous_data,
        control_data;
        filter_method = filter_method,
        parameters = parameters
    )

    model.current_state = get_last_state(filter_output)

    return filter_output
end

"""
$(TYPEDSIGNATURES)

Call the filtering process of the choosen method on the provided data (``observation_data``, ``exogenous_data`` 
and ``control_data``) to compute the likelihood of the model with respect of the observations.
"""
function loglike(
        model::ForecastingModel,
        observation_data,
        exogenous_data,
        control_data;
        filter_method::AbstractFilter = default_filter(model),
        parameters = model.parameters
)
    return filtering!(
        model.system,
        deepcopy(filter_method),
        observation_data,
        exogenous_data,
        control_data,
        parameters
    )
end

"""
$(TYPEDSIGNATURES)

Call the smoothing process of the choosen method on the provided data (``observation_data``, ``exogenous_data`` 
and ``control_data``) and the ``model`` using a previous output from a filtering process ``filter_output``.
"""
function smoothing(
        model::ForecastingModel,
        observation_data,
        exogenous_data,
        control_data,
        filter_output::AbstractFilterOutput;
        smoother_method::AbstractSmoother = default_smoother(model),
        parameters = model.parameters
)
    smoother_output = get_smoother_output(smoother_method, model, observation_data)::AbstractSmootherOutput

    return smoothing!(
        smoother_output,
        filter_output,
        model.system,
        smoother_method,
        observation_data,
        exogenous_data,
        control_data,
        parameters
    )
end

"""
$(TYPEDSIGNATURES)

Call the filtering and the smoothing process of the choosen method on the provided data (``observation_data``, ``exogenous_data`` 
and ``control_data``) and the ``model``.
"""
function filtering_and_smoothing(
        model::ForecastingModel,
        observation_data,
        exogenous_data,
        control_data;
        filter_method::AbstractFilter = default_filter(model),
        smoother_method::AbstractSmoother = default_smoother(model),
        parameters = model.parameters
)

    # Apply filtering
    filter_output = filtering(
        model,
        observation_data,
        exogenous_data,
        control_data;
        filter_method = filter_method,
        parameters = parameters
    )

    # Smoothing step
    return smoothing(
        model,
        observation_data,
        exogenous_data,
        control_data,
        filter_output;
        smoother_method = smoother_method,
        parameters = parameters
    )
end