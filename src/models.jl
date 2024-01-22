mutable struct ForecastingModel{System}

    # System structure
    system::System

    # Current state of the system
    current_state::AbstractState

    # Parameters of the system
    parameters

end


function forecast(model::ForecastingModel, exogenous_variables, control_variables; n_steps_ahead=1, kwargs...)

    return forecast(model.system, model.current_state, exogenous_variables, control_variables, model.parameters; n_steps_ahead=n_steps_ahead, kwargs...)

end