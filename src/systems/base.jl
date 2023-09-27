@doc raw"""
    StateSpaceSystem

StateSpaceSystem is an abstract type describing a state-space system, which is a global class for models having a transition
equation relating hidden states at time `t` (`x_t`), to the states at time `t+1` (`x_{t+1}`), and an observation equation
relating hidden states `x_t` to observations `y_t`.
"""
abstract type StateSpaceSystem end

function (ssm::StateSpaceSystem)(current_x, exogenous_variables, control_variables, parameters) end