using NearestNeighbors
using Logging
import Base: convert

const AVAILABLE_KERNEL_TYPES = ["rectangular", "tricube"]
const TIME_PRECISION = 1 / 1440
#####################################################################################################################
############################################# Local Linear Regression ###############################################
#####################################################################################################################
"""
Structure for Local Linear Regression (LLR) based on temporal analogs.

This structure encapsulates the necessary data and parameters for performing 
Local Linear Regression using temporal analogs. It includes analog data, 
nearest neighbor tree for efficient search, and various hyperparameters 
to control the regression process.

$(TYPEDEF)

$(TYPEDFIELDS)
"""
mutable struct LocalLinearRegressor{Z <: Real, T <: NearestNeighbors.NNTree}
    """Time (in days, can be fractional) associated with each analog input."""
    analog_times_in_days::Vector{Z}

    """Matrix of input vectors corresponding to each analog (size: ``input_dim \\times n_analogs``)."""
    analog_inputs::Matrix{Z}

    """Matrix of output vectors corresponding to each analog (size: ``output_dim \\times n_analogs``)."""
    analog_outputs::Matrix{Z}

    """KDTree structure used for efficient nearest neighbor search."""
    neighbor_tree::T

    """Number of nearest neighbors to retrieve for local regression."""
    n_neighbors::Int64

    """Minimum lag (in days) required between a query point and a neighbor to avoid overfitting."""
    min_lag_in_days::Z

    """Kernel type used in the local linear regression ('rectangular', 'tricube', etc.)."""
    kernel_type::String

    """Maximum allowed cyclic lag (in days, modulo 1) between the query time and a neighbor (e.g., for seasonal filtering)."""
    max_cyclic_lag::Union{Z, Nothing}

    function LocalLinearRegressor(
            analog_times_in_days::Vector{Z},
            analog_inputs::Matrix{Z},
            analog_outputs::Matrix{Z},
            neighbor_tree::T;
            n_neighbors::Int64 = 10,
            min_lag_in_days::Z = 0.0,
            kernel_type::String = "rectangular",
            max_cyclic_lag::Union{Z, Nothing} = nothing
    ) where {Z <: Real, T <: NearestNeighbors.NNTree}
        new{Z, T}(
            analog_times_in_days,
            analog_inputs,
            analog_outputs,
            neighbor_tree,
            n_neighbors,
            min_lag_in_days,
            kernel_type,
            max_cyclic_lag
        )
    end
end

"""
Skip function to exclude visited indices.
"""
@inline function skip(i::Int; visited = Set())::Bool
    i ∈ visited
end

"""
Calculate kernel weights based on distances.
"""
function calculate_weights(distances, kernel_type)
    k = length(distances[1])
    n_particles = length(distances)
    weights = zeros(k, n_particles)
    if kernel_type == "tricube"
        max_dist = maximum(hcat(distances...), dims = 1)
        for i in 1:n_particles
            weights[:, i] = (1 .- (distances[i] ./ max_dist[i]) .^ 3) .^ 3
        end
    elseif kernel_type == "rectangular"# rectangular kernel (uniform weights)
        weights .= 1 / k
    else
        error("Unknown kernel_type = $(kernel_type). Avaible are $(AVAILABLE_KERNEL_TYPES).")
    end
    return weights ./ sum(weights, dims = 1)
end

"""
Local Linear Regression prediction function using temporal analogs.

This function predicts output values for given input data 'x' and time 't'
using Local Linear Regression (LLR) based on temporal analogs. It filters
neighbors based on temporal constraints and applies kernel-weighted regression.
"""
function (llr::LocalLinearRegressor{Z, T})(
        x::AbstractMatrix{Z}, t::Z) where {Z <: Real, T <: NearestNeighbors.NNTree}
    # Extract parameters from the Local Linear Regressor model
    n_neighbors = llr.n_neighbors
    min_lag_in_days = llr.min_lag_in_days
    kernel_type = llr.kernel_type
    analog_inputs = llr.analog_inputs # (input_dim, n_analogs)
    analog_outputs = llr.analog_outputs # (output_dim, n_analogs)
    analog_times_in_days = llr.analog_times_in_days # (n_analogs,)
    neighbor_tree = llr.neighbor_tree
    max_cyclic_lag = llr.max_cyclic_lag

    n_particles = size(x, 2)
    n_analogs = length(analog_times_in_days)

    # Filter neighbors based on temporal lag constraints
    ignored_nodes = Set{Int64}()
    if t != 0.0
        n = length(analog_times_in_days)
        ignored_nodes_bool = BitVector(undef, n)
        t_mod = t % 1.0 # Time modulo 1 day

        # Condition 1: Check indices within the range [t - min_lag_in_days, t + min_lag_in_days]
        if min_lag_in_days != 0.0
            in_range = (t - min_lag_in_days - TIME_PRECISION .<= analog_times_in_days) .&
                    (analog_times_in_days .<= t + min_lag_in_days + TIME_PRECISION)
        else
            in_range = repeat([false], n_analogs)
        end

        # Condition 2: Check indices modulo within the temporal range (for cyclic lag)
        if !isnothing(max_cyclic_lag)
            mod_indexes = analog_times_in_days .% 1
            out_of_mod_range = (
                (mod_indexes .<= t_mod - max_cyclic_lag) .&
                (mod_indexes .>= max(t + max_cyclic_lag, 1) % 1)
            ) .| ((mod_indexes .>= t_mod + max_cyclic_lag) .&
                                (mod_indexes .<= 1 + t - max_cyclic_lag))

            # Combiner les conditions
            ignored_nodes_bool .= in_range .| (max_cyclic_lag != 0 && out_of_mod_range)
        else
            ignored_nodes_bool .= in_range
        end

        ignored_nodes = Set(findall(ignored_nodes_bool))
    end

    # Determine the number of neighbors to use (k), adjusting for ignored nodes
    k = min(n_neighbors, n_analogs - length(ignored_nodes))

    if k < n_neighbors
        @warn "Number of selected neighbors reduced to $(k) due to constraints with max_cyclic_lag and min_lag_in_days."
    end

    if k < 2
        error("Not enough valid neighbors after lag filtering.")
    end

    # Find the k nearest neighbors, skipping ignored nodes
    knn_result = knn(
        neighbor_tree, x, k, false, (idx) -> skip(idx, visited = ignored_nodes))
    knn_indices = knn_result[1] # Vector{Vector{Int64}} (length n_particles)
    knn_distances = knn_result[2] # Vector{Vector{Float64}} (length n_particles)

    # Calculate kernel weights based on distances
    weights = calculate_weights(knn_distances, kernel_type)  # (k, n_particles)

    # Compute matrices A and B for local linear regression
    A = [vcat(ones(1, k), analog_inputs[:, knn_indices[i]]) .*
         reshape(weights[:, i], (1, k)) for i in 1:n_particles] # Vector{Matrix{Float64}} (length n_particles, each element (input_dim + 1, k))
    B = [analog_outputs[:, knn_indices[i]] .* weights[:, i]' for i in 1:n_particles] # Vector{Matrix{Float64}} (length n_particles, each element (output_dim, k))

    # Solve linear systems to get regression coefficients
    M = [A[i]' \ B[i]' for i in 1:n_particles] # Vector{Matrix{Float64}} (length n_particles, each element (input_dim + 1, output_dim))

    # Predict values using local linear regression
    x_extended = [vcat(ones(1, 1), x[:, i]) for i in 1:n_particles] # Vector{Matrix{Float64}} (length n_particles, each element (input_dim + 1, 1))
    predictions = [M[i]' * x_extended[i] for i in 1:n_particles] # Vector{Matrix{Float64}} (length n_particles, each element (1, output_dim))

    return hcat(predictions...) # Concatenate and return the predictions
end

#####################################################################################################################
####################################### GaussianNonParametricStateSpaceSystem #######################################
#####################################################################################################################

mutable struct GaussianNonParametricStateSpaceSystem{Z <: Real} <:
               AbstractNonLinearStateSpaceSystem{Z}

    # General components of gaussian non parametric state space systems
    """Provider ``M_t`` which is a ``n_X -> n_X`` function."""
    M_t::TransitionNonParametricProvider{Z}
    """Provider ``H_t`` which is a ``n_X -> n_Y`` function."""
    H_t::ObservationNonLinearProvider{Z}
    """Provider ``R_t`` returning a ``n_X \\times n_X`` matrix."""
    R_t::AbstractMatrixProvider{Z}
    """Provider ``Q_t`` returning a ``n_Y \\times n_Y`` matrix."""
    Q_t::AbstractMatrixProvider{Z}

    """Number of state variables."""
    n_X::Int
    """Number of observations."""
    n_Y::Int
    """Time between two timesteps in seconds."""
    dt::Z

    # Additional attributes for local linear regression structure
    llrs::Vector{LocalLinearRegressor{Z}}
    μ::Vector{Z}
    σ::Vector{Z}

    """Constructor with full arguments."""
    function GaussianNonParametricStateSpaceSystem{Z}(
            M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs;
            μ = zeros(n_X), σ = ones(n_X)) where {Z <: Real}
        return new{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs, μ, σ)
    end

    """Constructor with Type conversion."""
    function GaussianNonParametricStateSpaceSystem{Z}(
            M_t::Union{Function, TransitionNonParametricProvider{Z}},
            H_t::Union{Function, ObservationNonLinearProvider{Z}},
            R_t::Union{MatOrFun, AbstractMatrixProvider{Z}},
            Q_t::Union{MatOrFun, AbstractMatrixProvider{Z}},
            n_X, n_Y, dt, llrs; μ = zeros(n_X), σ = ones(n_X)) where {Z <: Real}

        # Convert types
        M_t = isa(M_t, TransitionNonParametricProvider) ? M_t :
              TransitionNonParametricProvider{Z}(M_t)
        H_t = isa(H_t, ObservationNonLinearProvider) ? H_t :
              ObservationNonLinearProvider{Z}(H_t)
        R_t = isa(R_t, AbstractMatrixProvider) ? R_t :
              (isa(R_t, Matrix) ? StaticMatrix{Z}(R_t) : DynamicMatrix{Z}(R_t))
        Q_t = isa(Q_t, AbstractMatrixProvider) ? Q_t :
              (isa(Q_t, Matrix) ? StaticMatrix{Z}(Q_t) : DynamicMatrix{Z}(Q_t))

        return new{Z}(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, llrs, μ, σ)
    end
end

"""
Updates the Local Linear Regressors (LLRs) within a GaussianNonParametricStateSpaceSystem.

$(TYPEDSIGNATURES)

This function updates the `llrs` field of the given state-space system (`ssm`). 
It uses either a provided `update_function!` or a default update function 
if no custom function is provided and the `llrs` list contains only one element.
"""
function update_llrs!(
        ssm::GaussianNonParametricStateSpaceSystem{Z},
        analog_indexes,
        analog_times_in_days,
        analog_states,
        exogenous_data,
        control_data;
        update_function!::Union{Function, Nothing} = nothing
) where {Z <: Real}
    if update_function! !== nothing
        update_function!(ssm.llrs, analog_indexes, analog_times_in_days,
            analog_states, exogenous_data, control_data, ssm.μ, ssm.σ)
    elseif length(ssm.llrs) == 1
        default_llrs_update!(ssm.llrs, analog_indexes, analog_times_in_days,
            analog_states, exogenous_data, control_data, ssm.μ, ssm.σ)
    else
        error("Cannot use default update function when length(ssm.llrs) != 1. Provide an update_function!.")
    end
end

"""
Default update function for Local Linear Regressors (LLRs) when the list contains only one element.

$(TYPEDSIGNATURES)

This function updates the `llrs` field with new analog inputs, outputs, times, and neighbor tree.
It assumes that `llrs` contains only one element.
"""
function default_llrs_update!(
        llrs::Vector{LocalLinearRegressor{Z}},
        analog_indexes,
        analog_times_in_days,
        analog_states,
        exogenous_data,
        control_data,
        μ,
        σ
) where {Z <: Real}
    n_X = size(analog_states, 3)

    # Construct new analog inputs by scaling and concatenating state and exogenous data
    scaled_states = _scale(reshape(analog_states[1:(end - 1), :, :], (:, n_X))', μ, σ)'
    new_analog_inputs = transpose(hcat(scaled_states, exogenous_data[analog_indexes, :]))

    # Construct new analog outputs by calculating the difference between consecutive states
    new_analog_outputs = reshape(
        analog_states[2:end, :, :] - analog_states[1:(end - 1), :, :], (:, n_X))

    # Update the first LLR in the list
    llrs[1].analog_times_in_days = analog_times_in_days
    llrs[1].analog_inputs = new_analog_inputs
    llrs[1].analog_outputs = new_analog_outputs'
    llrs[1].neighbor_tree = KDTree(new_analog_inputs)
end

"""
$(TYPEDSIGNATURES)

The ``default_filter`` for ``GaussianNonParametricStateSpaceSystem`` is the ``ParticleFilter``.
"""
function default_filter(
        model::ForecastingModel{Z, GaussianNonParametricStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return ParticleFilter(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``default_smoother`` for ``GaussianNonParametricStateSpaceSystem`` is the ``BackwardSimulationSmoother``.
"""
function default_smoother(
        model::ForecastingModel{Z, GaussianNonParametricStateSpaceSystem{Z}, S};
        kwargs...) where {Z <: Real, S <: AbstractState{Z}}
    return BackwardSimulationSmoother(model; kwargs...)
end

"""
$(TYPEDSIGNATURES)

The ``transition`` function for ``GaussianNonParametricStateSpaceSystem`` which is ``\\x{t+dt} &= M_t(\\x{t})``.
"""
function transition(
        ssm::GaussianNonParametricStateSpaceSystem{Z},
        state_variables::AbstractVecOrMat{Z},
        exogenous_variables::AbstractVector{Z},
        control_variables::AbstractVector{Z},
        parameters::AbstractArray,#Vector{Z},
        t::Z
) where {Z <: Real}#, A <: AbstractArray{Z}}
    return ssm.M_t(state_variables, _scale(state_variables, ssm.μ, ssm.σ),
        exogenous_variables, control_variables, parameters, ssm.llrs, t)
end

"""
$(TYPEDSIGNATURES)

The ``observation`` function for ``GaussianNonParametricStateSpaceSystem`` which is ``y_{t} &=  H_t \\x{t} + d_t + \\epsilon_{t} \\quad &\\epsilon_{t} \\sim \\mathcal{N}(0, Q_t)``
"""
function observation(
        ssm::GaussianNonParametricStateSpaceSystem{Z},
        state_variables::AbstractVecOrMat{Z},
        exogenous_variables::AbstractVector{Z},
        parameters::AbstractArray,#Vector{Z},
        t::Z
) where {Z <: Real}
    return ssm.H_t(state_variables, exogenous_variables, parameters, t)
end


"""
Finds the optimal number of neighbors for Local Linear Regression.

$(TYPEDSIGNATURES)

This function evaluates the performance of the LLR model for different numbers 
of neighbors and returns the number of neighbors that minimizes the error.
"""
function find_optimal_number_neighbors(
        ssm::GaussianNonParametricStateSpaceSystem{Z},
        parameters,
        inputs_states::Matrix{Z},
        target_states::Matrix{Z},
        exogenous_data::Matrix{Z},
        control_data::Matrix{Z},
        time_data::Vector{Z},
        neighbors_list::Vector{Int64}
) where {Z <: Real}
    n_t = size(inputs_states, 1) # Number of time steps
    n_llrs = length(ssm.llrs) # Number of LLR models in the system

    E = zeros(length(neighbors_list)) # Vector to store errors for each neighbor count

    for (index_neighbors, num_neighbors) in enumerate(neighbors_list)
        # Set the number of neighbors for all LLR models in the system
        for i in 1:n_llrs
            ssm.llrs[i].n_neighbors = num_neighbors
        end

        try
            err = 0.0 # Initialize error for this neighbor count
            for idx in 1:n_t
                local mean_xf
                # Predict output using the transition function of the state space system
                my_logger = ConsoleLogger(stderr, Logging.Error)
                with_logger(my_logger) do
                    mean_xf = transition(
                    ssm,
                    reshape(inputs_states[idx, :], :, 1),
                    exogenous_data[idx, :],
                    control_data[idx, :],
                    parameters,
                    time_data[idx]
                )
                end
                
                # Calculate innovation (error)
                innov = target_states[idx, :] .- mean_xf

                # Accumulate the squared error
                err += mean((innov) .^ 2)
            end
            E[index_neighbors] = sqrt(err) / n_t # Calculate Root Mean Squared Error (RMSE)
        catch e
            println("Error with num_neighbors = $num_neighbors: $e") # Print the error
            E[index_neighbors] = Inf # Set error to infinity if there's an exception
        end
    end

    ind_min = argmin(E) # Find the index of the minimum error
    optimal_num_neighbors = neighbors_list[ind_min] # Get the optimal number of neighbors

    return optimal_num_neighbors, E # Return the optimal number of neighbors and the error vector
end
