# StateSpaceIdentification.jl

StateSpaceIdentification.jl is a Julia package designed for filtering, smoothing, and parameter estimation of state-space models, including uncertainty quantification using variants of the Expectation-Maximization (EM) algorithm.

## Quick Start Example

Consider a simple linear state-space model defined as:

$$
\begin{cases}
x_t = A x_{t-1} + \eta_t, \\
y_t = x_t + \epsilon_t,
\end{cases}
$$

where \( \eta_t \) and \( \epsilon_t \) are independent Gaussian white noise sequences with variances \( R \) and \( Q \), respectively. The matrix \( A \) represents the autoregressive coefficient, and \( \theta = (A, Q, R) \) is the vector of unknown parameters.

### Defining the Model Components

```julia
using StateSpaceIdentification, PDMats, StaticArrays

@inline A(exogenous, params, t) = @SMatrix [params[1];;]
B = @SMatrix zeros(1, 1)
H = @SMatrix ones(1, 1)
@inline R(exogenous, params, t) = PDiagMat([params[2]])
@inline Q(exogenous, params, t) = PDiagMat([params[3]])
c = @SVector zeros(1)
d = @SVector zeros(1)
```

### Initializing the Model and Parameters

```julia
n_X = 1
n_Y = 1
glss = GaussianLinearStateSpaceSystem{Float64}(A, B, c, H, d, R, Q, n_X, n_Y, 1.0)

# Define initial state
x_0 = zeros(1)
init_P_0 = ones(1, 1)
init_state = GaussianStateStochasticProcess(0.0, x_0, init_P_0)

# Define model parameters
parameters = [0.9, 1.0, 1.0]
model = ForecastingModel(glss, init_state, parameters)
```

### Filtering and Smoothing

Using some generated data, we can apply filtering and smoothing procedures with default or user-specified filters.

```julia
Y = ...  # Observations
e = ...  # Exogenous inputs
U = ...  # Control inputs

# Filtering
filter_output_kf = filtering(model, Y, E, U)
filter_output_pf = filtering(model, Y, E, U; filter_method = ParticleFilter(model, n_particles=1_000))

# Smoothing
smoother_output_ks = smoothing(model, Y, E, U, filter_output_kf)
smoother_output_pfbs = smoothing(model, Y, E, U, filter_output_pf, smoother_method=BackwardSimulationSmoother(model, n_particles=1_000))
```

### Parameter Estimation

It is also possible to estimate the parameters of the model, as well as the uncertainty, using:

```julia
opt_parameters = ExpectationMaximization(model, Y, E, U, verbose=true)
```

## TODO

- Improve support for `StaticArrays`
- Performance optimization using `Jet` and `SnoopCompile`
- Implement parametric measurement transition for GaussianLinear and GaussianNonLinear models
- Refactor filters and smoothers to optimize memory usage
- Improve test coverage and documentation

---

This package is actively developed. Contributions, issues, and suggestions are welcome!

