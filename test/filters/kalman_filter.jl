using Distributions
using StateSpaceIdentification
using Plots


using PDMats
using StaticArrays
using SparseArrays


desired_type = Float64

x_0 = [2]
θ = Vector{desired_type}([0.9, 1, 1])
N = 100 #_000

x_t = Vector{desired_type}(UndefInitializer(), N)
y_t = Vector{desired_type}(UndefInitializer(), N)
x_t[1] = x_0[1]
y_t[1] = x_t[1] +  + rand(Normal(0, sqrt(θ[3])))
for i in 1:(N-1)
    x_t[i+1] = θ[1]*x_t[i] + rand(Normal(0, sqrt(θ[2])))
    y_t[i+1] = x_t[i+1] + rand(Normal(0, sqrt(θ[3])))
end

@inline A_t2(exogenous, params, t) = @SMatrix [params[1];;]
B_t2 = @SMatrix zeros(desired_type, 1, 1)
H_t2 = @SMatrix ones(desired_type, 1, 1)
@inline R_t2(exogenous, params, t) = PDiagMat([params[2]])
@inline Q_t2(exogenous, params, t) = PDiagMat([params[3]])
c_t2 = @SVector zeros(desired_type, 1)
d_t2 = @SVector zeros(desired_type, 1)

# Define the system
n_X = 1
n_Y = 1



dM_t(state, exogenous, control, params, t) = [params[1];;]
dH_t(state, exogenous, params, t) = [1.0;;]

glss2 = GaussianLinearStateSpaceSystem{desired_type}(
    A_t2, 
    B_t2, 
    c_t2, 
    H_t2, 
    d_t2, 
    R_t2, 
    Q_t2, 
    n_X, 
    n_Y, 
    desired_type(1.0)
)

# Define init state
init_P_0 = zeros(1, 1) .+   1 #0.001
init_state = GaussianStateStochasticProcess{desired_type}(0.0, x_0, init_P_0)

model2 = ForecastingModel(glss2, init_state, θ)

Y = hcat(y_t)
E = zeros(desired_type, N, 1)
U = zeros(desired_type, N, 1)

filter_output_kf = filtering(model2, Y, E, U)
filter_output_enkf = filtering(model2, Y, E, U; filtering_method = EnsembleKalmanFilter(model2, n_particles=1_000))
filter_output_ekf = filtering(model2, Y, E, U; filtering_method = ExtendedKalmanFilter(model2, dM_t, dH_t))
filter_output_pf = filtering(model2, Y, E, U; filtering_method = ParticleFilter(model2, n_particles=1_000))
filter_output_cpf = filtering(model2, Y, E, U; filtering_method = ConditionalParticleFilter(model2, n_particles=1))

plot(filter_output_kf.predicted_state, label="Kalman Filter", color="red")
plot!(filter_output_enkf.predicted_particles_swarm, label="Ensemble Kalman Filter", color="blue")
plot!(filter_output_ekf.predicted_state, label="Extended Kalman Filter", color="orange")
plot!(filter_output_pf.predicted_particles_swarm, label="Particle Filter", color="green")
plot!(filter_output_cpf.predicted_particles_swarm, label="Conditional Particle Filter", color="brown")


smoother_output_ks = smoothing(model2, Y, E, U, filter_output_kf)
smoother_output_enks = smoothing(model2, Y, E, U, filter_output_enkf, smoother_method=EnsembleKalmanSmoother(model2, n_particles=1_000))
smoother_output_eks = smoothing(model2, Y, E, U, filter_output_ekf, smoother_method=ExtendedKalmanSmoother(model2, dH_t))
smoother_output_pfat = smoothing(model2, Y, E, U, filter_output_pf, smoother_method=AncestorTrackingSmoother(model2, n_particles=300))
smoother_output_pfbs = smoothing(model2, Y, E, U, filter_output_pf, smoother_method=BackwardSimulationSmoother(model2, n_particles=1_000))

plot(smoother_output_ks.smoothed_state, label="Kalman Smoother", color="red")
plot!(smoother_output_enks.smoothed_particles_swarm, label="Ensemble Kalman Smoother", color="blue")
plot!(smoother_output_eks.smoothed_state, label="Extended Kalman Smoother", color="orange")
plot!(smoother_output_pfat.smoothed_particles_swarm, label="Ancestor Tracking with Particle Filter", color="green")
plot!(smoother_output_pfbs.smoothed_particles_swarm, label="Backward Simulation with Particle Filter", color="purple")


plot(filter_output.predicted_state, label="Predicted State")
plot!(filter_output.filtered_state, label="Filtered State")
plot!(smoother_output.smoothed_state, label="Smoothed State")
plot!(collect(0:size(x_t, 1)-1), x_t, label="Hidden state")
scatter!(collect(0:size(y_t, 1)-1), y_t, label="Observations")

@benchmark filtering(model2, Y, E, U)
# @benchmark StateSpaceIdentification.loglike(model2, Y, E, U)

model2.parameters = desired_type.([0.8, 1.3, 0.6])

using Optimization
using OptimizationOptimJL
using OptimizationNLopt
using OptimizationOptimisers
using ForwardDiff, Enzyme#, Zygote, Enzyme, ReverseDiff

optim_method = BFGS() #Newton()
diff_method = Optimization.AutoForwardDiff()#Optimization.AutoEnzyme() #Optimization.AutoReverseDiff() #Optimization.AutoEnzyme() #Optimization.AutoForwardDiff() #Optimization.AutoZygote()
StateSpaceIdentification.numerical_MLE(model2, Y, E, U, verbose=true, optim_method=optim_method, diff_method=diff_method)

@benchmark StateSpaceIdentification.numerical_MLE(model2, Y, E, U, verbose=false, optim_method=optim_method, diff_method=diff_method)
@profview StateSpaceIdentification.numerical_MLE(model2, Y, E, U, verbose=false, optim_method=optim_method, diff_method=diff_method)

optim_method = BFGS()# Adam()##NelderMead()#NLopt.LD_LBFGS() #BFGS() #GradientDescent(alphaguess=0.1) #
diff_method = Optimization.AutoForwardDiff()  # #Optimization.AutoForwardDiff() 
ExpectationMaximization(model2, Y, E, U, optim_method=optim_method, diff_method=diff_method, maxiters_em=30, filter_method = ExtendedKalmanFilter(model2, dM_t, dH_t), smoother_method = ExtendedKalmanSmoother(model2, dH_t))




@btime ExpectationMaximization(model2, Y, E, U, optim_method=optim_method, diff_method=diff_method, maxiters_em=5)
filtering(model2, Y, E, U)
@profview begin
    for i in 1:100
        a = filtering(model2, Y, E, U);
    end
end
