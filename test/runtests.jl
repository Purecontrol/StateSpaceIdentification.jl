using StateSpaceIdentification
using Test
using Aqua

# @testset "Aqua.jl" begin
#     Aqua.test_all(StateSpaceIdentification)
# end

include("time_series.jl")

# @test begin
    
#     a = TimeSeries{GaussianStateStochasticProcess}(10, 1, [GaussianStateStochasticProcess(t, ones(1), ones(1,1)) for t in 1:10])
#     plot(a)

# end

# @test begin

#     b = TimeSeries{StateSpaceIdentification.ParticleSwarmState}(10, 1, [StateSpaceIdentification.ParticleSwarmState(10, t, zeros(Float64, 1, 10) + rand(1, 10)) for t in 1:10])
#     plot(b)

# end
