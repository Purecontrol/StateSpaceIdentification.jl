@testset "TimeSeries" begin

    @testset "GaussianStateStochasticProcess" begin

        undef_state = GaussianStateStochasticProcess(10)
        @test size(undef_state.μ_t, 1) == 10
        @test size(undef_state.Σ_t) == (10, 10)

        undef_state = GaussianStateStochasticProcess(1.0, 1)
        @test undef_state.t == 1.0 
        @test size(undef_state.μ_t, 1) == 1
        @test size(undef_state.Σ_t) == (1, 1)


        multidim_state = GaussianStateStochasticProcess(0.0, [1.0, 2.0], [2.0 0.0;0.0 1.0])
        @test typeof(multidim_state.t) == Float64
        @test typeof(multidim_state.μ_t) == Vector{Float64}
        @test typeof(multidim_state.Σ_t) == Matrix{Float64}
    
        multidim_state_float32 = GaussianStateStochasticProcess{Float32}(0.0, [1.0], [2.0;;])
        @test typeof(multidim_state_float32.t) == Float32
        @test typeof(multidim_state_float32.μ_t) == Vector{Float32}
        @test typeof(multidim_state_float32.Σ_t) == Matrix{Float32}

    end

    @testset "ParticleSwarmState" begin



    end
    
    @testset "TimeSeries" begin

        

    end

end