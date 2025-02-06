using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(StateSpaceIdentification, stale_deps = false)
end