using Test
using Syntomic

# Test tiers (see README):
#   Tier 1 (default, CI):       always run.
#   Tier 2 (long):              SYNTOMIC_LONG_TESTS=1
#   Tier 3 (distributed):       SYNTOMIC_DISTRIBUTED_TESTS=1
const LONG_TESTS = get(ENV, "SYNTOMIC_LONG_TESTS", "0") == "1"
const DISTRIBUTED_TESTS = get(ENV, "SYNTOMIC_DISTRIBUTED_TESTS", "0") == "1"

include("helpers.jl")

@testset "Syntomic.jl" begin
    @testset "environment" begin
        # CI runs with JULIA_NUM_THREADS=2 so that the threaded code paths are exercised.
        if get(ENV, "CI", "false") == "true"
            @test Threads.nthreads() > 1
        end
    end

    include("test_params_precision.jl")
    include("test_galois_ring.jl")
    include("test_wseries.jl")
    include("test_linalg.jl")
    include("test_input.jl")
    include("test_envelopes.jl")
    include("test_end_to_end.jl")
    include("test_fixtures.jl")
    include("test_sweep.jl")
    include("test_infra.jl")
    include("test_style.jl")
    include("aqua.jl")
end
