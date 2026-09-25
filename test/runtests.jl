using Test
using Syntomic

# Test tiers (see README):
#   Tier 1 (default, CI):       always run.
#   Tier 2 (long):              SYNTOMIC_LONG_TESTS=1
#   Tier 3 (distributed):       SYNTOMIC_DISTRIBUTED_TESTS=1
const LONG_TESTS = get(ENV, "SYNTOMIC_LONG_TESTS", "0") == "1"
const DISTRIBUTED_TESTS = get(ENV, "SYNTOMIC_DISTRIBUTED_TESTS", "0") == "1"

@testset "Syntomic.jl" begin
    @testset "environment" begin
        # CI runs with JULIA_NUM_THREADS=2 so that the threaded code paths are exercised.
        if get(ENV, "CI", "false") == "true"
            @test Threads.nthreads() > 1
        end
        @test isdefined(Syntomic, :Nemo)
    end

    include("aqua.jl")
end
