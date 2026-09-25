# Oscar interoperability tests (tier 2; run by the LongTests workflow in its own environment).
using Test
using Oscar
using Syntomic

@testset "Oscar interoperability" begin
    # Syntomic depends only on Nemo and Hecke; with Oscar loaded, the types must be identical.
    @test Syntomic.Nemo === Oscar.Nemo
    @test Syntomic.Hecke === Oscar.Hecke
end
