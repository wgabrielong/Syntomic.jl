# Oscar interoperability tests (tier 2; run by the LongTests workflow in its own environment).
using Test
using Oscar
using Syntomic

@testset "Oscar interoperability" begin
    # Syntomic depends only on Nemo and Hecke; with Oscar loaded, the types must be identical.
    @test Syntomic.Nemo === Oscar.Nemo
    @test Syntomic.Hecke === Oscar.Hecke

    @testset "inputs constructed with Oscar" begin
        K, a = qadic_field(2, 2; precision=30)
        Kz, z = polynomial_ring(K, :z)
        E = z^2 + 2a * z + 2a
        X = finite_chain_ring(E, 5)
        @test (prime(X), inertia_degree(X), ramification_index(X)) == (2, 2, 2)
        # K_1 = principal units of O_L/ϖ^5, L = Q_4(ϖ) (checked by brute force in the main tests)
        @test syntomic_cohomology(X, 1).h1 == [1, 1, 1, 2, 3]
        L, _ = eisenstein_extension(E, :b)
        @test finite_chain_ring(L, 5) == X
        @test finite_chain_ring(residue_ring(ZZ, 16)[1]) == zpn(2, 4)
        Qp = padic_field(3; precision=10)
        Qz, y = polynomial_ring(Qp, :y)
        @test finite_chain_ring(y + 3, 4) == zpn(3, 4)
        Zt, t = polynomial_ring(ZZ, :t)
        @test finite_chain_ring(t + 3, 2; p=3, f=2) == truncated_witt_vectors(3, 2, 2)
    end

    @testset "K-groups as Oscar groups" begin
        k = k_groups(zpn(2, 2), 2; integral=true)                 # K_3(Z/4) = Z/24, K_2 = Z/2
        @test is_isomorphic(k.odd, abelian_group([24]))
        @test is_isomorphic(k.even, abelian_group([2]))
        @test k.odd isa FinGenAbGroup
    end
end
