@testset "syntomic cohomology end to end" begin
    coh(X, i; kw...) = (H = syntomic_cohomology(X, i; kw...); (H.h1, H.h2))

    @testset "Z/p^n ([AKN, Appendix A, pp. 61–64])" begin
        # (p, n, i) => (exponents of K_{2i-1}, exponents of K_{2i-2})
        table = Dict(
            (2, 2, 1) => ([1], Int[]),
            (2, 2, 2) => ([3], [1]),
            (2, 3, 1) => ([1, 1], Int[]),
            (2, 3, 2) => ([2, 3], [1]),
            (2, 3, 3) => ([1, 6], [1]),
            (2, 3, 4) => ([4, 4], Int[]),
            (3, 2, 2) => ([1, 1], Int[]),
            (3, 2, 3) => ([4], [1]),
            (3, 3, 2) => ([1, 3], Int[]),
            (5, 2, 3) => ([3], Int[]),
            (5, 3, 4) => ([1, 7], Int[]),
            (7, 2, 3) => ([3], Int[]),
            (7, 3, 2) => ([4], Int[]),
            (2, 4, 2) => ([3, 4], [1]),
        )
        for ((p, n, i), expected) in table
            @test coh(zpn(p, n), i) == expected
        end
        # n = 1: R = F_p has no p-adic K-theory in positive degrees
        for (p, i) in ((2, 3), (3, 2), (5, 4))
            @test coh(zpn(p, 1), i) == (Int[], Int[])
        end
    end

    @testset "W(F_q)-cases: brute force for i = 1 and σ-patched references" begin
        K, a = qadic_field(2, 2; precision=30, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        K3, b = qadic_field(3, 2; precision=30, cached=false)
        K3z, y = polynomial_ring(K3, :z; cached=false)
        K8, c = qadic_field(2, 3; precision=30, cached=false)
        K8z, w = polynomial_ring(K8, :z; cached=false)
        # K_1 = principal units (1 + ϖO_K)/(1 + ϖ^n O_K)
        @test coh(truncated_witt_vectors(2, 2, 3), 1) == ([1, 1, 2], Int[])
        @test coh(finite_chain_ring(z^2 + 2, 3), 1) == ([2, 2], Int[])
        @test coh(finite_chain_ring(z^2 + 2a * z + 2a, 5), 1) == ([1, 1, 1, 2, 3], Int[])
        @test coh(finite_chain_ring(y^2 + 3b * y + 3, 3), 1) == ([1, 1, 1, 1], Int[])
        @test coh(finite_chain_ring(w + 2, 3), 1) == ([1, 1, 2, 2], Int[])
        # weight 2 and 3
        @test coh(finite_chain_ring(z^2 + 2a * z + 2, 4), 2) == ([1, 1, 2, 2, 3, 4], [1])
        @test coh(finite_chain_ring(w + 2, 2), 2) == ([2, 2, 3], [1])
        @test coh(truncated_witt_vectors(5, 2, 3), 2) == ([4, 4], Int[])
    end

    @testset "invariance (isomorphic rings give isomorphic cohomology)" begin
        K, a = qadic_field(2, 2; precision=30, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        E = z^2 + 2a * z + 2
        Eσ = z^2 + 2 * frobenius(a) * z + 2
        Eaz = (a * z)^2 + 2a * (a * z) + 2                    # E(az): another uniformizer
        for (n, i) in ((4, 2), (3, 3))
            H = coh(finite_chain_ring(E, n), i)
            @test coh(finite_chain_ring(Eσ, n), i) == H
            @test coh(finite_chain_ring(Eaz, n), i) == H
        end
        # z + 2 (descent to f = 1 arithmetic) versus a·z + 2 (full W(F_4)-arithmetic)
        @test coh(finite_chain_ring(z + 2, 2), 3) == coh(finite_chain_ring(a * z + 2, 2), 3)
        @test coh(finite_chain_ring(z + 2, 3), 2; descent=false) ==
            coh(finite_chain_ring(z + 2, 3), 2)
        # R only depends on E modulo p^⌈n/e⌉: z^2 + 2 and z^2 + 4z + 2 at n = 3
        Zz, t = polynomial_ring(ZZ, :z)
        for i in 1:3
            @test coh(finite_chain_ring(t^2 + 2, 3; p=2), i) ==
                coh(finite_chain_ring(t^2 + 4t + 2, 3; p=2), i)
        end
    end

    @testset "precision policies, determinism and soundness" begin
        for (X, i) in ((zpn(2, 3), 5), (zpn(3, 2), 6), (truncated_witt_vectors(3, 2, 2), 3))
            H0 = coh(X, i; precision_policy=:apriori)
            @test coh(X, i; precision_policy=:exponent) == H0
            @test coh(X, i; precision_policy=:adaptive) == H0
            N = S.apriori_working_precision(X.p, X.n, i)
            @test coh(X, i; N=N + 10) == H0
        end
        K, a = qadic_field(2, 2; precision=30, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        X = finite_chain_ring(z^2 + 2a * z + 2a, 3)
        C1 = syntomic_complex(X, 3; threaded=false)
        C2 = syntomic_complex(X, 3; threaded=true)
        @test C1.syn0.M == C2.syn0.M && C1.syn1.M == C2.syn1.M
        # too little precision fails loudly instead of returning a wrong answer
        @test_throws PrecisionExhausted syntomic_cohomology(zpn(2, 3), 6; N=20)
    end

    @testset "edge cases" begin
        @test coh(zpn(3, 1), 1) == (Int[], Int[])            # L = 1: the complex is zero
        @test coh(zpn(7, 2), 1) == ([1], Int[])               # i = 1, L ≤ p
        Zz, t = polynomial_ring(ZZ, :z)
        @test coh(finite_chain_ring(t^3 + 2, 3; p=2), 2) == ([1, 3], Int[])   # n ≤ e
    end

    @testset "K-groups" begin
        k = k_groups(zpn(2, 2), 2)                            # K_3(Z/4; Z_2) = Z/8, K_2 = Z/2
        @test k.degrees == (3, 2)
        @test is_isomorphic(k.odd, abelian_group([8]))
        @test is_isomorphic(k.even, abelian_group([2]))
        kint = k_groups(zpn(2, 2), 2; integral=true)           # K_3(Z/4) = Z/24
        @test is_isomorphic(kint.odd, abelian_group([24]))
        k1 = k_groups(zpn(2, 3), 1; integral=true)             # K_1(Z/8) = (Z/2)², K_0 = Z
        @test is_isomorphic(k1.odd, abelian_group([2, 2]))
        @test is_isomorphic(k1.even, abelian_group([0]))
    end
end
