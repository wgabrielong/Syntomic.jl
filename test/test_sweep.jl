@testset "sweeps and Distributed" begin
    @testset "sweep equals individual computations" begin
        K, a = qadic_field(2, 2; precision=30, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        for (X, ns, is) in (
            (zpn(2, 3), 1:3, 1:5),
            (truncated_witt_vectors(3, 2, 2), 2:3, 1:3),
            (finite_chain_ring(z^2 + 2a * z + 2a, 3), 2:3, 1:3),
        )
            R = syntomic_sweep(X, ns, is)
            @test length(R) == length(ns) * length(is)
            @test issorted([(H.ring.n, H.i) for H in R])
            for H in R
                Hind = syntomic_cohomology(S.with_power(X, H.ring.n), H.i)
                @test (H.h1, H.h2) == (Hind.h1, Hind.h2)
            end
        end
        @test_throws ArgumentError syntomic_sweep(
            zpn(2, 2), 1:2, 1:2; precision_policy=:adaptive
        )
        @test isempty(syntomic_sweep(zpn(2, 2), Int[], 1:2))
        # the policy that chose N is recorded
        R = syntomic_sweep(zpn(2, 2), 2:2, 1:3; precision_policy=:apriori)
        @test all(H.policy == :apriori for H in R)
        @test [H.N for H in R] == [S.apriori_working_precision(2, 2, i) for i in 1:3]
        @test all(H.policy == :exponent for H in syntomic_sweep(zpn(2, 2), 2:2, 1:2))
        # rings whose normalized E lies in Z_p[z] only modulo a power of p: the sweep and each
        # of its pairs must use the same presentation
        for (X, ns, is) in (
            (finite_chain_ring(2, [1, 1, 1], [[2, big(2)^5], [1, 0]], 2), [2], [1, 4]),
            (
                finite_chain_ring(2, [1, 1, 1], [[2, big(2)^18], [1, big(2)^17]], 5),
                [5],
                1:2,
            ),
            (finite_chain_ring(z^2 + 16a * z + 2, 3), 2:3, 1:3),
        )
            R = syntomic_sweep(X, ns, is)
            for H in R
                Hind = syntomic_cohomology(S.with_power(X, H.ring.n), H.i)
                @test (H.h1, H.h2) == (Hind.h1, Hind.h2)
            end
            @test [r.exponents for r in k_theory_table(S.with_power(X, first(ns)), is)] == [
                r.exponents for
                r in k_theory_table(S.with_power(X, first(ns)), is; descent=false)
            ]
        end
        # the normalized E does not depend on the working precision (shared data are reduced)
        for X in (
            finite_chain_ring(2, [0, 1], [[-2], [1]], 5),
            finite_chain_ring(2, [1, 1, 1], [[2, 2], [0, 2], [1, 0]], 5),
        )
            function E(N)
                return S.ws_coefficients(
                    S.normalize_eisenstein(S.GaloisRing(2, X.htilde, N), X, 3)
                )
            end
            @test all(mod.(c45, ZZ(2)^18) == c18 for (c45, c18) in zip(E(45), E(18)))
        end
        # K-theory table of Z/27 ([AKN, Appendix A]; K_1(Z/27) = Z/18)
        tab = k_theory_table(zpn(3, 3), 1:4; integral=true)
        @test [r.degree for r in tab] == collect(0:7)
        @test tab[2].exponents == [2] && tab[2].prime_to_p_order == 2
        @test tab[4].exponents == [1, 3] && tab[4].prime_to_p_order == 8
        @test tab[5].exponents == [1]
    end

    if DISTRIBUTED_TESTS
        @testset "Distributed" begin
            using Distributed
            ids = add_syntomic_workers(2)
            try
                X = zpn(3, 2)
                Rd = syntomic_sweep(X, 2:3, 1:4; distributed=true)
                Rs = syntomic_sweep(X, 2:3, 1:4; distributed=false)
                @test [(H.h1, H.h2) for H in Rd] == [(H.h1, H.h2) for H in Rs]
                RR = syntomic_sweep([zpn(2, 2), zpn(3, 2)], [2], 1:3; distributed=true)
                @test [H.h1 for H in RR[1]] == [[1], [3], [3]]
            finally
                rmprocs(ids)
            end
        end
    end
end
