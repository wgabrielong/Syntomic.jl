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
        # K-theory table of Z/27 ([AKN24, Appendix A]; K_1(Z/27) = Z/18)
        tab = k_theory_table(zpn(3, 3), 1:4; integral=true)
        @test [r.degree for r in tab] == collect(0:7)
        @test tab[2].exponents == [2] && tab[2].prime_to_p_order == 2
        @test tab[4].exponents == [1, 3] && tab[4].prime_to_p_order == 8
        @test tab[5].exponents == [1]
    end

    @testset "rings with the same computing ring share data" begin
        K, a = qadic_field(3, 2; precision=10, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        Q3 = padic_field(3; precision=10)
        Q3z, y = polynomial_ring(Q3, :y; cached=false)
        Xs = [
            truncated_witt_vectors(3, 2, 2),
            zpn(2, 2),
            truncated_witt_vectors(3, 1, 2),
            finite_chain_ring(z^2 + 3 * z + 3, 3),
            truncated_witt_vectors(3, 3, 2),
            finite_chain_ring(3, S.default_htilde(3, 2), [[3, 0], [0, 3], [1, 0]], 3),
            finite_chain_ring(y^2 + 3 * y + 3, 3),
        ]
        ns, is = 1:3, 1:3
        # W_n(F_q) for q = 3, 9, 27 use Z/3^n; z^2 + 3z + 3 over W(F_9) descends to Z_3[z]
        @test S.sweep_groups(Xs, 3) == [[1, 3, 5], [2], [4, 7], [6]]
        @test S.sweep_groups(Xs, 3; descent=false) == [[k] for k in 1:7]
        RR = syntomic_sweep(Xs, ns, is)
        @test length(RR) == length(Xs)
        key(H) = (H.ring, H.i, H.h1, H.h2, H.N, H.prec0, H.prec1, H.policy)
        for (X, R) in zip(Xs, RR)
            Rs = syntomic_sweep(X, ns, is)
            @test map(key, R) == map(key, Rs)
            @test all(H.ring.f == X.f for H in R)
        end
        # individual computations
        for H in RR[5]
            Hind = syntomic_cohomology(S.with_power(Xs[5], H.ring.n), H.i)
            @test (H.h1, H.h2) == (Hind.h1, Hind.h2)
        end
        # one set of W-matrices for several residue degrees
        Xc = S.computing_ring(S.with_power(Xs[1], 2), true)
        rings = [S.with_power(Xs[k], 2) for k in (1, 3, 5)]
        N = S.working_precision(rings[3], 3; precision_policy=:exponent)
        cs = S._syntomic_complexes(rings, Xc, 3, N; policy=:fixed)
        for (X, (C, t)) in zip(rings, cs)
            C1 = syntomic_complex(X, 3; N=N)
            @test C.syn0.M == C1.syn0.M && C.syn1.M == C1.syn1.M
            @test haskey(t, :wmatrices) && haskey(t, :square)
        end
        @test_throws ArgumentError S._syntomic_complexes(
            [zpn(3, 3)], Xc, 3, N; policy=:fixed
        )
        @test isempty(syntomic_sweep(FiniteChainRing[], ns, is))
        @test syntomic_sweep(Xs[1:2], Int[], is) ==
            [SyntomicCohomology[], SyntomicCohomology[]]
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
                # one group (W_2(F_q), q = 3, 9): the pairs are distributed
                Ws = [truncated_witt_vectors(3, f, 2) for f in 1:2]
                Rd = syntomic_sweep(Ws, 2:3, 1:3; distributed=true)
                Rs = syntomic_sweep(Ws, 2:3, 1:3; distributed=false)
                @test [[(H.h1, H.h2) for H in R] for R in Rd] == [[(H.h1, H.h2) for H in R] for R in Rs]
                # two groups on two workers
                Rd = syntomic_sweep([Ws; [zpn(2, 2)]], 2:3, 1:3; distributed=true)
                @test [H.h1 for H in Rd[3]] == [H.h1 for H in syntomic_sweep(zpn(2, 2), 2:3, 1:3)]
                @test [H.h1 for H in Rd[2]] == [H.h1 for H in Rs[2]]
                # the memory check runs on each worker and on each host
                @test S._worker_memory_shares()[first(ids)] <= 1.0
                @test all(k >= 1 for (_, (_, k)) in S._host_memory())
                @test_throws ArgumentError syntomic_sweep(
                    zpn(2, 2), 2:2, 1:200; precision_policy=:apriori, distributed=true
                )
                @test_throws ArgumentError add_syntomic_workers(
                    1; threads_per_worker=Sys.CPU_THREADS + 1
                )
                ids2 = add_syntomic_workers(
                    1; threads_per_worker=Sys.CPU_THREADS + 1, oversubscribe=true
                )
                rmprocs(ids2)
            finally
                rmprocs(ids)
            end
        end
    end
end
