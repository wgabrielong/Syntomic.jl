@testset "sweep and linear-algebra infrastructure" begin
    rng = Xoshiro(314159)

    # A random block lower-triangular Z_p-matrix of size f·nb (diagonal blocks p^v times the
    # multiplication matrix of a unit), with the given valuations v of the diagonal blocks.
    function block_lower(R::S.GaloisRing, vs::Vector{Int})
        nb = length(vs)
        A = [S.w_zero(R) for _ in 1:nb, _ in 1:nb]
        for i in 1:nb, j in 1:i
            A[i, j] = if i == j
                S.w_mul_int(R, rand_unit(rng, R), ZZ(R.p)^vs[i])
            else
                rand_welem(rng, R)
            end
        end
        return S.zp_expand(S.WMatrix(R, A, R.N))
    end
    rand_pmatrix(R, r, c) =
        S.PMatrix(matrix(R.Zn, [rand(rng, 0:(R.pN - 1)) for _ in 1:r, _ in 1:c]), R.N)

    @testset "split_range" begin
        for (c, k) in ((10, 3), (7, 7), (5, 1), (0, 2), (64, 8))
            rs = S.split_range(c, k)
            @test length(rs) == k
            @test reduce(vcat, collect.(rs); init=Int[]) == collect(1:c)
            @test maximum(length, rs) - minimum(length, rs) <= 1
        end
        @test_throws ArgumentError S.split_range(3, 0)
    end

    @testset "parallel triangular solves" begin
        for (p, f, nb, N) in ((3, 1, 104, 40), (2, 2, 52, 60))
            R = S.GaloisRing(p, f == 1 ? [0, 1] : S.default_htilde(p, f), N)
            A = block_lower(R, [k % 8 == 0 ? rand(rng, 1:2) : 0 for k in 1:nb])
            n = f * nb
            @test S._solve_block_count(n, n, false) == 1
            nblocks = S.threads_available() ? min(div(n, 8), Threads.nthreads()) : 1
            @test S._solve_block_count(n, n, true) == nblocks
            @test S._solve_block_count(64, 64, true) == 1
            Ytrue = rand_pmatrix(R, n, n)
            X = S.pm_mul(A, Ytrue)
            Ys = S.solve_lower(A, X, f, p; threaded=false)
            Yt = S.solve_lower(A, X, f, p; threaded=true)
            @test Yt.M == Ys.M && Yt.prec == Ys.prec
            @test S.pm_isequal(Yt, Ytrue, p)
            X2 = S.pm_mul(Ytrue, A)
            Zs = S.solve_right_lower(X2, A, f, p; threaded=false)
            Zt = S.solve_right_lower(X2, A, f, p; threaded=true)
            @test Zt.M == Zs.M && Zt.prec == Zs.prec
            @test S.pm_isequal(Zt, Ytrue, p)
        end
        # the exception is the one of a serial solve: the right-hand sides of the first
        # columns fail at a late diagonal block, the others at an early one (with a different
        # power of p)
        p, N, nb = 3, 30, 104
        R = S.GaloisRing(p, N)
        vs = zeros(Int, nb)
        vs[5] = 1
        vs[100] = 2
        A = block_lower(R, vs)
        Y = rand_pmatrix(R, nb, nb)
        X = S.pm_mul(A, Y).M
        for j in 1:nb
            k = j <= div(nb, 2) ? 100 : 5        # break divisibility at diagonal block k
            X[k, j] += one(R.Zn)
        end
        Xp = S.PMatrix(X, N)
        errs = map((false, true)) do threaded
            try
                S.solve_lower(A, Xp, 1, p; threaded=threaded)
                nothing
            catch err
                err
            end
        end
        @test errs[1] isa PrecisionExhausted
        @test errs[2] isa PrecisionExhausted
        @test sprint(showerror, errs[1]) == sprint(showerror, errs[2])
        @test occursin("$p^1", sprint(showerror, errs[2]))
    end

    @testset "parallel expansions and square" begin
        for (p, f) in ((2, 3), (3, 2))
            R = S.GaloisRing(p, S.default_htilde(p, f), 20)
            W = S.WMatrix(R, [rand_welem(rng, R) for _ in 1:24, _ in 1:23], 18)
            for semilinear in (false, true)
                a = S.zp_expand(W; semilinear=semilinear, threaded=false)
                b = S.zp_expand(W; semilinear=semilinear, threaded=true)
                @test a.M == b.M && a.prec == b.prec == 18
            end
            R1 = S.GaloisRing(p, 20)
            P = S.PMatrix(
                matrix(R1.Zn, [rand(rng, 0:(R1.pN - 1)) for _ in 1:70, _ in 1:66]), 19
            )
            for semilinear in (false, true)
                a = S.zp_expand_descent(R, P; semilinear=semilinear, threaded=false)
                b = S.zp_expand_descent(R, P; semilinear=semilinear, threaded=true)
                @test a.M == b.M && a.prec == b.prec == 19
                # the descent expansion is the expansion of the W-matrix with entries in Z_p
                WP = S.WMatrix(
                    R, [S.w_int(R, lift(P.M[i, j])) for i in 1:70, j in 1:66], 19
                )
                @test a.M == S.zp_expand(WP; semilinear=semilinear).M
            end
        end
        # Steps 7–10: threaded and serial squares are bit-identical
        for (X, i) in ((zpn(2, 3), 6), (truncated_witt_vectors(3, 2, 2), 3))
            Xc = S.computing_ring(X, true)
            N = S.working_precision(X, i; precision_policy=:exponent)
            W = S.syntomic_wmatrices(Xc, i, N)
            Rf = S.GaloisRing(X.p, X.htilde, N)
            a = S.syntomic_square(W, Rf; descent=Xc.f != X.f, threaded=false)
            b = S.syntomic_square(W, Rf; descent=Xc.f != X.f, threaded=true)
            @test a[1].M == b[1].M && a[2].M == b[2].M
            @test (a[1].prec, a[2].prec, a[3]) == (b[1].prec, b[2].prec, b[3])
            @test all(a[4][k].M == b[4][k].M for k in keys(a[4]))
        end
    end

    @testset "memory model" begin
        @test S.coefficient_bytes(2, 62) == 8
        @test S.coefficient_bytes(2, 63) == 56
        @test S.coefficient_bytes(3, 100) == 48 + 8 * 3
        # the f-envelope table: the normal forms memoized by the f-envelope
        for (p, n, i) in ((2, 2, 7), (3, 2, 5), (2, 3, 4), (5, 1, 3), (2, 2, 1))
            X = zpn(p, n)
            prm = S.SyntomicParams(p, 1, n, i)
            R = S.GaloisRing(p, 30)
            ed = S.EisensteinData(R, X, prm.L)
            lam = S.LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)
            fenv = S.FEnvelope(prm, R, ed, lam)
            S.reduction_matrix(fenv; threaded=false)
            S.reduction_matrix_nabla(fenv; threaded=false)
            S.can_matrix(fenv, i; threaded=false)
            S.can_nabla_matrix(fenv, i; threaded=false)
            S.phi_matrix(fenv; threaded=false)
            entries, coefficients = S.f_table_entries(p, n, i)
            @test length(fenv.cache) <= entries
            @test coefficients <= entries * prm.L
        end
        @test S.f_table_entries(2, 2, 20)[1] == 3734
        # the g-envelope table: non-reduced digit vectors with digits ≤ 2p - 2, and φ-values
        @test S.g_table_entries(2, 1) == 0
        @test S.g_table_entries(2, 4) == 1 + 2          # g_0^2, and φ(1), φ(g_0)
        @test S.g_table_entries(3, 10) ==
            count(
            e -> any(>=(3), e) && e[1] + 3 * e[2] + 9 * e[3] < 10,
            Iterators.product(0:4, 0:4, 0:4),
        ) + 4
        @test S.shared_memory_estimate(2, 1, 3, 1, 40) == 0
        @test S.pair_memory_estimate(2, 1, 1, 1, 1, 40) == 0
        @test S.pair_memory_estimate(2, 1, 3, 2, 10, 80) >
            S.pair_memory_estimate(2, 1, 1, 2, 10, 80)
        @test S.pair_memory_estimate(2, 1, 1, 2, 10^5, 10^6) == S.MEMORY_CAP
        @test S.estimated_working_precision(2, 3, 5, :apriori) ==
            S.apriori_working_precision(2, 3, 5)
        @test S.estimated_working_precision(2, 3, 5, :exponent) ==
            S.exponent_working_precision(2, 3, 5, 12, 12)
        # estimates of a sweep
        for policy in (:exponent, :apriori)
            est = S.sweep_memory_estimate(
                zpn(2, 3), 2, 2:3, 1:4; precision_policy=policy, concurrency=2
            )
            @test issorted(est.pairs; rev=true) && length(est.pairs) == 8
            # the exponents (Phase 0) are computed while the envelope of O_K is built
            pairs2 = S._bytes(S.MEMORY_CONCURRENT * (est.pairs[1] + est.pairs[2]))
            @test est.total == max(est.phase0 + est.shared, est.nablas + pairs2)
            @test est.shared >= est.nablas > 0
            @test (est.phase0 > 0) == (policy == :exponent)
            @test S.check_sweep_memory(est) === nothing
            err = try
                S.check_sweep_memory(est; share=1.0e-15)
            catch e
                e
            end
            @test err isa ArgumentError && occursin("check_memory=false", err.msg)
        end
        @test S.sweep_memory_estimate(zpn(2, 2), 1, Int[], 1:3).total == 0
        # a sweep far too large for any machine is refused before any computation
        @test_throws ArgumentError syntomic_sweep(zpn(2, 2), 2:2, 1:200)
        @test_throws ArgumentError syntomic_sweep(
            [zpn(2, 2), zpn(3, 2)], 2:2, 1:200; precision_policy=:apriori
        )
    end

    @testset "Phase 0 at increasing precisions" begin
        @test S.exponent_phase_precisions(2, 2, 40) == [54, 108, 216, 432, 864, 1569]
        @test S.exponent_phase_precisions(2, 2, 1) == [3]
        for (X, i) in ((zpn(2, 2), 9), (zpn(3, 3), 6), (truncated_witt_vectors(2, 2, 3), 4))
            Xc = S.computing_ring(X, true)
            prm = S.SyntomicParams(Xc.p, Xc.f, Xc.n, i)
            Ps = S.exponent_phase_precisions(Xc.p, Xc.n, i)
            e = S.exponent_phase(Xc, i)
            # the exponents certified at the largest precision (which always suffices)
            @test e == S._exponents_at_precision(Xc, prm, i, last(Ps))
            # too small a precision is detected, not answered
            @test_throws PrecisionExhausted S._exponents_at_precision(
                Xc, prm, i, max(e...) + 1
            )
        end
    end

    @testset "add_syntomic_workers refuses to oversubscribe" begin
        np = S.Distributed.nprocs()
        @test_throws ArgumentError add_syntomic_workers(Sys.CPU_THREADS + 1)
        @test_throws ArgumentError add_syntomic_workers(
            2; threads_per_worker=Sys.CPU_THREADS
        )
        @test_throws ArgumentError add_syntomic_workers(0)
        @test_throws ArgumentError add_syntomic_workers(1; threads_per_worker=0)
        @test S.Distributed.nprocs() == np
    end
end

@testset "available memory" begin
    @test S.available_memory() >= Int(Sys.free_memory()) - 2^26     # pages change meanwhile
    @test S.available_memory() <= Sys.total_memory()
    @test S.check_worker_memory(NamedTuple[]) === nothing
end

@testset "estimated exponents and overlapping" begin
    rings = (
        (zpn(2, 3), 12),
        (zpn(3, 3), 10),
        (zpn(7, 3), 10),
        (finite_chain_ring([2, 2, 1]; p=2, f=1, n=5), 8),
        (finite_chain_ring([[0, 2], [0, 2], [1]]; p=2, f=2, n=4), 6),    # no descent
    )
    # the lower bounds hold; the estimated precision (with its margin) is not too small, also
    # where the formulas alone are (Z/4 at i = 18, 34; n ≤ e; n = 1)
    for (X, i) in (
        rings...,
        (zpn(2, 2), 18),
        (zpn(2, 2), 34),
        (zpn(2, 1), 12),
        (finite_chain_ring([2, 0, 0, 1]; p=2, f=1, n=2), 18),
        (finite_chain_ring([3, 0, 1]; p=3, f=1, n=2), 16),
    )
        Xc = S.computing_ring(X, true)
        er, ec = S.exponent_phase(Xc, i)
        lr, lc = S.exponent_lower_bounds(X.p, X.n, i)
        @test er >= lr && ec >= lc
        hr, hc = S.predicted_exponents(X.p, X.n, Xc.e, i)
        @test hr >= lr && hc >= lc
        @test S.predicted_exponent_working_precision(X.p, X.n, Xc.e, i) >=
            S.exponent_working_precision(X.p, X.n, i, er, ec)
    end
    # an adaptive attempt just above the single-word precision is lowered to it
    Nw = S.one_word_precision(3)
    @test big(3)^Nw < big(2)^62 <= big(3)^(Nw + 1)
    for i in 2:40
        N = S.predicted_adaptive_working_precision(
            3, 2, 1, i, S.initial_adaptive_guess(3, 2, i)
        )
        @test !(Nw < N <= Nw + 4)
    end
    # the precision :exponent with the envelope of O_K at the estimate, at a too small estimate
    # (recomputed) and without threads: the same complexes
    for (X, i) in rings
        Xc = S.computing_ring(X, true)
        C1, _ = only(S._exponent_complexes([X], Xc, i; threaded=true))
        C2, _ = only(S._exponent_complexes([X], Xc, i; threaded=false))
        C3, _ = only(S._exponent_complexes([X], Xc, i; threaded=true, Nspec=5))
        C4, _ = only(S._exponent_complexes([X], Xc, i; threaded=false, Nspec=5))
        @test C1.syn0.M == C2.syn0.M && C1.syn1.M == C2.syn1.M
        @test (C1.syn0.prec, C1.syn1.prec) == (C2.syn0.prec, C2.syn1.prec)
        @test C3.syn0.M == C4.syn0.M && C3.syn1.M == C4.syn1.M
        # at the exact precision, without the overlap: the same cohomology
        e = S.exponent_phase(Xc, i)
        N = S.exponent_working_precision(X.p, X.n, i, e...)
        @test C1.N == C3.N == N
        H = syntomic_cohomology(S._syntomic_complex(X, Xc, i, N; policy=:exponent))
        for C in (C1, C3)
            Hc = syntomic_cohomology(C)
            @test (Hc.h1, Hc.h2) == (H.h1, H.h2)
        end
    end
    # :adaptive: odd primes are certified at the first attempt; a small first guess reaches the
    # last attempt (at the :exponent precision) with the same result
    for (X, i) in ((zpn(3, 4), 8), (zpn(5, 3), 12), (zpn(7, 3), 10))
        H = syntomic_cohomology(X, i)
        @test H.restarts == 0
        He = syntomic_cohomology(X, i; precision_policy=:exponent)
        @test (H.h1, H.h2) == (He.h1, He.h2)
        Xc = S.computing_ring(X, true)
        for threaded in (true, false)
            Hs = S._adaptive_cohomology(X, Xc, i; Tguess=1, threaded=threaded)
            @test (Hs.h1, Hs.h2) == (H.h1, H.h2) && Hs.restarts >= 1
        end
    end
    # a failing speculative build is ignored when its result is not used (as without threads)
    X = finite_chain_ring([[0, 2], [0, 2], [1]]; p=2, f=2, n=2)
    Xc = S.computing_ring(X, true)
    C1, _ = only(S._exponent_complexes([X], Xc, 9; threaded=true, Nspec=1))
    C2, _ = only(S._exponent_complexes([X], Xc, 9; threaded=false, Nspec=1))
    @test C1.syn0.M == C2.syn0.M && C1.syn1.M == C2.syn1.M
    @test length(S._sweep_group([zpn(2, 2)], 1:2, 1:12; threaded=true, Nspec=3)[1]) == 24
    # the exported total time counts the overlapped stages once
    t = @elapsed H = syntomic_cohomology(zpn(2, 2), 20; precision_policy=:exponent)
    @test S._total_time(H) <= t + 0.05
    # sweeps: a too small estimate of the shared precision is detected; the same results
    X = zpn(3, 2)
    ref = S._sweep_group([X], 2:3, 1:6; threaded=true)
    for threaded in (true, false), Nspec in (nothing, 3)
        R = S._sweep_group([X], 2:3, 1:6; threaded=threaded, Nspec=Nspec)
        @test [(H.h1, H.h2, H.N, H.prec0, H.prec1) for H in only(R)] == [(H.h1, H.h2, H.N, H.prec0, H.prec1) for H in only(ref)]
    end
end
