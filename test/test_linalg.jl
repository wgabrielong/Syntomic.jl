@testset "linear algebra over Z_p and W(F_q)" begin
    rng = Xoshiro(271828)

    @testset "local elementary divisors" begin
        for p in (2, 3, 5), _ in 1:15
            r, c = rand(rng, 1:7), rand(rng, 1:7)
            A = [ZZ(rand(rng, -40:40)) for _ in 1:r, _ in 1:c]
            # oracle: p-adic valuations of the elementary divisors over Z
            eds = elementary_divisors(matrix(ZZ, A))
            expected = sort([valuation(d, p) for d in eds if !iszero(d)])
            zero_count = count(iszero, eds) + (min(r, c) - length(eds))
            prec = 60
            vals, nzero = local_elementary_divisors(A, p, prec)
            @test vals == expected
            @test nzero == zero_count
        end
        # prescribed elementary divisors, hidden by unimodular transformations
        for p in (2, 3, 7)
            n = 6
            a = sort(rand(rng, 0:5, n))
            D = diagonal_matrix(ZZ, [ZZ(p)^k for k in a])
            U = identity_matrix(ZZ, n)
            V = identity_matrix(ZZ, n)
            for _ in 1:30
                i, j = rand(rng, 1:n, 2)
                i == j && continue
                U[i, :] = U[i, :] + rand(rng, -3:3) * U[j, :]
                V[:, j] = V[:, j] + rand(rng, -3:3) * V[:, i]
            end
            M = U * D * V
            @test local_elementary_divisors(M, p, 20)[1] == a
            @test S.local_exponent(M, p, 20) == maximum(a)
            @test local_elementary_divisors(M, p, 3) == (filter(<(3), a), count(>=(3), a))
            @test_throws PrecisionExhausted S.local_exponent(M, p, maximum(a))
        end
    end

    # X·diag(p^a)·Y for random X (r × n) and Y (n × c) with small entries
    function product_matrix(rng, p, a, r, c)
        n = length(a)
        X = matrix(ZZ, [ZZ(rand(rng, -9:9)) for _ in 1:r, _ in 1:n])
        Y = matrix(ZZ, [ZZ(rand(rng, -9:9)) for _ in 1:n, _ in 1:c])
        return X * diagonal_matrix(ZZ, [ZZ(p)^e for e in a]) * Y
    end

    @testset "oracle: Hecke.hnf_modular_eldiv" begin
        # a full-rank A (r × n, r ≥ n) has the same elementary divisors as its HNF, computed
        # modulo D = det(AᵀA), a multiple of the largest elementary divisor (by Cauchy–Binet,
        # every n × n minor is divisible by the product of the elementary divisors)
        for p in (2, 3, 5, 7, 1000003), _ in 1:8
            k = S.snf_word_precision(p)
            n = rand(rng, 1:8)
            r = n + rand(rng, 0:4)
            # mostly small valuations, sometimes beyond the word-size precision k
            a = [rand(rng, Bool) ? rand(rng, 0:3) : rand(rng, 0:(k + 3)) for _ in 1:n]
            A = product_matrix(rng, p, a, r, n)
            D = det(transpose(A) * A)
            iszero(D) && continue                           # not of full rank
            H = Hecke.hnf_modular_eldiv(A, abs(D))
            Hn = H[1:n, :]
            @test is_upper_triangular(Hn)
            @test iszero(H[(n + 1):r, :])
            expected = sort([valuation(d, p) for d in elementary_divisors(Hn)])
            top = maximum(expected)
            @test local_elementary_divisors(A, p, top + 1) == (expected, 0)
            @test local_elementary_divisors(Hn, p, top + 1) == (expected, 0)
            @test local_elementary_divisors(Matrix{ZZRingElem}(A), p, top + 5) ==
                (expected, 0)
            @test S.local_exponent(Hn, p, top + 1) == top
            # the truncated modular regime: over Z/p^prec, and for a ZZModMatrix modulo
            # p^cap with prec ≤ cap
            for prec in unique([1, 2, k, k + 1, top, top + 1, rand(rng, 1:(top + 1))])
                prec >= 1 || continue
                want = (filter(<(prec), expected), count(>=(prec), expected))
                @test local_elementary_divisors(A, p, prec) == want
                cap = prec + rand(rng, 0:70)
                Am = matrix(residue_ring(ZZ, ZZ(p)^cap)[1], A)
                @test local_elementary_divisors(Am, p, prec) == want
                @test local_elementary_divisors(Am, p, prec; threaded=true) == want
            end
        end
    end

    @testset "word-size and ZZRingElem paths agree" begin
        @test S.snf_word_precision(2) == 62
        @test S.snf_word_precision(3) == 39
        @test S.snf_word_precision(1000003) == 3
        @test S.snf_word_precision(4611686018427388039) == 0          # a prime above 2^62
        for _ in 1:20
            x = rand(rng, UInt64) | 1
            @test S.inv_mod_2_64(x) * x == 1
        end
        for p in (2, 3, 5, 101, 1000003), _ in 1:4
            k = S.snf_word_precision(p)
            r, c = rand(rng, 15:45), rand(rng, 15:45)
            prec = rand(rng, (1, 2, k, rand(rng, 1:k)))
            n = rand(rng, 1:min(r, c))                      # rank at most n
            A = Matrix{ZZRingElem}(product_matrix(rng, p, rand(rng, 0:(prec + 1), n), r, c))
            if rand(rng, Bool)                              # large entries
                A .+= ZZ(p)^prec .* [ZZ(rand(rng, (-2^40):(2^40)))^3 for _ in 1:r, _ in 1:c]
            end
            ref = S.snf_big(A, p, prec)
            @test local_elementary_divisors(A, p, prec) == ref
            @test local_elementary_divisors(matrix(ZZ, A), p, prec) == ref
            for threaded in (false, true), min_work in (0, 1 << 30)
                @test S.snf_word(A, p, prec; threaded=threaded, min_work=min_work) == ref
                @test S.snf_big(A, p, prec; threaded=threaded, min_work=min_work) == ref
            end
            # the same valuations beyond the word size, where the word-size attempt modulo
            # p^k decides the result only if no elementary divisor vanishes modulo p^k
            big = prec + k + rand(rng, 1:30)
            @test local_elementary_divisors(A, p, big) == S.snf_big(A, p, big)
            @test local_elementary_divisors(A, p, big; threaded=true) ==
                S.snf_big(A, p, big; threaded=true, min_work=0)
        end
        # far beyond the word size the precisions k, 4k, 16k, … ≤ prec/4 are tried first
        @test S.snf_precisions(2, 30) == [30]
        @test S.snf_precisions(2, 62) == [62]
        @test S.snf_precisions(2, 63) == [62, 63]
        @test S.snf_precisions(2, 1564) == [62, 248, 1564]
        @test S.snf_precisions(3, 600) == [39, 600]
        @test S.snf_precisions(1000003, 200) == [3, 12, 48, 200]
        @test S.snf_precisions(4611686018427388039, 20) == [4, 20]
        @test S.snf_precisions(4611686018427388039, 1) == [1]
        for (p, a, prec) in (
            (2, [0, 5, 70, 70], 1564),          # decided modulo 2^248
            (2, [0, 5, 300, 1600], 1564),       # an elementary divisor ≡ 0 mod 2^1564
            (1000003, [0, 2, 20, 30], 200),     # decided modulo p^48
            (1000003, [1, 60, 60], 200),        # decided modulo p^200
        )
            A = Matrix{ZZRingElem}(product_matrix(rng, p, a, 6, length(a)))
            ref = S.snf_big(A, p, prec)
            @test local_elementary_divisors(A, p, prec) == ref
            @test local_elementary_divisors(A, p, prec; threaded=true) == ref
            @test ref[1] == filter(<(prec), local_elementary_divisors(A, p, 2000)[1])
        end
        # a prime above 2^62: only the ZZRingElem path applies
        p = 4611686018427388039
        A = [ZZ(p)^2*5 ZZ(p); ZZ(p)+1 ZZ(7)]
        @test local_elementary_divisors(A, p, 3) == ([0, 1], 0)
        @test local_elementary_divisors(A, p, 1) == ([0], 1)
        # a step large enough to run in parallel with the default threshold
        m = ZZ(3)^30
        A = [rand(rng, ZZ(0):(m - 1)) for _ in 1:520, _ in 1:560]
        ref = S.snf_word(A, 3, 30)
        @test ref[2] == 0
        @test local_elementary_divisors(A, 3, 30; threaded=true) == ref
        @test local_elementary_divisors(A, 2, 30; threaded=true) ==
            S.snf_word(A, 2, 30; threaded=false)
        # errors and degenerate cases
        @test_throws ArgumentError local_elementary_divisors(A, 1, 5)
        @test_throws ArgumentError S.snf_word(A, 2, 63)
        @test local_elementary_divisors(A[1:3, 1:2], 3, 0) == (Int[], 2)
        @test local_elementary_divisors([ZZ(0) for _ in 1:3, _ in 1:4], 3, 100) == (Int[], 3)
        @test local_elementary_divisors(Matrix{ZZRingElem}(undef, 0, 4), 2, 10) ==
            (Int[], 0)
    end

    for (p, f, N) in ((3, 1, 25), (2, 2, 30), (5, 3, 15))
        @testset "W-matrices and triangular solves, p=$p f=$f" begin
            R, _, _ = galois_ring_and_qadic(p, f, N)
            n = 5
            # a lower-triangular W-matrix with diagonal entries p^v·(unit)
            A = [S.w_zero(R) for _ in 1:n, _ in 1:n]
            for i in 1:n, j in 1:i
                A[i, j] = if i == j
                    S.w_mul_int(R, rand_unit(rng, R), ZZ(p)^rand(rng, 0:2))
                else
                    rand_welem(rng, R)
                end
            end
            WA = S.WMatrix(R, A, N)
            PA = S.zp_expand(WA)
            e = S.local_exponent(PA.M, p, N)
            @test e <= f * 2 * n

            # expansion is multiplicative on W-linear maps
            B = [rand_welem(rng, R) for _ in 1:n, _ in 1:n]
            AB = [
                reduce(
                    (x, y) -> S.w_add(R, x, y), [S.w_mul(R, A[i, k], B[k, j]) for k in 1:n]
                ) for i in 1:n, j in 1:n
            ]
            PB = S.zp_expand(S.WMatrix(R, B, N))
            @test S.pm_mul(PA, PB).M == S.zp_expand(S.WMatrix(R, AB, N)).M
            # semilinear rule: [v ↦ Aσ(v)]·[B] = [v ↦ Aσ(B)σ(v)]
            sB = [S.w_frobenius(R, x) for x in B]
            AsB = [
                reduce(
                    (x, y) -> S.w_add(R, x, y), [S.w_mul(R, A[i, k], sB[k, j]) for k in 1:n]
                ) for i in 1:n, j in 1:n
            ]
            @test S.pm_mul(S.zp_expand(WA; semilinear=true), PB).M ==
                S.zp_expand(S.WMatrix(R, AsB, N); semilinear=true).M

            # left and right solves recover the true solution to precision N - e
            m = 3
            Ytrue = S.PMatrix(
                matrix(R.Zn, [rand(rng, 0:(R.pN - 1)) for _ in 1:(f * n), _ in 1:m]), N
            )
            X = S.pm_mul(PA, Ytrue)
            Y = S.solve_lower(PA, X, f, p)
            @test Y.prec == N - e
            @test S.pm_isequal(Y, Ytrue, p)
            Ztrue = S.PMatrix(
                matrix(R.Zn, [rand(rng, 0:(R.pN - 1)) for _ in 1:m, _ in 1:(f * n)]), N
            )
            X2 = S.pm_mul(Ztrue, PA)
            Z2 = S.solve_right_lower(X2, PA, f, p)
            @test S.pm_isequal(Z2, Ztrue, p)
            # a right-hand side outside the image is detected: force A[1,1] = p·(unit)
            A[1, 1] = S.w_mul_int(R, rand_unit(rng, R), p)
            PA = S.zp_expand(S.WMatrix(R, A, N))
            e1 = zero_matrix(R.Zn, f * n, 1)
            e1[1, 1] = one(R.Zn)
            @test_throws PrecisionExhausted S.solve_lower(PA, S.PMatrix(e1, N), f, p)
        end
    end

    @testset "descent expansion" begin
        p, f, N = 3, 2, 20
        R, _, _ = galois_ring_and_qadic(p, f, N)
        R1 = S.GaloisRing(p, N)
        n = 4
        A = [ZZ(rand(rng, 0:(R.pN - 1))) for _ in 1:n, _ in 1:n]
        WA = S.WMatrix(R, [S.w_int(R, A[i, j]) for i in 1:n, j in 1:n], N)
        PA1 = S.PMatrix(matrix(R1.Zn, A), N)
        @test S.zp_expand_descent(R, PA1).M == S.zp_expand(WA).M
        @test S.zp_expand_descent(R, PA1; semilinear=true).M ==
            S.zp_expand(WA; semilinear=true).M
    end
end

@testset "precision below N, threads and errors" begin
    rng = Xoshiro(99)
    p, N, P = 3, 20, 8
    R = S.GaloisRing(p, N)
    n = 6
    A = [
        if i == j
            S.w_mul_int(R, S.w_one(R), ZZ(3)^rand(rng, 0:2))
        else
            (i > j ? rand_welem(rng, R) : S.w_zero(R))
        end for i in 1:n, j in 1:n
    ]
    PA = S.zp_expand(S.WMatrix(R, A, N))
    Ytrue = S.PMatrix(matrix(R.Zn, [rand(rng, 0:(R.pN - 1)) for _ in 1:n, _ in 1:2]), N)
    X = S.pm_mul(PA, Ytrue)
    # X known only modulo p^P: add garbage above p^P
    garbage = matrix(R.Zn, [ZZ(p)^P * rand(rng, 0:1000) for _ in 1:n, _ in 1:2])
    Xg = S.PMatrix(X.M + garbage, P)
    Y = S.solve_lower(PA, Xg, 1, p)
    @test Y.prec == P - S.local_exponent(PA.M, p, N)
    @test S.pm_isequal(Y, Ytrue, p)
    # precision exhausted: a solution would have precision ≤ 0
    @test_throws PrecisionExhausted S.solve_lower(PA, S.PMatrix(X.M, 1), 1, p; exponent=2)
    # not block lower-triangular
    U = S.PMatrix(matrix(R.Zn, [1 1; 0 1]), N)
    @test_throws ArgumentError S.solve_lower(
        U, S.PMatrix(identity_matrix(R.Zn, 2), N), 1, p
    )
    # prec above the modulus precision is rejected
    @test_throws ArgumentError local_elementary_divisors(PA.M, p, N + 1)
    # threaded and serial Smith normal forms agree
    B = [ZZ(rand(rng, -50:50)) * ZZ(p)^rand(rng, 0:3) for _ in 1:60, _ in 1:40]
    @test local_elementary_divisors(B, p, 30; threaded=true) ==
        local_elementary_divisors(B, p, 30; threaded=false)
    @test S.tmap(x -> x^2, (1, 2, 3); threaded=true) == [1, 4, 9]
end

@testset "threading layer" begin
    # exceptions of the interface keep their type whether or not threads are used
    boom(x) = x == 2 ? throw(PrecisionExhausted("boom")) : x
    for threaded in (true, false)
        @test_throws PrecisionExhausted S.tmap(boom, 1:4; threaded=threaded)
        @test_throws PrecisionExhausted S.tforeach(boom, 1:4; threaded=threaded)
        @test_throws PrecisionExhausted S.concurrently(
            () -> boom(2), () -> 1; threaded=threaded
        )
        @test_throws PrecisionExhausted S.concurrently(
            () -> 1, () -> boom(2); threaded=threaded
        )
        # other exceptions keep the backtrace of the task, wrapped
        err = try
            S.tmap(x -> x == 3 ? error("bug") : x, 1:4; threaded=threaded)
        catch e
            e
        end
        @test S.unwrap_task_exception(err) isa ErrorException
    end
    @test S.concurrently(() -> 1, () -> 2) == (1, 2)
    # the failing item with the smallest index wins, as in a serial loop
    two(x) = x >= 2 ? throw(ArgumentError("item $x")) : x
    err = try
        S.tmap(two, 1:50; threaded=true)
    catch e
        e
    end
    @test err == ArgumentError("item 2")
    # if f and g both fail, the exception of f is rethrown (the serial order)
    err = try
        S.concurrently(() -> throw(ArgumentError("f")), () -> throw(ArgumentError("g")))
    catch e
        e
    end
    @test err == ArgumentError("f")
    # after a failure no new items start, and no task outlives the call (the items are
    # CPU-bound, like the computations of the package, so they do not yield)
    started = Threads.Atomic{Int}(0)
    finished = Threads.Atomic{Int}(0)
    function work(k)
        Threads.atomic_add!(started, 1)
        k == 1 && throw(PrecisionExhausted("first"))
        t0 = time()
        while time() - t0 < 0.05
            GC.safepoint()
        end
        Threads.atomic_add!(finished, 1)
        return k
    end
    @test_throws PrecisionExhausted S.tmap(work, 1:100; threaded=true)
    @test started[] == finished[] + 1
    @test started[] < 100
    done = Threads.Atomic{Bool}(false)
    slow() = (sleep(0.2); done[] = true; 1)
    @test_throws PrecisionExhausted S.concurrently(slow, () -> boom(2))
    @test done[]
    # exceptions from Distributed are unwrapped too
    remote = S.Distributed.RemoteException(
        2, CapturedException(PrecisionExhausted("remote"), backtrace())
    )
    @test S.unwrap_task_exception(remote) == PrecisionExhausted("remote")
    @test_throws PrecisionExhausted try
        throw(remote)
    catch e
        S.rethrow_unwrapped(e)
    end
end
