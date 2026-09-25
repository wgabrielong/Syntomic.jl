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
    # exceptions keep their type whether or not threads are used
    boom(x) = x == 2 ? throw(PrecisionExhausted("boom")) : x
    for threaded in (true, false)
        @test_throws PrecisionExhausted S.tmap(boom, 1:4; threaded=threaded)
        @test_throws PrecisionExhausted S.tforeach(boom, 1:4; threaded=threaded)
        @test_throws PrecisionExhausted fetch(S.spawn_if(threaded, () -> boom(2)))
    end
    @test S.tmap(x -> x^2, (1, 2, 3); threaded=true) == [1, 4, 9]
end
