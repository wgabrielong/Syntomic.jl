@testset "Galois ring W(F_q)/p^N" begin
    rng = Xoshiro(20260925)
    for (p, f, N) in
        ((2, 1, 30), (7, 1, 12), (2, 2, 40), (3, 2, 25), (5, 3, 15), (2, 3, 30), (3, 4, 12))
        @testset "p=$p f=$f N=$N" begin
            R, K, a = galois_ring_and_qadic(p, f, N)
            @test R.f == f
            # σ has order f and lifts the p-th power map
            Sm = matrix(R.Zn, R.sigma)
            @test Sm^f == identity_matrix(R.Zn, f)
            x = S.w_gen(R)
            @test S.w_isequal(R, S.w_frobenius(R, x), S.w_pow(R, x, p); prec=1)
            for _ in 1:20
                u = rand_welem(rng, R)
                v = rand_welem(rng, R)
                qu = welem_to_qadic(K, a, u)
                qv = welem_to_qadic(K, a, v)
                @test S.w_mul(R, u, v) == qadic_to_welem(R, qu * qv)
                @test S.w_add(R, u, v) == qadic_to_welem(R, qu + qv)
                @test S.w_frobenius(R, u) == qadic_to_welem(R, frobenius(qu))
                # multiplication matrices: M(uv) = M(u)M(v) and S·M(u) = M(σu)·S
                Mu = S.mult_matrix(R, u)
                Mv = S.mult_matrix(R, v)
                @test matrix(R.Zn, Mu) * matrix(R.Zn, Mv) ==
                    matrix(R.Zn, S.mult_matrix(R, S.w_mul(R, u, v)))
                @test Sm * matrix(R.Zn, Mu) ==
                    matrix(R.Zn, S.mult_matrix(R, S.w_frobenius(R, u))) * Sm
                w = rand_unit(rng, R)
                @test S.w_mul(R, w, S.w_inv(R, w)) == S.w_one(R)
                @test S.w_inv(R, w) == qadic_to_welem(R, inv(welem_to_qadic(K, a, w)))
            end
            @test_throws DomainError S.w_inv(R, S.w_int(R, p))
            # valuations and exact division by powers of p
            u = rand_unit(rng, R)
            pu = S.w_mul_int(R, u, ZZ(p)^3)
            @test S.w_valuation(R, pu) == 3
            @test S.w_valuation(R, S.w_zero(R)) == N
            @test S.w_isequal(R, S.w_divexact_p(R, pu, 3), u; prec=N - 3)
            @test_throws PrecisionExhausted S.w_divexact_p(R, u, 1)
            # change of precision keeps h̃ and σ
            R2 = S.change_precision(R, N + 5)
            @test all(mod(R2.sigma[i, j] - R.sigma[i, j], R.pN) == 0 for i in 1:f, j in 1:f)
        end
    end

    @testset "non-Conway presentation" begin
        # F_9 presented by x^2 + x + 2 (irreducible mod 3), not the Conway polynomial x^2 + 2x + 2.
        R = S.GaloisRing(3, [2, 1, 1], 20)
        x = S.w_gen(R)
        sx = S.w_frobenius(R, x)
        # σ(x) is a root of h̃ congruent to x^3 mod 3
        hx = S._evaluate_integer_polynomial(R, R.htilde, sx)
        @test S.w_iszero(R, hx)
        @test S.w_isequal(R, sx, S.w_pow(R, x, 3); prec=1)
        @test_throws ArgumentError S.GaloisRing(3, [2, 0, 1], 10)   # x^2 + 2 = (x+1)(x+2) mod 3
        @test_throws ArgumentError S.GaloisRing(3, [2, 0, 2], 10)   # not monic
    end
end
