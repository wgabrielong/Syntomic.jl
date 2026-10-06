@testset "power series over W(F_q)" begin
    rng = Xoshiro(314159)
    for (p, f, N) in
        ((2, 1, 30), (3, 1, 20), (2, 2, 30), (3, 2, 20), (5, 3, 12), (2, 3, 25))
        @testset "p=$p f=$f N=$N" begin
            R, K, a = galois_ring_and_qadic(p, f, N)
            Z = 9
            for _ in 1:5
                za, zb = rand(rng, 0:2), rand(rng, 0:2)
                A = rand_wseries(rng, R, Z; zval=za)
                B = rand_wseries(rng, R, Z; zval=zb)
                cA = S.ws_coefficients(A)
                cB = S.ws_coefficients(B)

                # multiplication against the naive convolution
                AB = S.ws_mul(A, B)
                @test AB.zprec == min(Z + zb, Z + za)
                naive = naive_series_product(R, cA, cB, AB.zprec)
                @test S.ws_coefficients(AB) == naive

                # addition with different valuation offsets
                ApB = S.ws_add(A, B)
                @test S.ws_coefficients(ApB) == [S.w_add(R, cA[k], cB[k]) for k in 1:Z]
                @test S.ws_iszero(S.ws_sub(ApB, S.ws_add(B, A)))

                # Frobenius: φ(Σ a_k z^k) = Σ σ(a_k) z^{pk}
                phiA = S.ws_frobenius(A; zprec=Z)
                for k in 0:(Z - 1)
                    expected =
                        k % p == 0 ? S.w_frobenius(R, cA[div(k, p) + 1]) : S.w_zero(R)
                    @test S.ws_coefficient(phiA, k) == expected
                end
                @test S.ws_coefficients(S.ws_sigma(A)) == [S.w_frobenius(R, c) for c in cA]

                # δ is a δ-structure ([AKN24, §3.1, (5)–(7), p. 14]):
                #   δ(ab) = a^p δ(b) + b^p δ(a) + p δ(a) δ(b)
                #   δ(a+b) = δ(a) + δ(b) - Σ_{0<j<p} (C(p,j)/p) a^j b^{p-j}
                dA = S.ws_delta(A)
                dB = S.ws_delta(B)
                @test dA.prec == N - 1
                lhs = S.ws_delta(S.ws_mul(A, B; zprec=Z); zprec=Z)
                rhs = S.ws_add(
                    S.ws_add(
                        S.ws_mul(S.ws_pow(A, p; zprec=Z), dB; zprec=Z),
                        S.ws_mul(S.ws_pow(B, p; zprec=Z), dA; zprec=Z),
                    ),
                    S.ws_scale_int(S.ws_mul(dA, dB; zprec=Z), p),
                )
                @test S.ws_isequal(S.ws_truncate(lhs, Z), S.ws_truncate(rhs, Z))
                lhs = S.ws_delta(S.ws_add(A, B))
                cross = S.ws_zero(R, Z)
                for j in 1:(p - 1)
                    t = S.ws_mul(
                        S.ws_pow(A, j; zprec=Z), S.ws_pow(B, p - j; zprec=Z); zprec=Z
                    )
                    cross = S.ws_add(cross, S.ws_scale_int(t, div(binomial(p, j), p)))
                end
                rhs = S.ws_sub(S.ws_add(dA, dB), cross)
                @test S.ws_isequal(S.ws_truncate(lhs, Z), S.ws_truncate(rhs, Z))

                # exact division by p
                pA = S.ws_scale_int(A, p^2)
                @test S.ws_isequal(S.ws_divexact_p(pA, 2), A; prec=N - 2)
            end

            # δ vanishes on Teichmüller lifts; δ(z) = 0
            t = qadic_to_welem(R, teichmuller(a))
            @test S.ws_iszero(S.ws_delta(S.ws_const(R, t, 5)))
            @test S.ws_iszero(S.ws_delta(S.ws_monomial(R, S.w_one(R), 1, 6)))

            # inverses of unit series
            U = S.ws_add(
                S.ws_const(R, rand_unit(rng, R), Z), rand_wseries(rng, R, Z; zval=1)
            )
            @test S.ws_isequal(S.ws_mul(U, S.ws_inv(U)), S.ws_one(R, Z))
            @test_throws DomainError S.ws_inv(S.ws_scale_int(U, p))

            # PrecisionExhausted on impossible divisions
            @test_throws PrecisionExhausted S.ws_divexact_p(U, 1)
            low = S.WSeries(R, U.c, U.zval, U.zprec, 0)
            @test_throws PrecisionExhausted S.ws_divexact_p(low, 1)

            # z-valuation and change of precision
            @test S.ws_z_valuation(S.ws_shift(U, 3)) == 3
            R2 = S.change_precision(R, N - 3)
            U2 = S.ws_change_ring(U, R2)
            @test U2.prec == N - 3
            @test all(
                S.w_isequal(R, S.ws_coefficient(U, k), S.ws_coefficient(U2, k); prec=N - 3)
                for k in 0:(Z - 1)
            )
        end
    end
end
