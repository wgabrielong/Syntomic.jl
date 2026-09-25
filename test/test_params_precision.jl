@testset "parameters and precision" begin
    @testset "ilog and digits" begin
        @test S.ilog(3, 243) == 5
        @test S.ilog(3, 242) == 4
        @test S.ilog(2, 1) == 0
        @test S.ilog(7, big(7)^20) == 20
        @test S.ilog(7, big(7)^20 - 1) == 19
        for p in (2, 3, 5, 11), x in 1:500
            @test S.ilog(p, x) == Nemo.flog(ZZ(x), ZZ(p))
        end
        @test S.base_p_digits(0, 3, 2) == [0, 0]
        @test S.base_p_digits(11, 3, 3) == [2, 0, 1]
        @test_throws ArgumentError S.base_p_digits(9, 3, 2)
    end

    @testset "SyntomicParams" begin
        prm = S.SyntomicParams(7, 1, 2, 3)
        @test (prm.L, prm.nf, prm.ng) == (6, 1, 1)
        @test S.SyntomicParams(2, 1, 3, 8).nf == 3           # M_f = ⌊log_2 7⌋ = 2
        @test S.SyntomicParams(2, 1, 3, 8).ng == 5           # M_g = ⌊log_2 23⌋ = 4
        @test S.SyntomicParams(3, 2, 2, 1).nf == 0           # no f-variables for i = 1
        @test S.SyntomicParams(3, 1, 1, 1).ng == 0           # L = 1
        @test_throws ArgumentError S.SyntomicParams(4, 1, 2, 2)
        @test_throws ArgumentError S.SyntomicParams(3, 0, 2, 2)
        @test_throws ArgumentError S.SyntomicParams(3, 1, 0, 2)
        @test_throws ArgumentError S.SyntomicParams(3, 1, 2, 0)
        # one reduced basis monomial per F-weight
        prm = S.SyntomicParams(2, 1, 3, 8)
        @test [S.f_weight(prm, S.f_basis_exponents(prm, w)...) for w in 0:(prm.L - 1)] == collect(0:(prm.L - 1))
        @test [S.g_weight(prm.p, S.g_basis_exponents(prm, w)) for w in 0:(prm.L - 1)] == collect(0:(prm.L - 1))
        @test S.f_basis_exponents(S.SyntomicParams(7, 1, 2, 3), 5) == (1, [2])
    end

    @testset "target precision and losses" begin
        for p in (2, 3, 5, 7), n in 1:5, i in 1:6
            L = n * i
            @test sum((S.epsilon(i, j, n) for j in 1:(L - 1)); init=0) == i * (n - 1)
            @test S.det_valuation_can1(n, i) == S.loss_nygaard(n, i)
            @test S.det_valuation_red(p, n, i) == S.loss_quotient(p, n, i)
            @test S.det_valuation_nabla_N(p, n, i) == S.target_precision(p, n, i)
            # Lemma 4.62: v(det ∇_R) = v(det red∇) + v(det ∇_OK) - v(det red) ≥ 0 bookkeeping
            @test S.h2_exponent_bound(p, n, i) <= S.target_precision(p, n, i)
        end
        @test S.target_precision(7, 2, 3) == 3
        @test S.apriori_working_precision(7, 2, 3) == 14
        @test S.loss_quotient(3, 2, 3) == 0          # no loss up to weight p (Remark 7.7)
        @test S.loss_quotient(2, 3, 4) == 3 * (1 + 1)  # s = 2, 3: v_2(2!) = 1, v_2(3!) = 1
        @test S.loss_delta(2, 3, 8) == 5
        @test S.initial_adaptive_guess(2, 3, 8) >= 1
    end
end
