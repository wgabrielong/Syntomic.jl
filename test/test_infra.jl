@testset "sweep and linear-algebra infrastructure" begin
    @testset "Phase 0 at increasing precisions" begin
        @test S.exponent_phase_precisions(2, 2, 40) == [88, 176, 352, 704, 1408, 1569]
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
end
