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
        @test_throws PrecisionExhausted syntomic_cohomology(zpn(2, 3), 2; N=1)
        @test_throws ArgumentError syntomic_cohomology(
            zpn(2, 3), 2; precision_policy=:other
        )
        # the recorded policy
        @test syntomic_cohomology(zpn(2, 3), 3).policy == :adaptive
        @test syntomic_cohomology(zpn(2, 3), 3; precision_policy=:apriori).policy ==
            :apriori
        @test syntomic_cohomology(zpn(2, 3), 3; N=60).policy == :fixed
    end

    @testset "the final check" begin
        # replacing syn^1 by a matrix without elementary divisors > 1 violates
        # Σ h1 - Σ h2 = f·i·(n-1) ([AKN, Prop. 2.18])
        C = syntomic_complex(zpn(2, 2), 2)
        m = size(C.syn1.M, 1)
        Zn = base_ring(C.syn1.M)
        one_block = hcat(identity_matrix(Zn, m), zero_matrix(Zn, m, m))
        Cbad = SyntomicComplex(
            C.ring,
            C.i,
            C.N,
            C.syn0,
            S.PMatrix(one_block, C.syn1.prec),
            C.exponents,
            C.policy,
            C.intermediates,
        )
        @test_throws CertificateError syntomic_cohomology(Cbad)
        H = syntomic_cohomology(Cbad; check=false)
        @test (H.h1, H.h2) == ([3], Int[])
    end

    @testset "matrix invariants ([AKN, Lemmas 4.61–4.64])" begin
        # the valuations of the diagonal (f×f) blocks of a Z_p-matrix in F-weight order
        function diagonal_valuations(A::S.PMatrix, p, f)
            M = lift(A.M)
            return map(1:div(ncols(M), f)) do j
                r = (f * (j - 1) + 1):(f * j)
                vals, nzero = local_elementary_divisors(M[r, r], p, A.prec)
                (nzero == 0 && allequal(vals)) ? vals[1] : -1
            end
        end
        K, a = qadic_field(2, 2; precision=30, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        for (X, i) in (
            (zpn(2, 3), 3),
            (zpn(5, 2), 6),
            (truncated_witt_vectors(3, 2, 2), 3),              # descent to f = 1
            (finite_chain_ring(z^2 + 2a * z + 2a, 3), 2),       # W(F_4)-arithmetic
        )
            C = syntomic_complex(X, i)
            p, f, n, L = X.p, X.f, X.n, X.n * i
            I = C.intermediates
            vp(j) = S.padic_valuation(p, j)
            @test diagonal_valuations(I[:nablaOK], p, f) == [vp(j) for j in 1:(L - 1)]
            @test diagonal_valuations(I[:nablaR], p, f) ==
                [vp(S.braces(j, n)) for j in 1:(L - 1)]
            vN = diagonal_valuations(I[:nablaN], p, f)
            @test vN == [S.epsilon(i, j, n) + vp(S.braces(j, n)) for j in 1:(L - 1)]
            @test sum(vN) == S.det_valuation_nabla_N(p, n, i)
            @test diagonal_valuations(I[:can0], p, f) == [i - div(j, n) for j in 1:(L - 1)]
            @test diagonal_valuations(I[:can1], p, f) ==
                [i - 1 - div(m, n) for m in 0:(L - 2)]
            @test diagonal_valuations(I[:red], p, f) ==
                [vp(factorial(big(div(j, n)))) for j in 1:(L - 1)]
            # syn^1 ∘ syn^0 = 0 (a bookkeeping check of Steps 7–10)
            P = min(C.syn0.prec, C.syn1.prec)
            D = lift(C.syn1.M * C.syn0.M)
            @test all(iszero(mod(x, ZZ(p)^P)) for x in D)
        end
    end

    @testset "σ matters for f > 1" begin
        # The pipeline over a Galois ring whose Frobenius is replaced by the identity must not
        # reproduce the brute-force K_1 of O_L/ϖ^5, L = Q_4(ϖ), ϖ² + 2aϖ + 2a = 0.
        K, a = qadic_field(2, 2; precision=30, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        X = finite_chain_ring(z^2 + 2a * z + 2a, 5)
        function cohomology_over(R, X, i)
            prm = S.SyntomicParams(X.p, X.f, X.n, i)
            ed = S.EisensteinData(R, X, prm.L)
            lam = S.LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)
            W = (;
                prm,
                R,
                S._f_side(prm, R, ed, lam, i)...,
                nablaOK=S._g_side(prm, R, ed, lam, i),
            )
            syn0, syn1, exps, inter = S.syntomic_square(W, R)
            C = SyntomicComplex(X, i, R.N, syn0, syn1, exps, :fixed, inter)
            H = syntomic_cohomology(C; check=false)
            return (H.h1, H.h2)
        end
        N = S.apriori_working_precision(2, 5, 1)
        R = S.GaloisRing(X.p, X.htilde, N)
        @test cohomology_over(R, X, 1) == ([1, 1, 1, 2, 3], Int[])
        Rid = S.GaloisRing(X.p, X.htilde, N)
        Rid.sigma .= [d == e ? one(ZZ) : zero(ZZ) for d in 1:2, e in 1:2]
        wrong = try
            cohomology_over(Rid, X, 1) != ([1, 1, 1, 2, 3], Int[])
        catch err
            err isa PrecisionExhausted
        end
        @test wrong
    end

    @testset "W(F_9) presented by x² + 1 instead of the Conway polynomial" begin
        # x = ζ^2 for a primitive 8th root of unity ζ (the Teichmüller lift of the Conway
        # generator), up to the Frobenius: E = x·z + 3 and E = z² + 3x·z + 3 over Z_3[x]/(x² + 1)
        # present the same rings as ζ²·z + 3 and z² + 3ζ²·z + 3.
        h = [1, 0, 1]
        for (E, Et, n, i) in (
            ([[3, 0], [0, 1]], [[3], [0, 0, 1]], 3, 2),
            ([[3, 0], [0, 3], [1, 0]], [[3], [0, 0, 3], [1]], 3, 2),
            ([[3, 0], [0, 3], [1, 0]], [[3], [0, 0, 3], [1]], 2, 3),
        )
            X = finite_chain_ring(3, h, E, n)
            @test S.descent_ring(X) === nothing                 # genuine W(F_9)-arithmetic
            Xt = finite_chain_ring(Et; p=3, f=2, n=n, basis=:teichmuller)
            @test coh(X, i) == coh(Xt, i)
        end
        @test coh(finite_chain_ring(3, h, [[3, 0], [0, 1]], 3), 2) ==
            coh(truncated_witt_vectors(3, 2, 3), 2)
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
