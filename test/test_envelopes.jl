@testset "prismatic envelopes" begin
    # Build the envelopes for a ring and weight at the a priori precision.
    function envelopes(X, i)
        prm = S.SyntomicParams(X.p, X.f, X.n, i)
        N = S.apriori_working_precision(X.p, X.n, i)
        R = S.GaloisRing(X.p, X.htilde, N)
        ed = S.EisensteinData(R, X, prm.L)
        lam = S.LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)
        return prm, R, ed, lam, S.FEnvelope(prm, R, ed, lam), S.GEnvelope(prm, R, ed, lam)
    end
    wval(R, a) = S.w_valuation(R, a)

    @testset "Eisenstein data and λ" begin
        K, a = qadic_field(2, 2; precision=40, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        X = finite_chain_ring(z^2 + 2a * z + 2a, 4)
        prm, R, ed, lam, _, _ = envelopes(X, 3)
        @test S.ws_coefficient(ed.E, 0) == S.w_int(R, 2)                 # E(0) = p
        # δ(E)(0) = 1 - p^{p-1}
        @test S.w_isequal(
            R, S.ws_coefficient(ed.deltaE, 0), S.w_int(R, 1 - 2); prec=R.N - 1
        )
        @test S.descent_ring(X) === nothing
        @test ed.phiE.zprec >= prm.L
        # λ_0 δ(E) = -1 and λ_{u+1} D_u = λ_u^p ([AKN24, Lemma 3.24])
        @test S.ws_isequal(
            S.ws_mul(lam.lambda[1], ed.deltaE), S.ws_const(R, S.w_int(R, -1), prm.L)
        )
        for u in 1:(length(lam.lambda) - 1)
            lhs = S.ws_mul(lam.lambda[u + 1], lam.D[u])
            rhs = S.ws_pow(lam.lambda[u], 2; zprec=prm.L)
            @test S.ws_isequal(lhs, rhs)
        end
    end

    @testset "f-envelope (R = O_K/ϖ^n)" begin
        # can0 for Z/49, i = 3 ([AKN24, Lemma 4.63]: diagonal p^{i-⌊j/n⌋})
        prm, R, ed, lam, fenv, _ = envelopes(zpn(7, 2), 3)
        can0 = [c[1] for c in S.can_matrix(fenv, 3).A]
        @test can0 == [343 0 0 0 0; 147 49 0 0 0; 21 14 49 0 0; 1 1 14 7 0; 0 0 1 1 7]
        for (X, i) in ((zpn(2, 3), 8), (zpn(3, 2), 7), (truncated_witt_vectors(2, 2, 3), 4))
            prm, R, ed, lam, fenv, _ = envelopes(X, i)
            p, n, L = prm.p, prm.n, prm.L
            can0 = S.can_matrix(fenv, i).A
            can1 = S.can_nabla_matrix(fenv, i).A
            red = S.reduction_matrix(fenv).A
            Phi = S.phi_matrix(fenv).A
            for j in 1:(L - 1), k in (j + 1):(L - 1)
                @test all(iszero, can0[j, k]) && all(iszero, red[j, k])
            end
            @test [wval(R, can0[j, j]) for j in 1:(L - 1)] == [i - div(j, n) for j in 1:(L - 1)]
            @test [wval(R, can1[m + 1, m + 1]) for m in 0:(L - 2)] == [i - 1 - div(m, n) for m in 0:(L - 2)]
            @test [wval(R, red[j, j]) for j in 1:(L - 1)] == [S.padic_valuation(p, factorial(big(div(j, n)))) for j in 1:(L - 1)]
            # φ multiplies F-weights by p: column w is supported in rows ≥ p·w
            @test all(
                all(iszero, Phi[r, w]) for w in 1:(L - 1), r in 1:(L - 1) if r < p * w
            )
        end
    end

    @testset "f-envelope: normal forms against the recursion over monomials" begin
        # The reduce subalgorithm as a memoized recursion over monomials z_0^a ∏ f_u^{e_u}:
        # z_0^n → f_0 first, then the lowest f_u^p → c_u f_{u+1} ([AKN24, Lemma 3.24, p. 21]), with
        # the precision rules of the f-envelope. The level-wise normal forms must agree with it
        # modulo p^N, not only modulo the precision.
        function ref_nf(fenv, cache, a, e)
            prm, R = fenv.prm, fenv.R
            w = S.f_monomial_weight(prm, a, e)
            w >= prm.L && return (S.wpoly_zero(R), R.N)
            return get!(cache, (a, copy(e))) do
                if a >= prm.n
                    e2 = copy(e)
                    e2[1] += 1
                    return ref_nf(fenv, cache, a - prm.n, e2)
                end
                u = findfirst(>=(prm.p), e)
                u === nothing && return (S.f_basis_element(fenv, w).v, R.N)
                e2 = copy(e)
                e2[u] -= prm.p
                e2[u + 1] += 1
                return ref_times(fenv, cache, fenv.c[u], a, e2)
            end
        end
        function ref_times(fenv, cache, s, a, e)
            prm, R = fenv.prm, fenv.R
            needed = prm.L - S.f_monomial_weight(prm, a, e)
            needed <= 0 && return (S.wpoly_zero(R), R.N)
            out = S.wpoly_zero(R)
            prec = min(s.prec, fenv.prec)
            for k in (s.zval):(needed - 1)
                sk = S.ws_coefficient(s, k)
                all(iszero, sk) && continue
                v, pv = ref_nf(fenv, cache, a + k, e)
                prec = min(prec, pv)
                S.wpoly_addmul_scalar!(R, out, v, sk)
            end
            return (out, prec)
        end
        same(x::S.FElem, (v, prec)) = x.v == v && x.prec == prec

        K, a = qadic_field(2, 2; precision=40, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        rng = Xoshiro(1729)
        for (X, i) in (
            (zpn(2, 2), 9),
            (zpn(3, 2), 7),
            (zpn(2, 3), 6),
            (zpn(5, 1), 12),
            (finite_chain_ring(z^2 + 2a * z + 2a, 3), 5),
            (truncated_witt_vectors(2, 3, 2), 4),
        )
            prm, R, ed, lam, fenv, _ = envelopes(X, i)
            p, n, L, nf = prm.p, prm.n, prm.L, prm.nf
            cache = Dict{Tuple{Int,Vector{Int}},Any}()
            @test all(
                same(S.f_z_power(fenv, k), ref_nf(fenv, cache, k, zeros(Int, nf))) for
                k in 0:(L + 1)
            )
            Ep = S.f_E_powers(fenv, i)
            for w in 0:(L - 1)
                c, e = S.f_basis_exponents(prm, w)
                for j in (0, 1, i - div(w, n))
                    @test Ep[j + 1].c == S.ws_pow(ed.E, j; zprec=L).c
                    @test same(
                        S.f_can_image(fenv, w, j), ref_times(fenv, cache, Ep[j + 1], c, e)
                    )
                end
                phi = S.f_phi_image(fenv, w)
                if nf == 0 || e[end] > 0
                    @test same(phi, (S.wpoly_zero(R), R.N))
                else
                    shifted = [0; e[1:(end - 1)]]
                    lamw = S.f_phi_series(fenv, e)
                    @test same(phi, ref_times(fenv, cache, lamw, p * c, shifted))
                end
            end
            # arbitrary monomials of F-weight < L and series, with carries of every digit
            for _ in 1:40
                e, aa = zeros(Int, nf), 0
                while true
                    e = [rand(rng, 0:max(div(L - 1, n * p^u), 0)) for u in 0:(nf - 1)]
                    aa = rand(rng, 0:(L - 1))
                    S.f_monomial_weight(prm, aa, e) < L && break
                end
                @test same(
                    S.f_monomial_normal_form(fenv, aa, e), ref_nf(fenv, cache, aa, e)
                )
                s = rand_wseries(rng, R, L; zval=rand(rng, 0:2))
                @test same(
                    S.f_series_times_monomial(fenv, s, aa, e),
                    ref_times(fenv, cache, s, aa, e),
                )
            end
            # the matrices do not depend on threads (each on a new envelope)
            for mat in (
                (env, thr) -> S.reduction_matrix(env; threaded=thr),
                (env, thr) -> S.reduction_matrix_nabla(env; threaded=thr),
                (env, thr) -> S.can_matrix(env, i; threaded=thr),
                (env, thr) -> S.can_nabla_matrix(env, i; threaded=thr),
                (env, thr) -> S.phi_matrix(env; threaded=thr),
            )
                A = mat(envelopes(X, i)[5], true)
                B = mat(envelopes(X, i)[5], false)
                @test A.A == B.A && A.prec == B.prec
            end
        end
        # invalid input: negative exponents, and series not known modulo z_0^L
        prm, R, ed, lam, fenv, _ = envelopes(zpn(2, 2), 9)
        @test_throws ArgumentError S.f_z_power(fenv, -1)
        @test_throws ArgumentError S.f_monomial_normal_form(fenv, 0, [-1, 0, 0, 0])
        @test_throws ArgumentError S.f_monomial_normal_form(fenv, -1, [1, 0, 0, 0])
        ed2 = S.EisensteinData(R, zpn(2, 2), prm.L - 1)
        @test_throws ArgumentError S.FEnvelope(prm, R, ed2, S.LambdaData(ed2, prm.nf - 2))
    end

    @testset "g-envelope (O_K over W[[z_0, z_1]])" begin
        K, a = qadic_field(3, 2; precision=40, cached=false)
        Kz, z = polynomial_ring(K, :z; cached=false)
        for (X, i) in (
            (zpn(2, 3), 6),
            (zpn(3, 3), 4),
            (finite_chain_ring(z^2 + 3a * z + 3a, 3), 4),
            (zpn(5, 2), 7),
        )
            prm, R, ed, lam, _, genv = envelopes(X, i)
            p, L = prm.p, prm.L
            @test genv.ng >= 2
            # φ(g_0) = (z_0 + g_0)^p - z_0^p  (φ(z_1) = z_1^p with z_1 = z_0 + g_0)
            z0 = S.g_from_series(genv, S.ws_monomial(R, S.w_one(R), 1, L))
            z1 = S.g_add(genv, z0, S.g_monomial(genv, 1))
            @test S.g_isequal(
                genv, genv.phi[1], S.g_sub(genv, S.g_pow(genv, z1, p), S.g_pow(genv, z0, p))
            )
            for u in 0:(length(genv.rel) - 1)
                # φ(g_u^p) = φ(g_u)^p: the Frobenius table is compatible with the relations
                @test S.g_isequal(
                    genv,
                    S.g_frobenius(genv, genv.rel[u + 1]),
                    S.g_pow(genv, genv.phi[u + 1], p),
                )
                # R'_u ∈ F^{≥p^{u+1}}: its g^{μ(t)}-coefficient is divisible by z_0^{p^{u+1}-t}
                Rp = genv.Rp[u + 1]
                for t in 0:(L - 1), d in 1:(R.f)
                    c = Rp.c[t + 1][d]
                    k0 = findfirst(k -> !iszero(coeff(c, k)), 0:max(length(c) - 1, 0))
                    k0 === nothing || @test k0 - 1 >= p^(u + 1) - t
                end
            end
            # δ is a δ-structure on the envelope ([AKN24, §3.1, (7), p. 14]) for random elements
            rng = Xoshiro(7)
            for _ in 1:2
                x = S.GElem(
                    [[rand_poly(rng, R, L - t) for _ in 1:(R.f)] for t in 0:(L - 1)], R.N
                )
                y = S.GElem(
                    [[rand_poly(rng, R, L - t) for _ in 1:(R.f)] for t in 0:(L - 1)], R.N
                )
                lhs = S.g_delta(genv, S.g_mul(genv, x, y))
                dx = S.g_delta(genv, x)
                dy = S.g_delta(genv, y)
                rhs = S.g_add(
                    genv,
                    S.g_add(
                        genv,
                        S.g_mul(genv, S.g_pow(genv, x, p), dy),
                        S.g_mul(genv, S.g_pow(genv, y, p), dx),
                    ),
                    S.g_scale_int(genv, S.g_mul(genv, dx, dy), p),
                )
                @test S.g_isequal(genv, lhs, rhs)
                # δ(x + y) = δ(x) + δ(y) - Σ_{0<j<p} (C(p,j)/p) x^j y^{p-j}
                # ([AKN24, §3.1, (5)–(6), p. 14])
                cross = S.g_zero(genv)
                for j in 1:(p - 1)
                    term = S.g_mul(genv, S.g_pow(genv, x, j), S.g_pow(genv, y, p - j))
                    cross = S.g_add(
                        genv, cross, S.g_scale_int(genv, term, div(binomial(p, j), p))
                    )
                end
                lhs = S.g_delta(genv, S.g_add(genv, x, y))
                rhs = S.g_sub(genv, S.g_add(genv, dx, dy), cross)
                @test S.g_isequal(genv, lhs, rhs)
            end
        end
    end
end
