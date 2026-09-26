# The g-side of [AKN, §7.1, Steps 3–5]: arithmetic modulo F^{≥L'} in the envelope of O_K over
# W(F_q)[[z_0, z_1]], the F-weight ladder for w(v), the powers w(v^i), and the transposed chain of
# ∇_{O_K}. Everything is compared with straightforward computations modulo F^{≥L}.

# The g-envelope of X in weight i (F-precision L = n·i) at the precision N (a priori by default).
function gside_env(X, i; N=nothing, threaded=true, eager=threaded && S.threads_available())
    prm = S.SyntomicParams(X.p, X.f, X.n, i)
    NN = N === nothing ? S.apriori_working_precision(X.p, X.n, i) : N
    R = S.GaloisRing(X.p, X.htilde, NN)
    ed = S.EisensteinData(R, X, prm.L)
    lam = S.LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)
    return S.GEnvelope(prm, R, ed, lam; threaded=threaded, eager=eager)
end

# The envelope data of env with empty tables (to recompute the levels of the normal forms).
function gside_empty_tables(env)
    return S.GEnvelope(
        env.prm,
        env.L,
        env.ng,
        env.R,
        env.ed,
        env.lam,
        env.Rp,
        env.rel,
        env.phi,
        S.OnceCache{Vector{Int},S.GElem}(),
        S.OnceCache{Int,S.GElem}(),
        env.digits,
        env.threaded,
    )
end

# The normal forms of a table, by exponent vector, as integers with their precisions.
function gside_table(env)
    return Dict(k => (gside_key(v.value), v.value.prec) for (k, v) in env.nf_cache.dict)
end

# A random element of the g-envelope modulo F^{≥L}.
function rand_gelem(rng, env, L=env.L)
    R = env.R
    return S.GElem([[rand_poly(rng, R, L - t) for _ in 1:(R.f)] for t in 0:(L - 1)], R.N)
end

# The components of x as integers (comparable across Galois rings).
gside_key(x::S.GElem) = [[lift.(coefficients(a)) for a in c] for c in x.c]

# Whether x and y have the same components and precision.
gside_same(x::S.GElem, y::S.GElem) = x.c == y.c && x.prec == y.prec

# Whether x and y (possibly over different Galois rings with the same p^N) have the same
# components and precision.
gside_same_values(x::S.GElem, y::S.GElem) = gside_key(x) == gside_key(y) && x.prec == y.prec

# Whether x has the components of ref and at least its precision (a computation modulo F^{≥L'} may
# use fewer normal forms, whose precision the result records).
gside_agree(x::S.GElem, ref::S.GElem) = x.c == ref.c && x.prec >= ref.prec

# w(v) by K iterations modulo F^{≥L} (no ladder).
function gside_bk_unit_v_full(env, wu)
    x = wu
    for _ in 1:S.bk_unit_iterations(env.R.p, env.L)
        x = S.g_mul(env, wu, S.g_frobenius(env, x))
    end
    return x
end

# Q = (E(z_1) - E(z_0))/(z_1 - z_0) with the powers g_0^m for all m < L.
function gside_divided_difference_full(env)
    R, L = env.R, env.L
    coeffs = S.ws_coefficients(env.ed.E)
    out = S.g_zero(env)
    g0pow = S.g_one(env)
    for m in 0:(L - 1)
        qm = [S.w_zero(R) for _ in 1:(L - m)]
        for k in (m + 1):(length(coeffs) - 1)
            k - m - 1 < L - m || break
            qm[k - m] = S.w_add(
                R, qm[k - m], S.w_mul_int(R, coeffs[k + 1], binomial(big(k), m + 1))
            )
        end
        if !all(all(iszero, c) for c in qm)
            s = S.ws_from_coefficients(R, qm, L)
            out = S.g_add(env, out, S.g_scale_series(env, s, g0pow))
        end
        m < L - 1 && (g0pow = S.g_mul_g0(env, g0pow))
    end
    return S.GElem(out.c, min(out.prec, env.ed.E.prec))
end

# The rings of the tests: f = 1 and f > 1, unramified and ramified, p = 2, 3, 5.
function gside_rings()
    K, a = qadic_field(2, 2; precision=30, cached=false)
    Kz, z = polynomial_ring(K, :z; cached=false)
    K3, _ = qadic_field(3, 1; precision=30, cached=false)
    K3z, z3 = polynomial_ring(K3, :z; cached=false)
    return [
        (zpn(2, 2), 9),
        (zpn(2, 3), 7),
        (zpn(3, 2), 8),
        (zpn(5, 2), 7),
        (truncated_witt_vectors(2, 2, 2), 7),
        (truncated_witt_vectors(3, 2, 2), 5),
        (finite_chain_ring(z^2 + 2a * z + 2a, 3), 6),
        (finite_chain_ring(z3^2 - 3, 3), 5),
        (finite_chain_ring(z^3 + 2, 3), 5),
    ]
end

@testset "g-side" begin
    rng = Xoshiro(314159)

    @testset "arithmetic modulo F^{≥L'}" begin
        for (X, i) in
            ((zpn(2, 2), 8), (truncated_witt_vectors(3, 2, 2), 5), gside_rings()[7])
            env = gside_env(X, i)
            L, p, R = env.L, env.R.p, env.R
            # the cached normal forms must not change
            nf_before = Dict(k => gside_key(v.value) for (k, v) in env.nf_cache.dict)
            for _ in 1:2
                x, y = rand_gelem(rng, env), rand_gelem(rng, env)
                xy = S.g_mul(env, x, y)
                xx = S.g_mul(env, x, x)
                phix = S.g_frobenius(env, x)
                g0x = S.g_mul_g0(env, x)
                x3 = S.g_pow(env, x, 3)
                for Lt in (1, 2, p + 1, div(L, 2), L - 1, L)
                    tr(u) = S.g_truncate(env, u, Lt)
                    xt, yt = tr(x), tr(y)
                    @test S.g_length(xt) == Lt
                    @test gside_agree(S.g_mul(env, x, y; L=Lt), tr(xy))
                    @test gside_agree(S.g_mul(env, xt, yt), tr(xy))
                    @test gside_agree(S.g_mul(env, xt, y), tr(xy))
                    @test gside_agree(S.g_mul(env, x, x; L=Lt), tr(xx))
                    @test gside_agree(S.g_mul(env, xt, xt), tr(xx))
                    @test gside_agree(S.g_mul_g0(env, x; L=Lt), tr(g0x))
                    @test gside_agree(S.g_mul_g0(env, xt), tr(g0x))
                    @test gside_agree(S.g_pow(env, x, 3; L=Lt), tr(x3))
                    @test gside_agree(S.g_pow(env, xt, 0), S.g_one(env; L=Lt))
                    # φ modulo F^{≥L'} only needs x modulo F^{≥⌈L'/p⌉}
                    @test gside_agree(S.g_frobenius(env, x; L=Lt), tr(phix))
                    xs = S.g_truncate(env, x, cld(Lt, p))
                    @test gside_agree(S.g_frobenius(env, xs; L=Lt), tr(phix))
                    @test S.g_length(S.g_frobenius(env, xs)) == min(L, p * cld(Lt, p))
                    th = S.g_theta(env, x; L=Lt)
                    @test th.zprec == max(Lt - 1, 0)
                    @test S.ws_isequal(th, S.ws_truncate(S.g_theta(env, x), Lt - 1))
                    @test gside_agree(S.g_add(env, xt, y), tr(S.g_add(env, x, y)))
                    @test gside_agree(S.g_sub(env, x, yt), tr(S.g_sub(env, x, y)))
                    @test S.g_isequal(env, xt, x)
                    @test S.g_isequal(env, S.g_mul(env, xt, yt), xy; L=Lt)
                end
                @test_throws ArgumentError S.g_mul(env, S.g_truncate(env, x, 3), y; L=4)
                @test_throws ArgumentError S.g_frobenius(
                    env, S.g_truncate(env, x, 2); L=2 * p + 1
                )
                @test_throws ArgumentError S.g_truncate(env, x, L + 1)
            end
            @test all(
                gside_key(v.value) == nf_before[k] for
                (k, v) in env.nf_cache.dict if haskey(nf_before, k)
            )
            # F-weight valuations: z_0^k g^{μ(t)} has F-weight k + t
            for t in 0:(L - 1), k in 0:2
                s = S.ws_monomial(R, S.w_one(R), k, L - t)
                x = S.g_monomial(env, t; coefficient=s)
                @test S.g_weight_valuation(x) == min(k + t, L)
            end
            @test S.g_weight_valuation(S.g_zero(env)) == L
        end
    end

    @testset "the table of normal forms, level by level" begin
        for (X, i) in gside_rings()
            # every normal form that products, g_0·x, φ and the transposed chain need, computed
            # when first needed
            lazy = gside_env(X, i; threaded=false, eager=false)
            seed = rand(rng, UInt)
            elems(env) = (r = Xoshiro(seed); (rand_gelem(r, env), rand_gelem(r, env)))
            x, y = elems(lazy)
            function results(env, x, y)
                return [
                    S.g_mul(env, x, y),
                    S.g_mul(env, x, x),
                    S.g_mul_g0(env, x),
                    S.g_frobenius(env, x),
                    S.g_pow(env, y, 3),
                ]
            end
            lazy_results = results(lazy, x, y)
            S.theta_chain(lazy; threaded=false)
            # built with the envelope, level by level, threaded and serial
            eager = gside_env(X, i; threaded=true, eager=true)
            serial = gside_env(X, i; threaded=false, eager=true)
            table = gside_table(eager)
            @test gside_table(serial) == table
            @test issubset(keys(lazy.nf_cache.dict), keys(table))
            @test all(table[k] == v for (k, v) in gside_table(lazy))
            # nothing is computed afterwards, and the products are the same
            n = length(eager.nf_cache)
            xe, ye = elems(eager)
            eager_results = results(eager, xe, ye)
            S.theta_chain(eager; threaded=true)
            @test length(eager.nf_cache) == n
            @test all(
                gside_same_values(a, b) for (a, b) in zip(eager_results, lazy_results)
            )
            # the levels: every normal form only needs normal forms of earlier levels; with all
            # relations every one of the table is reached, with fewer only those they determine
            empty = gside_empty_tables(eager)
            p, umax = eager.R.p, eager.ng - 2
            levels = S.g_table_levels(empty, umax)
            seen = Set{Vector{Int}}()
            ordered = true
            for level in levels
                for node in level, (_, s) in node.terms
                    S._is_reduced(p, s) || s in seen || (ordered = false)
                end
                union!(seen, [node.e for node in level])
            end
            @test ordered
            @test seen == Set(keys(table))
            @test sum(length, levels; init=0) == length(seen)
            @test isempty(S.g_table_levels(eager, umax))
            if umax >= 1
                part = S.g_table_levels(empty, umax - 1)
                partial = Set(node.e for level in part for node in level)
                @test issubset(partial, seen)
                @test all(findfirst(>=(p), e) <= umax for e in partial)
                @test length(partial) < length(seen)
            end
        end
    end

    @testset "Step 4: the F-weight ladder for w(v)" begin
        @test S.bk_unit_ladder(2, 60) == [16, 32, 60, 60]
        @test S.bk_unit_ladder(2, 4) == Int[]
        @test S.bk_unit_ladder(2, 5) == [5]
        @test S.bk_unit_ladder(3, 48) == [48, 48]
        @test S.bk_unit_ladder(5, 25) == Int[]
        @test S.bk_unit_ladder(5, 26) == [26]
        @test all(
            length(S.bk_unit_ladder(p, L)) == S.bk_unit_iterations(p, L) for p in (2, 3, 5),
            L in 1:200
        )
        @test S.bk_unit_fixed_point_ladder(2, 60) == [4, 8, 15, 30, 60]
        @test S.bk_unit_fixed_point_ladder(3, 9) == [9]
        for p in (2, 3, 5), L in 2:150
            Ts = S.bk_unit_fixed_point_ladder(p, L)
            @test Ts[1] <= p^2 &&
                Ts[end] == L &&
                all(Ts[k] <= p * Ts[k - 1] for k in 2:length(Ts))
        end
        for (X, i) in gside_rings()
            env = gside_env(X, i)
            @test gside_same(
                S.divided_difference_E(env), gside_divided_difference_full(env)
            )
            wu = S.bk_unit_u(env)
            wv = S.bk_unit_v(env, wu)
            @test gside_same(wv, gside_bk_unit_v_full(env, wu))
            # w(v) is the fixed point of x ↦ w(u)·φ(x)
            @test gside_agree(S.bk_unit_fixed_point(env, wu), wv)
            @test S.g_isequal(env, S.g_mul(env, wu, S.g_frobenius(env, wv)), wv)
        end
        # the consistency check: 1 + g_0 does not lie in 1 + F^{≥p}
        env = gside_env(zpn(2, 2), 8)
        bad = S.g_add(env, S.g_one(env), S.g_monomial(env, 1))
        @test_throws ErrorException S.bk_unit_v(env, bad)
        @test_throws ArgumentError S.bk_unit_fixed_point(env, bad)
    end

    @testset "Step 5: the transposed chain" begin
        for (X, i) in gside_rings(), threaded in (true, false)
            env = gside_env(X, i; threaded=threaded)
            L = env.L
            wu = S.bk_unit_u(env)
            wv = S.bk_unit_v(env, wu)
            chain = S.theta_chain(env; threaded=threaded)
            @test length(chain.r) == L
            @test [length(r) for r in chain.r] == [L - j for j in 0:(L - 1)]
            for k in unique([1, 2, i])
                vk = S.g_pow(env, wv, k)
                fwd = S.nabla_OK_matrix(env, vk; method=:forward, threaded=threaded)
                trn = S.nabla_OK_matrix(env, vk; threaded=threaded)
                @test fwd.A == trn.A && fwd.prec == trn.prec
                @test S.nabla_OK_matrix(env, vk; chain=chain, threaded=threaded).A == fwd.A
                th1 = S.nabla_theta_table(env, vk)
                th2 = S.nabla_theta_table(env, chain, vk; threaded=threaded)
                @test all(
                    a.c == b.c &&
                    a.zval == b.zval &&
                    a.zprec == b.zprec &&
                    a.prec == b.prec for (a, b) in zip(th1, th2)
                )
                # modulo F^{≥L'}: the top-left block, from the chain at L or at L'
                for Lt in (2, div(L, 2), L - 1)
                    Mt = S.nabla_OK_matrix(env, vk; L=Lt, chain=chain, threaded=threaded)
                    @test size(Mt) == (Lt - 1, Lt - 1)
                    @test Mt.A == fwd.A[1:(Lt - 1), 1:(Lt - 1)]
                    vt = S.g_truncate(env, vk, Lt)
                    @test S.nabla_OK_matrix(env, vt).A == Mt.A
                    @test S.nabla_OK_matrix(env, vt; method=:forward).A == Mt.A
                end
            end
            @test_throws ArgumentError S.nabla_OK_matrix(env, wv; method=:sideways)
        end
    end

    @testset "Step 4: the powers w(v^i)" begin
        # the cost model
        @test S.binary_power_cost(2, 1, 50) == 0
        @test S.binary_power_cost(2, 2, 50) == S.g_mul_cost(2, 50; square=true)
        @test S.g_mul_cost(3, 40; w=40) == 0
        @test S.g_mul_cost(3, 40; square=true) < S.g_mul_cost(3, 40)
        @test S.g_mul_cost(3, 40; support=2) < S.g_mul_cost(3, 40)
        @test S.binomial_table_size(2, 60, 100) == 29
        @test S.binomial_table_size(0, 60, 7) == 7
        @test S.binomial_table_size(61, 60, 7) == 0
        # measured: for Z/4, i = 30 binary powering of w(v) beats the binomial table, and the
        # fixed point beats both (benchmark/gside.jl)
        @test S.bk_unit_power_method(2, 30, 60, 2) == :binary
        @test S.bk_unit_v_power_method(2, 30, 60, 2, 2) == :fixed_point
        @test S.bk_unit_v_powers_method(2, 1:12, Dict(i => 5 * i for i in 1:12), 2, 2) ==
            :fixed_point
        for (X, i) in gside_rings()
            env = gside_env(X, i)
            L = env.L
            wu = S.bk_unit_u(env)
            wv = S.bk_unit_v(env, wu)
            for k in unique([0, 1, 2, 3, i, i + 5])
                ref = S.g_pow(env, wv, k)
                for m in (:binary, :binomial, :auto)
                    @test gside_same(S.bk_unit_power(env, wv, k; method=m), ref)
                end
                B = S.binomial_powers(env, wv, max(k, 1))
                @test gside_same(S.binomial_power(env, B, k), ref)
                for Lt in (1, div(L, 2), L - 1)
                    @test S.binomial_power(env, B, k; L=Lt).c ==
                        S.g_truncate(env, ref, Lt).c
                    @test S.bk_unit_power(env, wv, k; L=Lt).c ==
                        S.g_truncate(env, ref, Lt).c
                end
                k == 0 && continue
                @test S.bk_unit_u_power(env, wu, k).c == S.g_pow(env, wu, k).c
                nabla = S.nabla_OK_matrix(env, ref)
                for m in (:fixed_point, :binary, :binomial, :auto)
                    v = S.bk_unit_v_power(env, wu, k; method=m)
                    @test gside_agree(v, ref)
                    # the same ∇_{O_K}, also its precision
                    M = S.nabla_OK_matrix(env, v)
                    @test M.A == nabla.A && M.prec == nabla.prec
                end
                vt = S.bk_unit_v_power(env, wu, k; L=div(L, 2), method=:fixed_point)
                @test vt.c == S.g_truncate(env, ref, div(L, 2)).c
            end
            # both ways of computing w(u)^i (binomial table or products)
            for k in (2, L)
                @test S.bk_unit_u_power(env, wu, k).c == S.g_pow(env, wu, k).c
            end
            # a sweep: w(v^k) modulo F^{≥2k}
            ks = [1, 2, 3, 5, div(L, 2)]
            Ls = Dict(k => min(L, 2 * k) for k in ks)
            refs = Dict(k => S.g_truncate(env, S.g_pow(env, wv, k), Ls[k]) for k in ks)
            for m in (:incremental, :binary, :binomial, :auto)
                P = S.bk_unit_powers(env, wv, ks, Ls; method=m)
                @test all(P[k].c == refs[k].c for k in ks)
            end
            for m in (:fixed_point, :incremental, :binary, :binomial, :auto)
                P = S.bk_unit_v_powers(env, wu, ks, Ls; method=m)
                @test all(P[k].c == refs[k].c for k in ks)
            end
            @test_throws ArgumentError S.bk_unit_power(env, wv, 2; method=:unary)
            @test_throws ArgumentError S.bk_unit_v_power(env, wu, 2; method=:unary)
            @test_throws ArgumentError S.bk_unit_powers(env, wv, ks, Ls; method=:unary)
            @test_throws ArgumentError S.bk_unit_v_powers(env, wu, ks, Ls; method=:unary)
            @test_throws ArgumentError S.bk_unit_v_power(env, wu, 0)
        end
    end

    @testset "sweeps: ∇_{O_K} modulo F^{≥n_max·i}" begin
        for (X, ns, is) in
            ((zpn(2, 2), 2:3, 1:5), (truncated_witt_vectors(3, 2, 2), 2:2, 1:3))
            Xc = S.computing_ring(S.with_power(X, last(ns)), true)
            Lmax = last(ns) * last(is)
            N = 40
            nablas = S.shared_nabla_OK(Xc, is, Lmax, N)
            # the same data at F-precision Lmax, powers by binary powering, the forward chain
            prm = S.SyntomicParams(Xc.p, Xc.f, Lmax, 1)
            R = S.GaloisRing(Xc.p, Xc.htilde, N)
            ed = S.EisensteinData(R, Xc, Lmax)
            lam = S.LambdaData(ed, max(prm.ng, 2) - 2)
            env = S.GEnvelope(prm, R, ed, lam)
            wv = gside_bk_unit_v_full(env, S.bk_unit_u(env))
            for i in is
                Li = last(ns) * i
                M = S.nabla_OK_matrix(env, S.g_pow(env, wv, i); method=:forward)
                @test size(nablas[i]) == (Li - 1, Li - 1)
                @test nablas[i].entries == S.to_plain(M).entries[1:(Li - 1), 1:(Li - 1)]
                @test nablas[i].prec >= M.prec
            end
        end
    end

    if LONG_TESTS
        @testset "long: larger F-precision" begin
            for (X, i) in
                ((zpn(2, 2), 20), (zpn(3, 2), 16), (truncated_witt_vectors(2, 2, 3), 8))
                env = gside_env(X, i)
                wu = S.bk_unit_u(env)
                wv = S.bk_unit_v(env, wu)
                @test gside_same(wv, gside_bk_unit_v_full(env, wu))
                ref = S.g_pow(env, wv, i)
                for m in (:fixed_point, :binomial)
                    @test gside_agree(S.bk_unit_v_power(env, wu, i; method=m), ref)
                end
                fwd = S.nabla_OK_matrix(env, ref; method=:forward)
                trn = S.nabla_OK_matrix(env, S.bk_unit_v_power(env, wu, i))
                @test fwd.A == trn.A && fwd.prec == trn.prec
            end
        end
    end
end

@testset "Step 4: consistency checks and truncated w(u)" begin
    # A Frobenius that does not multiply F-weights by p (φ(g_1) gets an extra g_1) is rejected by
    # the consistency check on every path, including the default fixed-point path.
    for (p, coeffs, n, i) in ((2, [2, 2, 1], 5, 5), (3, [3, 0, 1], 5, 6))
        X = finite_chain_ring(coeffs; p=p, f=1, n=n)
        good = gside_env(X, i; N=16, threaded=false)
        bad = gside_env(X, i; N=16, threaded=false)
        wu = S.bk_unit_u(good)
        U = S.bk_unit_u_power(good, wu, i)
        @test S.g_isequal(
            good,
            S.bk_unit_fixed_point(good, U),
            S.bk_unit_v_power(good, wu, i; method=:binary),
        )
        bad.phi[2] = S.g_add(bad, bad.phi[2], S.g_monomial(bad, p))
        empty!(bad.phimono_cache.dict)
        @test_throws ErrorException S.bk_unit_fixed_point(bad, U)
        @test_throws ErrorException S.bk_unit_v(bad, wu)
    end
    # w(u) known only modulo F^{≥L'}: every method gives w(v^i) modulo F^{≥L'}
    X = finite_chain_ring([2, 2, 1]; p=2, f=1, n=6)
    env = gside_env(X, 4)
    wu = S.bk_unit_u(env)
    for Lp in (5, 11, env.L - 3)
        short = S.g_truncate(env, wu, Lp)
        ref = S.g_truncate(env, S.bk_unit_v(env, wu), Lp)
        @test S.g_isequal(env, S.bk_unit_v(env, short), ref)
        for method in (:fixed_point, :binary, :binomial)
            v = S.bk_unit_v_power(env, short, 3; method=method)
            refi = S.g_truncate(env, S.bk_unit_v_power(env, wu, 3; method=:binary), Lp)
            @test S.g_length(v) == Lp && S.g_isequal(env, v, refi)
        end
    end
end
