# Construction of the g-envelope Δ^(1)_{O_K/A_1}: the elements R'_u, the relations for g_u^p and the
# Frobenius table φ(g_u).
#
# [AKN, §7.1, Step 3, p. 55]: "For the structure of Δ^(1)_{O_K/W(F_q)[[z_0,z_1]]}, use Construction 3.28
# to compute elements R'_u and a δ-ring structure on W(F_q)[[z_0]][g_u | 0 ≤ u ≤ M_g], using the
# additional trick that we may rewrite R'_0 = δ(z_1 - z_0)/δ(d) … Following Lemma 3.24, compute a
# lookup table of relations for g_u^p in terms of those."
#
# Here d = E(z_0) and r = z_1 - z_0 = g_0 (in Δ^(1)), and
#   R'_0     = δ(z_1 - z_0)/δ(E) = δ(E)^{-1} Σ_{0<j<p} (C(p,j)/p) z_0^{p-j} g_0^j,
#   g_u^p    = (-p + λ_u E^{p^{u+1}}) g_{u+1} + E^{p^{u+1}} R'_u          ([AKN, Lemma 3.24, p. 21]),
#   φ(g_u)   = φ(E)^{p^u} (λ_u g_{u+1} + R'_u)                           ([AKN, Lemma 3.26(2), p. 21]),
#   R'_{u+1} = D_u^{-1} (δ(R'_u) + w_1(λ_u g_{u+1}, R'_u)),  D_u = 1 - δ(E^{p^{u+1}} λ_u)
#                                                                         ([AKN, Construction 3.28, p. 22]),
# with w_1(x, y) = -Σ_{0<j<p} (C(p,j)/p) x^j y^{p-j} ([AKN, §3.1, (6), p. 14]). The tables are built in
# the order u = 0, 1, …: R'_u only involves g_0, …, g_u, so δ(R'_u) only needs φ(g_0), …, φ(g_u) and the
# relations for g_0^p, …, g_u^p.

"""
    g_Rprime_zero(env) -> GElem

`R'_0 = δ(z_1 - z_0)/δ(E) = δ(E)^{-1} Σ_{0<j<p} (C(p,j)/p) z_0^{p-j} g_0^j`, where `g_0 = z_1 - z_0`
([AKN, Lemma 3.24, p. 21]; [AKN, §7.1, Step 3, p. 55]).
"""
function g_Rprime_zero(env::GEnvelope)
    R = env.R
    p = R.p
    L = env.L
    dinv = ws_inv(env.ed.deltaE; zprec=L)
    out = g_zero(env)
    for j in 1:min(p - 1, L - 1)
        coeff_j = div(binomial(big(p), j), p)
        s = ws_scale_int(ws_shift(dinv, p - j), coeff_j)       # (C(p,j)/p) z_0^{p-j} δ(E)^{-1}
        out = g_add(env, out, g_monomial(env, j; coefficient=ws_truncate(s, L - j)))
    end
    return GElem(out.c, dinv.prec)
end

"""
    g_relation(env, u) -> GElem

The normal form of `g_u^p`: `(-p + λ_u E^{p^{u+1}}) g_{u+1} + E^{p^{u+1}} R'_u`
([AKN, Lemma 3.24, p. 21]). Requires `R'_u`.
"""
function g_relation(env::GEnvelope, u::Integer)
    R = env.R
    L = env.L
    Epow = env.lam.Epow[u + 2]                                  # E^{p^{u+1}}
    cu = ws_sub(ws_mul(env.lam.lambda[u + 1], Epow; zprec=L), ws_const(R, w_int(R, R.p), L))
    t1 = g_monomial(env, R.p^(u + 1); coefficient=cu)
    t2 = g_scale_series(env, Epow, env.Rp[u + 1])
    return g_add(env, t1, t2)
end

"""
    g_frobenius_generator(env, u) -> GElem

`φ(g_u) = φ(E)^{p^u}(λ_u g_{u+1} + R'_u)` ([AKN, Lemma 3.26(2), p. 21]). Requires `R'_u`.
"""
function g_frobenius_generator(env::GEnvelope, u::Integer)
    R = env.R
    L = env.L
    phiEpow = ws_pow(env.ed.phiE, R.p^u; zprec=L)
    inner = g_add(
        env, g_monomial(env, R.p^(u + 1); coefficient=env.lam.lambda[u + 1]), env.Rp[u + 1]
    )
    return g_scale_series(env, phiEpow, inner)
end

"""
    g_w1(env, x, y) -> GElem

`w_1(x, y) = (x^p + y^p - (x+y)^p)/p = -Σ_{0<j<p} (C(p,j)/p) x^j y^{p-j}` ([AKN, §3.1, (6), p. 14]).
"""
function g_w1(env::GEnvelope, x::GElem, y::GElem)
    p = env.R.p
    xpow = [g_one(env)]
    ypow = [g_one(env)]
    for _ in 1:(p - 1)
        push!(xpow, g_mul(env, xpow[end], x))
        push!(ypow, g_mul(env, ypow[end], y))
    end
    out = g_zero(env)
    for j in 1:(p - 1)
        term = g_mul(env, xpow[j + 1], ypow[p - j + 1])
        out = g_sub(env, out, g_scale_int(env, term, div(binomial(big(p), j), p)))
    end
    return GElem(out.c, min(x.prec, y.prec, out.prec))
end

"""
    g_Rprime_next(env, u) -> GElem

`R'_{u+1} = D_u^{-1}(δ(R'_u) + w_1(λ_u g_{u+1}, R'_u))` ([AKN, Construction 3.28, p. 22]). Requires
`R'_u` and the relations and Frobenius values of `g_0, …, g_u`.
"""
function g_Rprime_next(env::GEnvelope, u::Integer)
    R = env.R
    L = env.L
    Rpu = env.Rp[u + 1]
    lg = g_monomial(env, R.p^(u + 1); coefficient=env.lam.lambda[u + 1])   # λ_u g_{u+1}
    inner = g_add(env, g_delta(env, Rpu), g_w1(env, lg, Rpu))
    return g_scale_series(env, ws_inv(env.lam.D[u + 1]; zprec=L), inner)
end

"""
    GEnvelope(prm, R, ed, lam; threaded=true) -> GEnvelope

Build the g-envelope truncated at `F^{≥L}`, `L = prm.L` ([AKN, §7.1, Step 3, p. 55]): `R'_u`, the
relations for `g_u^p` and `φ(g_u)` for `0 ≤ u ≤ M_g - 1` (for `u = M_g` both vanish modulo
`F^{≥L}`). With `threaded = true` products in the envelope are computed in parallel.
"""
function GEnvelope(
    prm::SyntomicParams,
    R::GaloisRing,
    ed::EisensteinData,
    lam::LambdaData;
    threaded::Bool=true,
)
    digits = [base_p_digits(t, R.p, prm.ng) for t in 0:(prm.L - 1)]
    env = GEnvelope(
        prm,
        prm.L,
        prm.ng,
        R,
        ed,
        lam,
        GElem[],
        GElem[],
        GElem[],
        OnceCache{Vector{Int},GElem}(),
        OnceCache{Int,GElem}(),
        digits,
        threaded,
    )
    prm.L <= 1 && return env
    push!(env.Rp, g_Rprime_zero(env))
    for u in 0:(prm.ng - 2)
        push!(env.rel, g_relation(env, u))
        push!(env.phi, g_frobenius_generator(env, u))
        u + 1 <= prm.ng - 2 && push!(env.Rp, g_Rprime_next(env, u))
    end
    return env
end
