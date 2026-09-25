# The Frobenius lift and δ-structure on the g-envelope Δ^(1)_{O_K/A_1}.
#
# [AKN, §7.1, Step 3, p. 55]: "The δ-ring structure from Construction 3.28 is stored in terms of the
# Frobenius lift φ(g_u) = φ(d)^{p^u}(λ_u g_{u+1} + R'_u)" ([AKN, Lemma 3.26(2), p. 21]). On the
# coefficients W(F_q)[[z_0]], φ is the Frobenius of the Breuil–Kisin prism ([AKN, §2.3, p. 11]).
# Since φ multiplies F-weights by p, φ(x) modulo F^{≥L} only involves the terms of x of F-weight
# < L/p.

"""
    g_phi_monomial(env, t) -> GElem

`φ(g^{μ(t)}) = ∏_u φ(g_u)^{μ_u(t)}` for `p·t < L` (zero otherwise). Memoized.
"""
function g_phi_monomial(env::GEnvelope, t::Integer)
    p = env.R.p
    p * t >= env.L && return g_zero(env)
    t == 0 && return g_one(env)
    cached = lock(env.lock) do
        get(env.phimono_cache, Int(t), nothing)
    end
    cached === nothing || return cached
    d = g_digits(env, t)
    v = findfirst(>(0), d)                   # lowest nonzero digit (1-based)
    v <= length(env.phi) || error("g_phi_monomial: φ(g_$(v - 1)) is not available")
    result = g_mul(env, g_phi_monomial(env, t - p^(v - 1)), env.phi[v])
    lock(env.lock) do
        env.phimono_cache[Int(t)] = result
    end
    return result
end

"""
    g_frobenius(env, x) -> GElem

The Frobenius lift `φ(Σ_t c_t(z_0) g^{μ(t)}) = Σ_t φ(c_t) φ(g^{μ(t)})` on the g-envelope, with
`φ(c) = c^σ(z_0^p)` on coefficients and `φ(g_u)` from [AKN, Lemma 3.26(2), p. 21].
"""
function g_frobenius(env::GEnvelope, x::GElem)
    L = env.L
    p = env.R.p
    R = env.R
    out = [wpoly_zero(R) for _ in 1:L]
    prec = x.prec
    for t in 0:(L - 1)
        p * t >= L && break
        xc = x.c[t + 1]
        all(iszero, xc) && continue
        # φ(c_t) = c_t^σ(z_0^p), known modulo z_0^{p(L-t)}
        len = L                                   # enough for every output component
        phic = [
            truncate(inflate(a, p), len) for
            a in wpoly_sigma(R, [truncate(a, cld(len, p)) for a in xc])
        ]
        m = g_phi_monomial(env, t)
        prec = min(prec, m.prec)
        _g_addmul_series!(env, out, phic, m)
    end
    return GElem(out, prec)
end

"""
    g_divexact_p(env, x) -> GElem

`x/p` for an element all of whose coefficients are divisible by `p`.
"""
function g_divexact_p(env::GEnvelope, x::GElem)
    out = map(0:(env.L - 1)) do t
        ws_divexact_p(WSeries(env.R, x.c[t + 1], 0, env.L - t, x.prec), 1).c
    end
    return GElem(out, x.prec - 1)
end

"""
    g_delta(env, x) -> GElem

`δ(x) = (φ(x) - x^p)/p` on the g-envelope (Construction 3.28 of [AKN, p. 22]); loses one digit of
p-adic precision.
"""
function g_delta(env::GEnvelope, x::GElem)
    return g_divexact_p(env, g_sub(env, g_frobenius(env, x), g_pow(env, x, env.R.p)))
end
