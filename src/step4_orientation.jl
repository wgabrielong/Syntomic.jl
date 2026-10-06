# Step 4: the change-of-orientation unit w(v).
#
# [AKN24, §7.1, Step 4, pp. 55–56]: in Δ_{O_K/W(F_q)[[z_0]]} we have u = E(z_1)/E(z_0) = 1 + ((E(z_1) - E(z_0))/(z_1 - z_0))·b,
# and thus in Δ^(1)_{O_K/W(F_q)[[z_0,z_1]]}
#     w(u) = 1 + φ((E(z_1) - E(z_0))/(z_1 - z_0)) φ_1(g_0).
# Then w(v) = ∏_{r ≥ 0} φ^r(w(u)) ([AKN24, Lemma 4.15, p. 29]) is computed "as the fixed point of the
# recursion x = w(u)·φ(x) … by iterating (and reducing) until the value remains unchanged, starting
# with x = w(u)".
#
# Since w(u) - 1 ∈ F^{≥p} and φ multiplies F-weights by p, the k-th iterate x_k = ∏_{r ≤ k} φ^r(w(u))
# agrees with w(v) modulo F^{≥p^{k+2}}, so exactly K = max(0, ⌈log_p L⌉ - 2) iterations are needed.
# Moreover x_k = w(u)·φ(x_{k-1}) modulo F^{≥T} only needs x_{k-1} modulo F^{≥⌈T/p⌉}, so the k-th
# iteration is computed modulo F^{≥min(L, p^{k+3})} (an F-weight ladder).

"""
    divided_difference_E(env) -> GElem

`Q = (E(z_1) - E(z_0))/(z_1 - z_0)` in the g-envelope, where `z_1 = z_0 + g_0`:
`Q = Σ_m (E^{(m+1)}(z_0)/(m+1)!) g_0^m = Σ_k e_k Σ_{1≤j≤k} C(k,j) z_0^{k-j} g_0^{j-1}` ([AKN24, §7.1, Step 4]).
Only the powers `g_0^m` with `m < deg E` occur.
"""
function divided_difference_E(env::GEnvelope)
    R = env.R
    L = env.L
    E = env.ed.E
    coeffs = ws_coefficients(E)                        # e_0, …, e_{L-1} (E is a polynomial)
    degE = something(findlast(c -> !all(iszero, c), coeffs), 1) - 1
    mmax = min(L, degE)                                # q_m = 0 for m ≥ deg E
    out = g_zero(env)
    g0pow = g_one(env)                                 # normal form of g_0^m
    for m in 0:(mmax - 1)
        # q_m(z_0) = Σ_k e_k C(k, m+1) z_0^{k-m-1}, needed modulo z_0^{L-m}
        qm = [w_zero(R) for _ in 1:(L - m)]
        for k in (m + 1):(length(coeffs) - 1)
            idx = k - m - 1
            idx < L - m || break
            qm[idx + 1] = w_add(
                R, qm[idx + 1], w_mul_int(R, coeffs[k + 1], binomial(big(k), m + 1))
            )
        end
        if !all(all(iszero, c) for c in qm)
            s = ws_from_coefficients(R, qm, L)
            out = g_add(env, out, g_scale_series(env, s, g0pow))
        end
        m < mmax - 1 && (g0pow = g_mul_g0(env, g0pow))
    end
    return GElem(out.c, min(out.prec, E.prec))
end

"""
    phi_one_g0(env) -> GElem

`φ_1(g_0) = λ_0 g_1 + R'_0` ([AKN24, Lemma 3.26(1), p. 21] with `u = 0`).
"""
function phi_one_g0(env::GEnvelope)
    R = env.R
    lg1 = g_monomial(env, R.p; coefficient=env.lam.lambda[1])
    return g_add(env, lg1, env.Rp[1])
end

"""
    bk_unit_u(env) -> GElem

`w(u) = 1 + φ(Q)·φ_1(g_0)` with `Q = (E(z_1) - E(z_0))/(z_1 - z_0)` ([AKN24, §7.1, Step 4, p. 55]).
"""
function bk_unit_u(env::GEnvelope)
    Q = divided_difference_E(env)
    return g_add(env, g_one(env), g_mul(env, g_frobenius(env, Q), phi_one_g0(env)))
end

"""
    bk_unit_iterations(p, L) -> Int

The number `K = max(0, ⌈log_p L⌉ - 2)` of iterations `x ← w(u)·φ(x)` after `x = w(u)` needed to
determine `w(v)` modulo `F^{≥L}`.
"""
function bk_unit_iterations(p::Integer, L::Integer)
    k = 0
    while big(p)^(k + 2) < L
        k += 1
    end
    return k
end

"""
    bk_unit_ladder(p, L) -> Vector{Int}

The F-truncations `min(L, p^{k+3})`, `1 ≤ k ≤ K = bk_unit_iterations(p, L)`, at which the iterates
`x_k = w(u)·φ(x_{k-1})` of [AKN24, §7.1, Step 4, p. 56] are computed: `x_{k-1}` is known modulo
`F^{≥min(L, p^{k+2})}`, so `φ(x_{k-1})` is known modulo `F^{≥min(pL, p^{k+3})}`. The last one is
`L`.
"""
function bk_unit_ladder(p::Integer, L::Integer)
    return [Int(min(big(L), big(p)^(k + 3))) for k in 1:bk_unit_iterations(p, L)]
end

"""
    bk_unit_v(env, wu; L=g_length(wu)) -> GElem

`w(v) = ∏_{r ≥ 0} φ^r(w(u))` ([AKN24, Lemma 4.15, p. 29]), the fixed point of
`x ↦ w(u)·φ(x)` ([AKN24, §7.1, Step 4, p. 56]), modulo `F^{≥L}` (`L ≤ g_length(wu)`):
starting from `x_0 = w(u)`, the `K` iterations `x_k = w(u)·φ(x_{k-1})` are computed modulo
`F^{≥min(L, p^{k+3})}` (see `bk_unit_ladder`), which gives `x_K = w(v)` modulo `F^{≥L}`. Throws an
`ErrorException` if `x_K - x_{K-1} ∉ F^{≥p^{K+1}}` (an internal consistency check:
`x_K - x_{K-1} = x_{K-1}(φ^K(w(u)) - 1)`).
"""
function bk_unit_v(env::GEnvelope, wu::GElem; L::Integer=g_length(wu))
    L = _g_check_length(L, g_length(wu))
    x = wu
    prev = wu
    ladder = bk_unit_ladder(env.R.p, L)
    for T in ladder
        prev = x
        x = g_mul(env, wu, g_frobenius(env, x; L=T); L=T)
    end
    isempty(ladder) || _bk_unit_check(env, x, prev, length(ladder))
    return x
end

# The consistency check of `bk_unit_v`: x_K ≡ x_{K-1} modulo F^{≥p^{K+1}}
# ([AKN24, §7.1, Step 4, p. 56]).
function _bk_unit_check(env::GEnvelope, x::GElem, prev::GElem, K::Integer)
    T = Int(min(big(env.R.p)^(K + 1), g_length(prev), g_length(x)))
    return _bk_unit_check_modulo(env, x, prev, K, T, "w(v)")
end

# x_K ≡ x_{K-1} modulo F^{≥T}, or an internal error.
function _bk_unit_check_modulo(
    env::GEnvelope, x::GElem, prev::GElem, K::Integer, T::Integer, what::AbstractString
)
    p = env.R.p
    g_isequal(env, x, prev; L=T) || error(
        "internal error in [AKN24, §7.1, Step 4, p. 56]: the iterates x_$K and x_$(K - 1) for " *
        "$what differ modulo F^{≥$T} (p = $p, L = $(g_length(x))); w(u) - 1 must lie in F^{≥p}",
    )
    return nothing
end
