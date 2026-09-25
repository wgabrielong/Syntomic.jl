# Step 4: the change-of-orientation unit w(v).
#
# [AKN, §7.1, Step 4, pp. 55–56]: in Δ_{O_K/W(F_q)[[z_0]]} we have u = E(z_1)/E(z_0) = 1 + ((E(z_1) - E(z_0))/(z_1 - z_0))·b,
# and thus in Δ^(1)_{O_K/W(F_q)[[z_0,z_1]]}
#     w(u) = 1 + φ((E(z_1) - E(z_0))/(z_1 - z_0)) φ_1(g_0).
# Then w(v) = ∏_{r ≥ 0} φ^r(w(u)) ([AKN, Lemma 4.15, p. 29]) is computed "as the fixed point of the
# recursion x = w(u)·φ(x) … by iterating (and reducing) until the value remains unchanged, starting
# with x = w(u)".
#
# Since w(u) - 1 ∈ F^{≥p} and φ multiplies F-weights by p, the k-th iterate x_k = ∏_{r ≤ k} φ^r(w(u))
# agrees with w(v) modulo F^{≥p^{k+2}}, so exactly K = max(0, ⌈log_p L⌉ - 2) iterations are needed.

"""
    divided_difference_E(env) -> GElem

`Q = (E(z_1) - E(z_0))/(z_1 - z_0)` in the g-envelope, where `z_1 = z_0 + g_0`:
`Q = Σ_m (E^{(m+1)}(z_0)/(m+1)!) g_0^m = Σ_k e_k Σ_{1≤j≤k} C(k,j) z_0^{k-j} g_0^{j-1}` ([AKN, §7.1, Step 4]).
"""
function divided_difference_E(env::GEnvelope)
    R = env.R
    L = env.L
    E = env.ed.E
    coeffs = ws_coefficients(E)                        # e_0, …, e_{L-1} (E is a polynomial)
    out = g_zero(env)
    g0pow = g_one(env)                                 # normal form of g_0^m
    for m in 0:(L - 1)
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
        m < L - 1 && (g0pow = g_mul_g0(env, g0pow))
    end
    return GElem(out.c, min(out.prec, E.prec))
end

"""
    phi_one_g0(env) -> GElem

`φ_1(g_0) = λ_0 g_1 + R'_0` ([AKN, Lemma 3.26(1), p. 21] with `u = 0`).
"""
function phi_one_g0(env::GEnvelope)
    R = env.R
    lg1 = g_monomial(env, R.p; coefficient=env.lam.lambda[1])
    return g_add(env, lg1, env.Rp[1])
end

"""
    bk_unit_u(env) -> GElem

`w(u) = 1 + φ(Q)·φ_1(g_0)` with `Q = (E(z_1) - E(z_0))/(z_1 - z_0)` ([AKN, §7.1, Step 4, p. 55]).
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
    bk_unit_v(env, wu) -> GElem

`w(v) = ∏_{r ≥ 0} φ^r(w(u))` ([AKN, Lemma 4.15, p. 29]), the fixed point of `x ↦ w(u)·φ(x)`
([AKN, §7.1, Step 4, p. 56]).
"""
function bk_unit_v(env::GEnvelope, wu::GElem)
    x = wu
    for _ in 1:bk_unit_iterations(env.R.p, env.L)
        x = g_mul(env, wu, g_frobenius(env, x))
    end
    return x
end

"""
    bk_unit_power(env, wv, i) -> GElem

`w(v^i) = w(v)^i`, the factor relating the Breuil–Kisin twists in `η_R(w(s^i)) = w(v^i) w(s^i)`
([AKN, §7.1, Step 5, p. 56]).
"""
bk_unit_power(env::GEnvelope, wv::GElem, i::Integer) = g_pow(env, wv, i)
