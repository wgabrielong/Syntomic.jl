# Step 5: ∇ for O_K.
#
# [AKN, §7.1, Step 5, p. 56]: "A basis for Δ^(1)_{O_K/W(F_q)[[z_0]]}{i} is given by z_0^k w(s^i) for
# 1 ≤ k ≤ in-1 and ∇(z_0^k w(s^i)) can be computed as (w(s^{i-1}) times) the coefficient of g_0 with
# respect to the W(F_q)[[z_0]]-basis ∏ g_u^{e_u} in w(v^i)(z_0 + g_0)^k - z_0^k, after reducing."
#
# Since θ (the g_0-coefficient) is W(F_q)[[z_0]]-linear and θ(z_0^k) = 0,
#     ∇(z_0^k w(s^i)) = Σ_{j=0}^{k} C(k,j) z_0^{k-j} θ(w(v^i) g_0^j),
# so after computing y_j = w(v^i) g_0^j (a chain of multiplications by g_0) all columns are
# independent.

"""
    nabla_theta_table(env, vi) -> Vector{WSeries}

`θ(w(v^i) g_0^j)` for `0 ≤ j ≤ L-1`, each a series known modulo `z_0^{L-1}`: the coefficients of
`g_0` needed in [AKN, §7.1, Step 5, p. 56] (θ as in [AKN, Def. 4.41, p. 37]).
"""
function nabla_theta_table(env::GEnvelope, vi::GElem)
    thetas = WSeries[]
    y = vi
    for j in 0:(env.L - 1)
        push!(thetas, g_theta(env, y))
        j < env.L - 1 && (y = g_mul_g0(env, y))
    end
    return thetas
end

"""
    nabla_OK_column(env, thetas, k) -> WSeries

`θ(w(v^i)(z_0 + g_0)^k - z_0^k) = Σ_{j=0}^{k} C(k,j) z_0^{k-j} θ(w(v^i) g_0^j)`, modulo `z_0^{L-1}`
([AKN, §7.1, Step 5, p. 56]).
"""
function nabla_OK_column(env::GEnvelope, thetas::Vector{WSeries}, k::Integer)
    R = env.R
    js = [j for j in 0:k if k - j < env.L - 1]
    # the sum is known modulo z_0^zp and p^prec
    zp = minimum((thetas[j + 1].zprec + k - j for j in js); init=env.L - 1)
    prec = minimum((thetas[j + 1].prec for j in js); init=R.N)
    out = wpoly_zero(R)
    t = zero(R.Znz)
    for j in js
        th = thetas[j + 1]
        s = k - j + th.zval                   # z_0^{k-j} θ_j, relative to z_0^0
        s >= zp && continue
        c = mod(ZZ(binomial(big(k), j)), R.pN)
        iszero(c) && continue
        for d in 1:(R.f)
            iszero(th.c[d]) && continue
            t = poly_shift_left!(poly_set_trunc!(t, th.c[d], zp - s), s)
            out[d] = poly_addmul!(out[d], t, c)
        end
    end
    return WSeries(R, out, 0, zp, prec)
end

"""
    nabla_OK_matrix(env, vi; threaded=true) -> WMatrix

The matrix of `∇: F^{[1,in-1]}Δ^(1)_{O_K/A_0}{i} → F^{[1,in-1]}Δ^(1),∇_{O_K/A_0}{i}` in the bases
`z_0^k w(s^i)` (`1 ≤ k ≤ L-1`, columns) and `z_0^m w(s^{i-1})` (`0 ≤ m ≤ L-2`, rows), where
`vi = w(v^i)` ([AKN, §7.1, Step 5, p. 56]).
"""
function nabla_OK_matrix(env::GEnvelope, vi::GElem; threaded::Bool=true)
    L = env.L
    thetas = nabla_theta_table(env, vi)
    cols = tmap(k -> nabla_OK_column(env, thetas, k), 1:(L - 1); threaded=threaded)
    A = [ws_coefficient(cols[k], m) for m in 0:(L - 2), k in 1:(L - 1)]
    prec = minimum((c.prec for c in cols); init=vi.prec)
    return WMatrix(env.R, A, prec)
end
