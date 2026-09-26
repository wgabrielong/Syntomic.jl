# Step 5: ∇ for O_K.
#
# [AKN, §7.1, Step 5, p. 56]: "A basis for Δ^(1)_{O_K/W(F_q)[[z_0]]}{i} is given by z_0^k w(s^i) for
# 1 ≤ k ≤ in-1 and ∇(z_0^k w(s^i)) can be computed as (w(s^{i-1}) times) the coefficient of g_0 with
# respect to the W(F_q)[[z_0]]-basis ∏ g_u^{e_u} in w(v^i)(z_0 + g_0)^k - z_0^k, after reducing."
#
# Since θ (the g_0-coefficient) is W(F_q)[[z_0]]-linear and θ(z_0^k) = 0,
#     ∇(z_0^k w(s^i)) = Σ_{j=0}^{k} C(k,j) z_0^{k-j} θ(w(v^i) g_0^j),
# so after computing θ(w(v^i) g_0^j) all columns are independent. These are computed either by the
# forward chain y_0 = w(v^i), y_{j+1} = g_0·y_j, or by the transposed chain: writing
# w(v^i) = Σ_w c_w g^{μ(w)} and r_j[w] = θ(g^{μ(w)} g_0^j) (after reducing),
#     θ(w(v^i) g_0^j) = Σ_w c_w r_j[w],
#     r_0[w] = δ_{w,1},   r_{j+1}[w] = Σ_{w'} G_0[w][w'] r_j[w'],
# where G_0[w][w'] is the coefficient of g^{μ(w')} in the normal form of g^{μ(w)} g_0; this is
# g^{μ(w+1)} unless the lowest base-p digit of w is p - 1, so then r_{j+1}[w] = r_j[w+1]. The r_j
# depend neither on i nor on w(v): in a sweep over i they are computed once.

"""
    nabla_theta_table(env, vi; L=g_length(vi)) -> Vector{WSeries}

`θ(w(v^i) g_0^j)` for `0 ≤ j ≤ L-1`, each a series known modulo `z_0^{L-1}`, where `vi = w(v^i)`
is known modulo `F^{≥L}`: the coefficients of `g_0` needed in [AKN, §7.1, Step 5, p. 56] (θ as in
[AKN, Def. 4.41, p. 37]), computed by the forward chain `y_{j+1} = g_0·y_j`, `y_0 = w(v^i)`.
"""
function nabla_theta_table(env::GEnvelope, vi::GElem; L::Integer=g_length(vi))
    thetas = WSeries[]
    y = g_truncate(env, vi, L)
    for j in 0:(L - 1)
        push!(thetas, g_theta(env, y))
        j < L - 1 && (y = g_mul_g0(env, y))
    end
    return thetas
end

"""
    ThetaChain

The transposed chain of [AKN, §7.1, Step 5, p. 56] modulo `F^{≥L}`: `r[j+1][w+1]` is
`θ(g^{μ(w)} g_0^j)` (after reducing), a series known modulo `z_0^{L-1}` and divisible by
`z_0^{w+j-1}`, for `0 ≤ j ≤ L-1` and `0 ≤ w < L-j` (it vanishes modulo `z_0^{L-1}` for larger
`w`). The series are shared between `r[j+1]` and `r[j+2]` where no carry occurs. Built by
`theta_chain`; used by `nabla_theta_table(env, chain, vi)`.
"""
struct ThetaChain
    L::Int
    r::Vector{Vector{WSeries}}
end

"""
    theta_chain(env; L=env.L, threaded=env.threaded) -> ThetaChain

The transposed chain `r_j[w] = θ(g^{μ(w)} g_0^j)` modulo `F^{≥L}`, from `r_0[w] = δ_{w,1}` and
`r_{j+1}[w] = Σ_{w'} G_0[w][w'] r_j[w']` ([AKN, §7.1, Step 5, p. 56]). It depends only on the
envelope, not on the weight `i`. For each `j`, the entries with a carry are computed in parallel
when `threaded`.
"""
function theta_chain(env::GEnvelope; L::Integer=env.L, threaded::Bool=env.threaded)
    L = _g_check_length(L, env.L)
    R = env.R
    L >= 2 || return ThetaChain(L, Vector{WSeries}[])
    zp = L - 1
    r = [[w == 1 ? ws_one(R, zp) : ws_zero(R, zp) for w in 0:(L - 1)]]
    for j in 0:(L - 2)
        push!(r, _theta_chain_step(env, r[end], j, L, threaded))
    end
    return ThetaChain(L, r)
end

# r_{j+1} from rj = r_j modulo F^{≥L}: r_{j+1}[w] = r_j[w+1] without a carry, a sum over the normal
# form of g^{μ(w)} g_0 otherwise ([AKN, §7.1, Step 5, p. 56]).
function _theta_chain_step(
    env::GEnvelope, rj::Vector{WSeries}, j::Integer, L::Int, threaded::Bool
)
    p = env.R.p
    n = L - j - 1                                       # the entries w < L - (j + 1)
    carries = [w for w in 0:(n - 1) if w % p == p - 1]
    sums = tmap(w -> _theta_chain_carry(env, rj, w, L), carries; threaded=threaded)
    out = Vector{WSeries}(undef, n)
    for w in 0:(n - 1)
        w % p == p - 1 || (out[w + 1] = rj[w + 2])
    end
    for (k, w) in enumerate(carries)
        out[w + 1] = sums[k]
    end
    return out
end

# Σ_{w'} G_0[w][w'] r_j[w'] modulo z_0^{L-1}, for w whose lowest base-p digit is p - 1:
# G_0[w][w'] is the g^{μ(w')}-coefficient of the normal form of g^{μ(w)} g_0 (known modulo
# z_0^{L-w'}, which suffices since r_j[w'] is divisible by z_0^{w'-1}).
function _theta_chain_carry(env::GEnvelope, rj::Vector{WSeries}, w::Integer, L::Int)
    R = env.R
    zp = L - 1
    e = g_digits(env, w)
    e[1] += 1
    nf = g_monomial_normal_form(env, e)
    prec = nf.prec
    c = with_scratch(env.scratch) do ws
        acc = scratch_acc!(ws, zp)
        for w2 in 0:(length(rj) - 1)
            s = rj[w2 + 1]
            v = nf.vals[w2 + 1]
            v + s.zval >= zp && continue
            acc_addmul!(acc, nf.sh[w2 + 1], v, s.c, s.zval, ws)
            prec = min(prec, s.prec)
        end
        acc_finish!(acc)
    end
    return _ws_from_wpoly(R, c, zp, prec)
end

# The series given by the W-polynomial A (modulo z^zp, which A is reduced modulo), stored with its
# z-valuation removed.
function _ws_from_wpoly(
    R::GaloisRing, A::Vector{ZZModPolyRingElem}, zp::Integer, prec::Integer
)
    v = wpoly_valuation(A)
    v >= zp && return ws_zero(R, zp; prec=prec)
    return WSeries(R, wpoly_shift_right(A, v), v, zp, prec)
end

"""
    nabla_theta_table(env, chain::ThetaChain, vi; L=g_length(vi), threaded=true)
        -> Vector{WSeries}

`θ(w(v^i) g_0^j) = Σ_w c_w r_j[w]` for `0 ≤ j ≤ L-1`, modulo `z_0^{L-1}`, where
`vi = w(v^i) = Σ_w c_w g^{μ(w)}` is known modulo `F^{≥L}` and `L ≤ chain.L`, from the transposed
chain ([AKN, §7.1, Step 5, p. 56]); the same values as the forward chain (the recorded
precision may be higher, since fewer normal forms are reduced). Parallel over `j` when `threaded`.
"""
function nabla_theta_table(
    env::GEnvelope,
    chain::ThetaChain,
    vi::GElem;
    L::Integer=g_length(vi),
    threaded::Bool=true,
)
    L = _g_check_length(L, min(g_length(vi), chain.L))
    return tmap(j -> _theta_from_chain(env, chain, vi, j, L), 0:(L - 1); threaded=threaded)
end

# θ(vi g_0^j) = Σ_{w < L-j} c_w r_j[w] modulo z_0^{L-1} (c_w is known modulo z_0^{L-w} and r_j[w]
# is divisible by z_0^{w+j-1}).
function _theta_from_chain(env::GEnvelope, chain::ThetaChain, vi::GElem, j::Integer, L::Int)
    R = env.R
    zp = L - 1
    rj = chain.r[j + 1]
    prec = vi.prec
    c = with_scratch(env.scratch) do ws
        acc = scratch_acc!(ws, zp)
        for w in 0:(min(length(rj), L - j) - 1)
            s = rj[w + 1]
            v = vi.vals[w + 1]
            v + s.zval >= zp && continue
            acc_addmul!(acc, vi.sh[w + 1], v, s.c, s.zval, ws)
            prec = min(prec, s.prec)
        end
        acc_finish!(acc)
    end
    return WSeries(R, c, 0, zp, prec)
end

"""
    nabla_OK_column(env, thetas, k; L=env.L) -> WSeries

`θ(w(v^i)(z_0 + g_0)^k - z_0^k) = Σ_{j=0}^{k} C(k,j) z_0^{k-j} θ(w(v^i) g_0^j)`, modulo
`z_0^{L-1}` ([AKN, §7.1, Step 5, p. 56]).
"""
function nabla_OK_column(
    env::GEnvelope, thetas::Vector{WSeries}, k::Integer; L::Integer=env.L
)
    R = env.R
    js = [j for j in 0:k if k - j < L - 1]
    # the sum is known modulo z_0^zp and p^prec
    zp = minimum((thetas[j + 1].zprec + k - j for j in js); init=L - 1)
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
    nabla_OK_matrix(env, vi; threaded=true, L=g_length(vi), chain=nothing, method=:transposed)
        -> WMatrix

The matrix of `∇: F^{[1,in-1]}Δ^(1)_{O_K/A_0}{i} → F^{[1,in-1]}Δ^(1),∇_{O_K/A_0}{i}` in the bases
`z_0^k w(s^i)` (`1 ≤ k ≤ L-1`, columns) and `z_0^m w(s^{i-1})` (`0 ≤ m ≤ L-2`, rows), where
`vi = w(v^i)` is known modulo `F^{≥L}` ([AKN, §7.1, Step 5, p. 56]). The coefficients
`θ(w(v^i) g_0^j)` come from the transposed chain: `chain` (a `ThetaChain` modulo `F^{≥L'}`,
`L' ≥ L`, e.g. shared by a sweep) if one is given, a new one modulo `F^{≥L}` for
`method = :transposed`; or from the forward chain for `method = :forward`. The entries are the
same; the recorded precision may be higher with the transposed chain.
"""
function nabla_OK_matrix(
    env::GEnvelope,
    vi::GElem;
    threaded::Bool=true,
    L::Integer=g_length(vi),
    chain::Union{Nothing,ThetaChain}=nothing,
    method::Symbol=:transposed,
)
    L = _g_check_length(L, g_length(vi))
    method in (:transposed, :forward) ||
        throw(ArgumentError("unknown method :$method (use :transposed or :forward)"))
    L <= 1 && return WMatrix(env.R, Matrix{WElem}(undef, 0, 0), vi.prec)   # F^{[1,0]} = 0
    thetas = if chain === nothing && method === :forward
        nabla_theta_table(env, vi; L=L)
    else
        ch = something(chain, theta_chain(env; L=L, threaded=threaded))
        nabla_theta_table(env, ch, vi; L=L, threaded=threaded)
    end
    cols = tmap(k -> nabla_OK_column(env, thetas, k; L=L), 1:(L - 1); threaded=threaded)
    A = Matrix{WElem}(undef, max(L - 1, 0), max(L - 1, 0))
    for k in 1:(L - 1), m in 0:(L - 2)
        A[m + 1, k] = ws_coefficient(cols[k], m)
    end
    prec = minimum((c.prec for c in cols); init=vi.prec)
    return WMatrix(env.R, A, prec)
end
