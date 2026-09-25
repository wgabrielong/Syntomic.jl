# Elements of the prismatic envelope of O_K over A_1 = W(F_q)[[z_0, z_1]] ("the g-envelope") and
# their normal forms.
#
# [AKN, §7.1, Step 3, p. 55]: for Δ^(1)_{O_K/W(F_q)[[z_0,z_1]]} "we may work in W(F_q)[[z_0]][g_u | 0 ≤ u ≤
# M_g] instead of the bigger W(F_q)[[z_0,z_1]][g_u | 0 ≤ u ≤ M_g]", using z_1 = z_0 + g_0 (the image of
# the relation τ g_0 = z_1 - z_0, [AKN, Cor. 3.31, p. 23]). A W(F_q)[[z_0]]-basis is given by the
# monomials ∏ g_u^{e_u} with e_u < p ([AKN, Prop. 3.34(1), p. 24]); g_u has F-weight p^u and z_0 has
# F-weight 1, so each basis monomial g^{μ(t)}, μ(t) = base-p digits of t, has F-weight t.
#
# An element truncated at F^{≥L} is Σ_{t<L} c_t(z_0) g^{μ(t)} with c_t known modulo z_0^{L-t}. All
# products below are truncated using the F-filtration: the normal form of a monomial of F-weight W
# has its g^{μ(t)}-component divisible by z_0^{W-t} ([AKN, Prop. 3.34(1)]).
#
# Since F^{≥L'} is an ideal, the operations below also work modulo F^{≥L'} for any L' ≤ L (the
# keyword `L`): an element known modulo F^{≥L'} is stored with L' components, the operands may be
# known to a higher F-precision, and the normal forms (computed once, modulo F^{≥L}) are truncated
# when they are used.

"""
    GElem

An element `Σ_{t<L} c_t(z_0) g^{μ(t)}` of `F^{[0,L-1]}Δ^(1)_{O_K/A_1}`, where `L = length(c)` (the
F-truncation, at most the `L` of the envelope): `c[t+1]` is the W-polynomial (vector of `f`
polynomials in `z_0`, see [`wpoly_mullow`](@ref)) `c_t`, known modulo `z_0^{L-t}` and modulo
`p^prec`. For fast multiplication, `vals[t+1]` is the `z_0`-adic valuation of `c_t` and
`sh[t+1] = c_t / z_0^{vals[t+1]}` (the normal form of a monomial of F-weight `W` has
`vals[t+1] ≥ W - t`).
"""
struct GElem
    c::Vector{Vector{ZZModPolyRingElem}}
    prec::Int
    vals::Vector{Int}
    sh::Vector{Vector{ZZModPolyRingElem}}
end

function GElem(c::Vector{Vector{ZZModPolyRingElem}}, prec::Integer)
    vals = [wpoly_valuation(ct) for ct in c]
    sh = [wpoly_shift_right(c[t], vals[t]) for t in eachindex(c)]
    return GElem(c, Int(prec), vals, sh)
end

"""
    GEnvelope

The g-envelope `Δ^(1)_{O_K/A_1}` truncated at `F^{≥L}` ([AKN, §7.1, Step 3, p. 55]) with the data of
[AKN, Lemma 3.24, p. 21], [AKN, Lemma 3.26, p. 21] and [AKN, Construction 3.28, p. 22]:

- `Rp[u+1] = R'_u` (`0 ≤ u ≤ M_g - 1`);
- `rel[u+1]`: the normal form of `g_u^p`, `(-p + λ_u E^{p^{u+1}}) g_{u+1} + E^{p^{u+1}} R'_u`;
- `phi[u+1] = φ(g_u) = φ(E)^{p^u}(λ_u g_{u+1} + R'_u)`;
- thread-safe memo tables ([`OnceCache`](@ref)) of normal forms of non-reduced monomials and of
  `φ(g^{μ(t)})`;
- `digits[t+1] = μ(t)`, and `threaded` (whether products are computed in parallel).

Build it with [`GEnvelope(prm, R, ed, lam)`](@ref GEnvelope(::SyntomicParams, ::GaloisRing, ::EisensteinData, ::LambdaData)).
"""
struct GEnvelope
    prm::SyntomicParams
    L::Int
    ng::Int
    R::GaloisRing
    ed::EisensteinData
    lam::LambdaData
    Rp::Vector{GElem}
    rel::Vector{GElem}
    phi::Vector{GElem}
    nf_cache::OnceCache{Vector{Int},GElem}
    phimono_cache::OnceCache{Int,GElem}
    digits::Vector{Vector{Int}}
    threaded::Bool
end

# ---------------------------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------------------------

"""
    g_length(x) -> Int

The F-truncation of `x`: the element is known modulo `F^{≥g_length(x)}`.
"""
g_length(x::GElem) = length(x.c)

# The F-truncation `L` requested for an operation on operands known modulo F^{≥Lx}: 0 ≤ L ≤ Lx.
function _g_check_length(L::Integer, Lx::Integer)
    0 <= L <= Lx || throw(
        ArgumentError(
            "F-truncation $L: the operands are only known modulo F^{≥$Lx} (need 0 ≤ L ≤ $Lx)",
        ),
    )
    return Int(L)
end

"""
    g_weight_valuation(x) -> Int

The F-weight valuation of `x`: the largest `w ≤ g_length(x)` with `x ∈ F^{≥w}` modulo
`F^{≥g_length(x)}`, i.e. the minimum of `t + v_{z_0}(c_t)` (the basis monomial `g^{μ(t)}` has
F-weight `t` and `z_0` has F-weight 1).
"""
function g_weight_valuation(x::GElem)
    L = g_length(x)
    return minimum((t + min(x.vals[t + 1], L - t) for t in 0:(L - 1)); init=L)
end

"""
    g_zero(env; L=env.L) -> GElem

Zero, modulo `F^{≥L}`.
"""
g_zero(env::GEnvelope; L::Integer=env.L) = GElem([wpoly_zero(env.R) for _ in 1:L], env.R.N)

"""
    g_truncate(env, x, L) -> GElem

`x` modulo `F^{≥L}` for `L ≤ g_length(x)` (`x` itself if `L = g_length(x)`): the components `c_t`,
`t < L`, modulo `z_0^{L-t}`.
"""
function g_truncate(env::GEnvelope, x::GElem, L::Integer)
    _g_check_length(L, g_length(x))
    L == g_length(x) && return x
    return GElem([_g_truncated(x.c[t + 1], L - t) for t in 0:(L - 1)], x.prec)
end

# The W-polynomial A modulo z^len (A itself if it is already reduced; the result is not to be
# modified in place).
function _g_truncated(A::Vector{ZZModPolyRingElem}, len::Integer)
    all(a -> length(a) <= len, A) && return A
    return [truncate(a, max(Int(len), 0)) for a in A]
end

"""
    g_digits(env, t) -> Vector{Int}

The exponent vector `μ(t)` (base-`p` digits) of the basis monomial of F-weight `t`.
"""
g_digits(env::GEnvelope, t::Integer) = copy(env.digits[t + 1])

"""
    g_exponent_weight(env, e) -> Int

The F-weight `Σ_u p^u e_u` of `∏ g_u^{e_u}`.
"""
g_exponent_weight(env::GEnvelope, e::AbstractVector{<:Integer}) = g_weight(env.R.p, e)

_is_reduced(p::Integer, e::AbstractVector{<:Integer}) = all(<(p), e)

"""
    g_from_series(env, s::WSeries; L=env.L) -> GElem

The element `s(z_0)` (constant in the `g_u`), modulo `F^{≥L}`.
"""
function g_from_series(env::GEnvelope, s::WSeries; L::Integer=env.L)
    c = [wpoly_zero(env.R) for _ in 1:L]
    L >= 1 && (c[1] = _aligned(s, 0, L))
    return GElem(c, s.prec)
end

"""
    g_monomial(env, t; coefficient=nothing, L=env.L) -> GElem

The basis monomial `g^{μ(t)}` of F-weight `t`, optionally times a series coefficient, modulo
`F^{≥L}`.
"""
function g_monomial(
    env::GEnvelope,
    t::Integer;
    coefficient::Union{Nothing,WSeries}=nothing,
    L::Integer=env.L,
)
    t >= L && return g_zero(env; L=L)
    c = [wpoly_zero(env.R) for _ in 1:L]
    if coefficient === nothing
        c[t + 1] = [d == 1 ? one(env.R.Znz) : zero(env.R.Znz) for d in 1:(env.R.f)]
        return GElem(c, env.R.N)
    end
    c[t + 1] = _aligned(coefficient, 0, L - t)
    return GElem(c, coefficient.prec)
end

"""
    g_add(env, x, y; L=min(g_length(x), g_length(y))) -> GElem

`x + y` modulo `F^{≥L}`.
"""
function g_add(env::GEnvelope, x::GElem, y::GElem; L::Integer=min(g_length(x), g_length(y)))
    return _g_combine(.+, x, y, L)
end

"""
    g_sub(env, x, y; L=min(g_length(x), g_length(y))) -> GElem

`x - y` modulo `F^{≥L}`.
"""
function g_sub(env::GEnvelope, x::GElem, y::GElem; L::Integer=min(g_length(x), g_length(y)))
    return _g_combine(.-, x, y, L)
end

# op.(x, y) componentwise, modulo F^{≥L}.
function _g_combine(op, x::GElem, y::GElem, L::Integer)
    _g_check_length(L, min(g_length(x), g_length(y)))
    if L == g_length(x) == g_length(y)
        return GElem([op(x.c[t], y.c[t]) for t in 1:L], min(x.prec, y.prec))
    end
    c = [
        op(_g_truncated(x.c[t + 1], L - t), _g_truncated(y.c[t + 1], L - t)) for
        t in 0:(L - 1)
    ]
    return GElem(c, min(x.prec, y.prec))
end

"""
    g_scale_int(env, x, k) -> GElem

`k·x` for an integer `k`.
"""
function g_scale_int(env::GEnvelope, x::GElem, k::IntegerLike)
    kk = env.R.Zn(k)
    return GElem([[kk * a for a in c] for c in x.c], x.prec)
end

"""
    g_scale_series(env, s::WSeries, x; L=g_length(x)) -> GElem

`s(z_0)·x` modulo `F^{≥L}`, for a power series `s` known modulo `z_0^L`.
"""
function g_scale_series(env::GEnvelope, s::WSeries, x::GElem; L::Integer=g_length(x))
    _g_check_length(L, g_length(x))
    S = _aligned(s, 0, L)
    ws = WScratch(env.R)
    out = [wpoly_mullow(env.R, S, x.c[t + 1], L - t; ws=ws) for t in 0:(L - 1)]
    return GElem(out, min(s.prec, x.prec))
end

"""
    g_component_series(env, x, t; L=g_length(x)) -> WSeries

The coefficient `c_t(z_0)` of `g^{μ(t)}` modulo `F^{≥L}`, as a series known modulo `z_0^{L-t}`.
"""
function g_component_series(env::GEnvelope, x::GElem, t::Integer; L::Integer=g_length(x))
    _g_check_length(L, g_length(x))
    t < L || return ws_zero(env.R, 0; prec=x.prec)
    return WSeries(env.R, _g_truncated(x.c[t + 1], L - t), 0, L - t, x.prec)
end

"""
    g_theta(env, x; L=g_length(x)) -> WSeries

`θ(x)`: the coefficient of `g_0` with respect to the `W(F_q)[[z_0]]`-basis `∏ g_u^{e_u}`, a series
known modulo `z_0^{L-1}` for `x` modulo `F^{≥L}` ([AKN, §7.1, Step 5, p. 56]; [AKN, Def. 4.41,
p. 37]).
"""
g_theta(env::GEnvelope, x::GElem; L::Integer=g_length(x)) =
    g_component_series(env, x, 1; L=L)

"""
    g_isequal(env, x, y; prec=min(x.prec, y.prec), L=min(g_length(x), g_length(y))) -> Bool

Whether `x ≡ y` modulo `p^prec` and `F^{≥L}`.
"""
function g_isequal(
    env::GEnvelope,
    x::GElem,
    y::GElem;
    prec::Integer=min(x.prec, y.prec),
    L::Integer=min(g_length(x), g_length(y)),
)
    d = g_sub(env, x, y; L=L)
    for t in 0:(L - 1)
        ws_iszero(WSeries(env.R, d.c[t + 1], 0, L - t, prec); prec=prec) || return false
    end
    return true
end

# out[t'] += (s · y)_{t'} for a W-polynomial s whose product with y is known modulo z_0^{L-t'} in
# every component, where L = length(out) ≤ g_length(y) is the F-truncation of the result
# (F-filtration bookkeeping is the caller's responsibility). The polynomials of `out` are owned by
# the caller and updated in place; `ws` is scratch space of the calling task.
function _g_addmul_series!(
    env::GEnvelope,
    out,
    s::Vector{ZZModPolyRingElem},
    y::GElem,
    ws::WScratch=WScratch(env.R),
)
    L = length(out)
    vs = wpoly_valuation(s)
    vs >= L && return out
    ssh = wpoly_shift_right(s, vs)
    for t in 0:(L - 1)
        vy = y.vals[t + 1]
        vs + vy >= L - t && continue
        wpoly_addmul!(env.R, out[t + 1], ssh, vs, y.sh[t + 1], vy, L - t, ws)
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Normal forms ([AKN, §7.1, reduce subalgorithm, p. 55])
# ---------------------------------------------------------------------------------------------

"""
    g_monomial_normal_form(env, e) -> GElem

The normal form of `∏ g_u^{e_u}` (arbitrary exponents) modulo `F^{≥L}`, obtained by repeatedly
rewriting `g_u^p` with the relation of [AKN, Lemma 3.24, p. 21] (after `d̃ ↦ E(z_0)`):
`g_u^p = (-p + λ_u E^{p^{u+1}}) g_{u+1} + E^{p^{u+1}} R'_u`. Memoized.
"""
function g_monomial_normal_form(env::GEnvelope, e::AbstractVector{<:Integer})
    p = env.R.p
    W = g_exponent_weight(env, e)
    W >= env.L && return g_zero(env)
    _is_reduced(p, e) && return g_monomial(env, W)
    key = Vector{Int}(e)
    return get_or_compute!(env.nf_cache, key) do
        _g_monomial_normal_form_uncached(env, key, W)
    end
end

function _g_monomial_normal_form_uncached(env::GEnvelope, e::Vector{Int}, W::Int)
    p = env.R.p
    u = findfirst(>=(p), e)                  # 1-based: rewrite g_{u-1}^p
    u <= length(env.rel) ||
        error("g_monomial_normal_form: relation for g_$(u - 1)^p is not available")
    rel = env.rel[u]
    e1 = copy(e)
    e1[u] -= p
    W1 = W - p^u                              # F-weight of g^{e1}
    out = [wpoly_zero(env.R) for _ in 1:(env.L)]
    ws = WScratch(env.R)
    prec = rel.prec
    for t in 0:(env.L - 1 - W1)
        rc = rel.c[t + 1]
        all(iszero, rc) && continue
        sub = g_monomial_normal_form(env, e1 .+ g_digits(env, t))
        prec = min(prec, sub.prec)
        _g_addmul_series!(env, out, rc, sub, ws)
    end
    return GElem(out, prec)
end

# Products c_{t1}·d_{t2} modulo F^{≥L} for the rows t1 in `t1s`: carry-free products are added to
# `out`, the others are accumulated per exponent vector in `carries` (both new, owned by the
# caller). For a square (x === y) only the pairs t1 ≤ t2 are formed, the off-diagonal ones with a
# factor 2.
function _g_mul_rows(env::GEnvelope, x::GElem, y::GElem, t1s, square::Bool, L::Int)
    R = env.R
    p = R.p
    D = env.digits
    out = [wpoly_zero(R) for _ in 1:L]
    carries = Dict{Vector{Int},Vector{ZZModPolyRingElem}}()
    ws = WScratch(R)
    for t1 in t1s
        vx = x.vals[t1 + 1]
        xs = x.sh[t1 + 1]
        for t2 in (square ? t1 : 0):(L - 1 - t1)
            vy = y.vals[t2 + 1]
            prod = wpoly_product!(R, ws, xs, vx, y.sh[t2 + 1], vy, L - t1 - t2)
            prod === nothing && continue
            square && t2 != t1 && wpoly_double!(prod, R.f)
            e = D[t1 + 1] .+ D[t2 + 1]
            acc =
                _is_reduced(p, e) ? out[t1 + t2 + 1] : get!(() -> wpoly_zero(R), carries, e)
            wpoly_add!(acc, prod)
        end
    end
    return out, carries
end

# out ← out + other (componentwise, in place; the polynomials of `out` are owned by the caller)
function _g_accumulate!(out, other)
    for t in eachindex(out)
        wpoly_add!(out[t], other[t])
    end
    return out
end

# The number of tasks to use for work on `m` independent pieces.
function _g_ntasks(env::GEnvelope, m::Integer)
    return (env.threaded && threads_available() && m >= 8) ? min(Threads.nthreads(), m) : 1
end

"""
    g_mul(env, x, y; L=min(g_length(x), g_length(y))) -> GElem

The product in `F^{[0,L-1]}Δ^(1)_{O_K/A_1}`, i.e. modulo the ideal `F^{≥L}`, in the basis of
[AKN, §7.1, Step 2, p. 54] (the reduce subalgorithm, [AKN, §7.1, p. 55]). Products of basis
monomials whose base-`p` digits add without carry are basis monomials; the others are rewritten
with [`g_monomial_normal_form`](@ref) (computed modulo `F^{≥env.L}` and truncated). Terms with the
same exponent vector are accumulated before the normal form is applied. When `env.threaded`, the
rows of the product and the normal forms are distributed over tasks with separate accumulators
(the result does not depend on the number of tasks).
"""
function g_mul(env::GEnvelope, x::GElem, y::GElem; L::Integer=min(g_length(x), g_length(y)))
    L = _g_check_length(L, min(g_length(x), g_length(y)))
    square = x === y
    rows = [t1 for t1 in 0:(L - 1) if x.vals[t1 + 1] < L - t1]
    out, carries = _g_mul_products(env, x, y, rows, square, L)
    ks = collect(keys(carries))
    nfs = tmap(e -> g_monomial_normal_form(env, e), ks; threaded=env.threaded)
    prec = minimum((nfe.prec for nfe in nfs); init=min(x.prec, y.prec))
    _g_apply_normal_forms!(env, out, [carries[e] for e in ks], nfs)
    return GElem(out, prec)
end

# The products of the rows of x with y modulo F^{≥L}, distributed over tasks with separate
# accumulators.
function _g_mul_products(
    env::GEnvelope, x::GElem, y::GElem, rows::Vector{Int}, square::Bool, L::Int
)
    ntasks = _g_ntasks(env, length(rows))
    ntasks == 1 && return _g_mul_rows(env, x, y, rows, square, L)
    parts = [rows[k:ntasks:end] for k in 1:ntasks]
    results = tmap(part -> _g_mul_rows(env, x, y, part, square, L), parts)
    out, carries = results[1]
    for (out2, carries2) in results[2:end]
        _g_accumulate!(out, out2)
        for (e, s) in carries2
            acc = get(carries, e, nothing)
            acc === nothing ? (carries[e] = s) : wpoly_add!(acc, s)
        end
    end
    return out, carries
end

# out += Σ_j coefficients[j] · normal_forms[j] modulo F^{≥length(out)}, distributed over tasks.
function _g_apply_normal_forms!(env::GEnvelope, out, coefficients, normal_forms)
    L = length(out)
    m = length(coefficients)
    ntasks = _g_ntasks(env, m)
    if ntasks == 1
        ws = WScratch(env.R)
        for j in 1:m
            _g_addmul_series!(env, out, coefficients[j], normal_forms[j], ws)
        end
        return out
    end
    partial = tmap(1:ntasks) do k
        local acc = [wpoly_zero(env.R) for _ in 1:L]
        local ws = WScratch(env.R)
        for j in k:ntasks:m
            _g_addmul_series!(env, acc, coefficients[j], normal_forms[j], ws)
        end
        acc
    end
    for acc in partial
        _g_accumulate!(out, acc)
    end
    return out
end

"""
    g_mul_g0(env, x; L=g_length(x)) -> GElem

`g_0 · x` modulo `F^{≥L}` (used to form `(z_0 + g_0)^k` in Step 5).
"""
function g_mul_g0(env::GEnvelope, x::GElem; L::Integer=g_length(x))
    L = _g_check_length(L, g_length(x))
    p = env.R.p
    out = [wpoly_zero(env.R) for _ in 1:L]
    ws = WScratch(env.R)
    prec = x.prec
    for t in 0:(L - 2)
        xc = x.c[t + 1]
        all(iszero, xc) && continue
        if t % p != p - 1
            # out[t+2] only receives terms known modulo z_0^{L-t-1}, so truncating is exact
            o = out[t + 2]
            for d in eachindex(o)
                o[d] = poly_add_trunc!(o[d], o[d], xc[d], L - t - 1)
            end
        else
            e = g_digits(env, t)
            e[1] += 1
            nfe = g_monomial_normal_form(env, e)
            prec = min(prec, nfe.prec)
            _g_addmul_series!(env, out, xc, nfe, ws)
        end
    end
    return GElem(out, prec)
end

"""
    g_pow(env, x, k; L=g_length(x)) -> GElem

`x^k` modulo `F^{≥L}` for an integer `k ≥ 0` (binary powering).
"""
function g_pow(env::GEnvelope, x::GElem, k::Integer; L::Integer=g_length(x))
    k >= 0 || throw(ArgumentError("g_pow: negative exponent"))
    result = g_one(env; L=L)
    result = GElem(result.c, x.prec)
    base = g_truncate(env, x, L)
    m = k
    first = true
    while m > 0
        if isodd(m)
            result = first ? base : g_mul(env, result, base)
            first = false
        end
        m >>= 1
        m > 0 && (base = g_mul(env, base, base))
    end
    return result
end

"""
    g_one(env; L=env.L) -> GElem

One, modulo `F^{≥L}`.
"""
g_one(env::GEnvelope; L::Integer=env.L) = g_monomial(env, 0; L=L)
