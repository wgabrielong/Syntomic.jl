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

"""
    GElem

An element `Σ_{t<L} c_t(z_0) g^{μ(t)}` of `F^{[0,L-1]}Δ^(1)_{O_K/A_1}`: `c[t+1]` is the W-polynomial
(vector of `f` polynomials in `z_0`, see [`wpoly_mullow`](@ref)) `c_t`, known modulo `z_0^{L-t}` and
modulo `p^prec`. For fast multiplication, `vals[t+1]` is the `z_0`-adic valuation of `c_t` and
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
    g_zero(env) -> GElem
"""
g_zero(env::GEnvelope) = GElem([wpoly_zero(env.R) for _ in 1:(env.L)], env.R.N)

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
    g_from_series(env, s::WSeries) -> GElem

The element `s(z_0)` (constant in the `g_u`).
"""
function g_from_series(env::GEnvelope, s::WSeries)
    c = [wpoly_zero(env.R) for _ in 1:(env.L)]
    c[1] = _aligned(s, 0, env.L)
    return GElem(c, s.prec)
end

"""
    g_monomial(env, t; coefficient=nothing) -> GElem

The basis monomial `g^{μ(t)}` of F-weight `t`, optionally times a series coefficient.
"""
function g_monomial(env::GEnvelope, t::Integer; coefficient::Union{Nothing,WSeries}=nothing)
    t >= env.L && return g_zero(env)
    c = [wpoly_zero(env.R) for _ in 1:(env.L)]
    if coefficient === nothing
        c[t + 1] = [d == 1 ? one(env.R.Znz) : zero(env.R.Znz) for d in 1:(env.R.f)]
        return GElem(c, env.R.N)
    end
    c[t + 1] = _aligned(coefficient, 0, env.L - t)
    return GElem(c, coefficient.prec)
end

"""
    g_add(env, x, y) -> GElem
"""
function g_add(env::GEnvelope, x::GElem, y::GElem)
    return GElem([x.c[t] .+ y.c[t] for t in 1:(env.L)], min(x.prec, y.prec))
end

"""
    g_sub(env, x, y) -> GElem
"""
function g_sub(env::GEnvelope, x::GElem, y::GElem)
    return GElem([x.c[t] .- y.c[t] for t in 1:(env.L)], min(x.prec, y.prec))
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
    g_scale_series(env, s::WSeries, x) -> GElem

`s(z_0)·x` for a power series `s` known modulo `z_0^L`.
"""
function g_scale_series(env::GEnvelope, s::WSeries, x::GElem)
    S = _aligned(s, 0, env.L)
    out = [wpoly_mullow(env.R, S, x.c[t + 1], env.L - t) for t in 0:(env.L - 1)]
    return GElem(out, min(s.prec, x.prec))
end

"""
    g_component(x, t) -> Vector{ZZModPolyRingElem}

The coefficient `c_t(z_0)` of `g^{μ(t)}` (as a W-polynomial).
"""
g_component(x::GElem, t::Integer) = x.c[t + 1]

"""
    g_component_series(env, x, t) -> WSeries

The coefficient `c_t(z_0)` of `g^{μ(t)}`, as a series known modulo `z_0^{L-t}`.
"""
g_component_series(env::GEnvelope, x::GElem, t::Integer) =
    WSeries(env.R, x.c[t + 1], 0, env.L - t, x.prec)

"""
    g_theta(env, x) -> WSeries

`θ(x)`: the coefficient of `g_0` with respect to the `W(F_q)[[z_0]]`-basis `∏ g_u^{e_u}`, a series
known modulo `z_0^{L-1}` ([AKN, §7.1, Step 5, p. 56]; [AKN, Def. 4.41, p. 37]).
"""
g_theta(env::GEnvelope, x::GElem) = g_component_series(env, x, 1)

"""
    g_isequal(env, x, y; prec=min(x.prec, y.prec)) -> Bool

Whether `x ≡ y` modulo `p^prec` and `F^{≥L}`.
"""
function g_isequal(env::GEnvelope, x::GElem, y::GElem; prec::Integer=min(x.prec, y.prec))
    for t in 0:(env.L - 1)
        ws_iszero(
            WSeries(env.R, x.c[t + 1] .- y.c[t + 1], 0, env.L - t, prec); prec=prec
        ) || return false
    end
    return true
end

# out[t'] += (s · y)_{t'} for a W-polynomial s whose product with y is known modulo z_0^{L-t'} in
# every component (F-filtration bookkeeping is the caller's responsibility).
function _g_addmul_series!(env::GEnvelope, out, s::Vector{ZZModPolyRingElem}, y::GElem)
    L = env.L
    vs = wpoly_valuation(s)
    vs >= L && return out
    ssh = wpoly_shift_right(s, vs)
    for t in 0:(L - 1)
        vy = y.vals[t + 1]
        vs + vy >= L - t && continue
        prod = wpoly_mullow_shifted(env.R, ssh, vs, y.sh[t + 1], vy, L - t)
        out[t + 1] = out[t + 1] .+ prod
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
    prec = rel.prec
    for t in 0:(env.L - 1 - W1)
        rc = rel.c[t + 1]
        all(iszero, rc) && continue
        sub = g_monomial_normal_form(env, e1 .+ g_digits(env, t))
        prec = min(prec, sub.prec)
        _g_addmul_series!(env, out, rc, sub)
    end
    return GElem(out, prec)
end

# Products c_{t1}·d_{t2} for the rows t1 in `t1s`: carry-free products are added to `out`,
# the others are accumulated per exponent vector in `carries`. For a square (x === y) only the
# pairs t1 ≤ t2 are formed, the off-diagonal ones with a factor 2.
function _g_mul_rows(env::GEnvelope, x::GElem, y::GElem, t1s, square::Bool)
    L = env.L
    p = env.R.p
    D = env.digits
    two = env.R.Zn(2)
    out = [wpoly_zero(env.R) for _ in 1:L]
    carries = Dict{Vector{Int},Vector{ZZModPolyRingElem}}()
    for t1 in t1s
        vx = x.vals[t1 + 1]
        xs = x.sh[t1 + 1]
        for t2 in (square ? t1 : 0):(L - 1 - t1)
            vy = y.vals[t2 + 1]
            vx + vy >= L - t1 - t2 && continue
            prod = wpoly_mullow_shifted(env.R, xs, vx, y.sh[t2 + 1], vy, L - t1 - t2)
            square && t2 != t1 && (prod = [two * c for c in prod])
            e = D[t1 + 1] .+ D[t2 + 1]
            if _is_reduced(p, e)
                out[t1 + t2 + 1] = out[t1 + t2 + 1] .+ prod
            else
                acc = get(carries, e, nothing)
                carries[e] = acc === nothing ? prod : acc .+ prod
            end
        end
    end
    return out, carries
end

# out ← out + other (componentwise)
function _g_accumulate!(out, other)
    for t in eachindex(out)
        out[t] = out[t] .+ other[t]
    end
    return out
end

# The number of tasks to use for work on `m` independent pieces.
function _g_ntasks(env::GEnvelope, m::Integer)
    return (env.threaded && threads_available() && m >= 8) ? min(Threads.nthreads(), m) : 1
end

"""
    g_mul(env, x, y) -> GElem

The product in `F^{[0,L-1]}Δ^(1)_{O_K/A_1}`. Products of basis monomials whose base-`p` digits add
without carry are basis monomials; the others are rewritten with [`g_monomial_normal_form`](@ref).
Terms with the same exponent vector are accumulated before the normal form is applied. When
`env.threaded`, the rows of the product and the normal forms are distributed over tasks with
separate accumulators (the result does not depend on the number of tasks).
"""
function g_mul(env::GEnvelope, x::GElem, y::GElem)
    L = env.L
    square = x === y
    rows = [t1 for t1 in 0:(L - 1) if x.vals[t1 + 1] < L - t1]
    out, carries = _g_mul_products(env, x, y, rows, square)
    ks = collect(keys(carries))
    nfs = tmap(e -> g_monomial_normal_form(env, e), ks; threaded=env.threaded)
    prec = minimum((nfe.prec for nfe in nfs); init=min(x.prec, y.prec))
    _g_apply_normal_forms!(env, out, [carries[e] for e in ks], nfs)
    return GElem(out, prec)
end

# The products of the rows of x with y, distributed over tasks with separate accumulators.
function _g_mul_products(
    env::GEnvelope, x::GElem, y::GElem, rows::Vector{Int}, square::Bool
)
    ntasks = _g_ntasks(env, length(rows))
    ntasks == 1 && return _g_mul_rows(env, x, y, rows, square)
    parts = [rows[k:ntasks:end] for k in 1:ntasks]
    results = tmap(part -> _g_mul_rows(env, x, y, part, square), parts)
    out, carries = results[1]
    for (out2, carries2) in results[2:end]
        _g_accumulate!(out, out2)
        for (e, s) in carries2
            acc = get(carries, e, nothing)
            carries[e] = acc === nothing ? s : acc .+ s
        end
    end
    return out, carries
end

# out += Σ_j coefficients[j] · normal_forms[j], distributed over tasks.
function _g_apply_normal_forms!(env::GEnvelope, out, coefficients, normal_forms)
    L = env.L
    m = length(coefficients)
    ntasks = _g_ntasks(env, m)
    if ntasks == 1
        for j in 1:m
            _g_addmul_series!(env, out, coefficients[j], normal_forms[j])
        end
        return out
    end
    partial = tmap(1:ntasks) do k
        local acc = [wpoly_zero(env.R) for _ in 1:L]
        for j in k:ntasks:m
            _g_addmul_series!(env, acc, coefficients[j], normal_forms[j])
        end
        acc
    end
    for acc in partial
        _g_accumulate!(out, acc)
    end
    return out
end

"""
    g_mul_g0(env, x) -> GElem

`g_0 · x` (used to form `(z_0 + g_0)^k` in Step 5).
"""
function g_mul_g0(env::GEnvelope, x::GElem)
    L = env.L
    p = env.R.p
    out = [wpoly_zero(env.R) for _ in 1:L]
    prec = x.prec
    for t in 0:(L - 2)
        xc = x.c[t + 1]
        all(iszero, xc) && continue
        if t % p != p - 1
            out[t + 2] = out[t + 2] .+ [truncate(a, L - t - 1) for a in xc]
        else
            e = g_digits(env, t)
            e[1] += 1
            nfe = g_monomial_normal_form(env, e)
            prec = min(prec, nfe.prec)
            _g_addmul_series!(env, out, xc, nfe)
        end
    end
    return GElem(out, prec)
end

"""
    g_pow(env, x, k) -> GElem

`x^k` for an integer `k ≥ 0` (binary powering).
"""
function g_pow(env::GEnvelope, x::GElem, k::Integer)
    k >= 0 || throw(ArgumentError("g_pow: negative exponent"))
    result = g_one(env)
    result = GElem(result.c, x.prec)
    base = x
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
    g_one(env) -> GElem
"""
g_one(env::GEnvelope) = g_monomial(env, 0)
