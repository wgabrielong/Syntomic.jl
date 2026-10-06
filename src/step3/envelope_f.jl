# The prismatic envelope of R = O_K/ϖ^n over the Breuil–Kisin prism A_0 = W(F_q)[[z_0]]
# ("the f-envelope"), truncated at F^{≥L}, L = n·i.
#
# [AKN24, §7.1, Step 3, p. 55]: "For Δ^(1)_{R/W(F_q)[[z_0]]}, find elements λ_u ∈ W(F_q)[[z_0]] as in
# Lemma 3.24 and create a lookup table of relations for f_u^p in the prismatic envelope in terms of
# those, noting that here the terms R'_u are all zero (Remark 3.29). … Also create a lookup table for
# the values of φ_{p^u}(f_u) following Lemma 3.26."
#
# With d = E(z_0) and r = z_0^n (so δ(r) = 0 and R'_u = 0, [AKN24, Remark 3.29, p. 22]):
#   * relations (after d̃ ↦ d):  z_0^n = f_0  and  f_u^p = c_u f_{u+1},  c_u = -p + λ_u E^{p^{u+1}}
#     ([AKN24, Lemma 3.24, p. 21; Cor. 3.31, p. 23]);
#   * divided Frobenius:  φ_{p^u}(f_u) = λ_u f_{u+1}  ([AKN24, Lemma 3.26(1), p. 21]);
#   * a W(F_q)-basis of F^{[0,L-1]}Δ^(1)_{R/A_0} is given by the monomials z_0^c ∏ f_u^{e_u} with
#     c < n, e_u < p and F-weight c + n Σ p^u e_u < L ([AKN24, Prop. 3.34(2), p. 24]).
# There is exactly one basis monomial b_w in each F-weight 0 ≤ w < L (see `f_basis_exponents`), so
# an element is a vector of L coefficients in W(F_q).
#
# The normal form ("reduce subalgorithm", [AKN24, §7.1, p. 55]) is computed level by level. Write
# B_s = ∏ f_u^{e_u} for s = Σ_u e_u p^u with digits e_u < p, so that b_w = z_0^c B_s for
# w = c + n·s. Then z_0^n B_s = f_0 B_s = C_{k(s)} B_{s+1} with C_k = c_0 c_1 ⋯ c_{k-1}, where k(s) is
# the number of trailing digits p - 1 of s (the carries of s + 1: f_0^p = c_0 f_1, then
# f_1^p = c_1 f_2, …). Hence, for a power series Q(z_0),
#   Q·B_s = Σ_{c<n} Q_c z_0^c B_s + (Q div z_0^n)·C_{k(s)}·B_{s+1},
# one truncated product of power series per level (none when k(s) = 0; terms of F-weight ≥ L are
# dropped). The result, modulo p^N, does not depend on the order in which the relations are
# applied, for any coefficients c_u with c_u(0) ≠ 0 over W(F_q): z_0 ↦ z, f_u ↦ z^{np^u}/Π_u with
# Π_0 = 1, Π_{u+1} = Π_u^p c_u maps the ring with these relations to Frac(W(F_q))((z)), sending
# F^{≥L} into z^L·Frac(W(F_q))[[z]] and b_w to z^w times a unit, so the b_w stay linearly
# independent modulo F^{≥L} (as in the envelope itself, [AKN24, Prop. 3.34(2), p. 24]).

"""
    FElem

An element `Σ_{w<L} a_w b_w` of `F^{[0,L-1]}Δ^(1)_{R/A_0}` (the f-envelope truncated at `F^{≥L}`),
known modulo `p^prec`. The coefficients are stored as a W-polynomial in an auxiliary variable `X`
(see [`wpoly_mullow`](@ref)): the coefficient of `X^w` in `v` is `a_w`, so that sums and scalar
multiples are computed by FLINT.
"""
struct FElem
    v::Vector{ZZModPolyRingElem}
    prec::Int
end

"""
    FEnvelope(prm::SyntomicParams, R::GaloisRing, ed, lam)

The f-envelope `Δ^(1)_{R/A_0}` of `R = O_K/ϖ^n` truncated at `F^{≥L}` ([AKN24, §7.1, Step 3, p. 55]),
with the relation coefficients `c_u = -p + λ_u E^{p^{u+1}}` ([AKN24, Lemma 3.24, p. 21]), their products
`C_k = c_0 ⋯ c_{k-1}` (the coefficient of `z_0^n B_s = C_{k(s)} B_{s+1}`, see
[`f_series_times_monomial`](@ref)), the series `P_j` with `z_0^{nj} = f_0^j = P_j B_j`, and a memo
table of the normal forms of the powers `z_0^k` (shared by `red` and `red^∇`, Step 6). The series
of `ed` and `lam` must be known modulo `z_0^L`.
"""
struct FEnvelope
    prm::SyntomicParams
    R::GaloisRing
    ed::EisensteinData
    lam::LambdaData
    c::Vector{WSeries}
    prec::Int
    carry::Vector{Vector{ZZModPolyRingElem}}      # C_k mod z_0^L, k = 1, …, n_f - 1
    zpre::Vector{Vector{ZZModPolyRingElem}}       # P_j mod z_0^{L - n·j}, j = 0, …, i - 1
    zpow::OnceCache{Int,FElem}
    scratch::ScratchPool
end

"""
    f_relation_coefficient(ed, lam, u) -> WSeries

`c_u = -p + λ_u E^{p^{u+1}}`, the coefficient in the relation `f_u^p = c_u f_{u+1}` in
`Δ^(1)_{R/A_0}` ([AKN24, Lemma 3.24, p. 21] with `R'_u = 0` by [AKN24, Remark 3.29, p. 22]).
"""
function f_relation_coefficient(ed::EisensteinData, lam::LambdaData, u::Integer)
    zp = ed.zprec
    t = ws_mul(lam.lambda[u + 1], lam.Epow[u + 2]; zprec=zp)
    return ws_sub(t, ws_const(ed.R, w_int(ed.R, ed.R.p), zp))
end

function FEnvelope(prm::SyntomicParams, R::GaloisRing, ed::EisensteinData, lam::LambdaData)
    L = prm.L
    c = [f_relation_coefficient(ed, lam, u) for u in 0:(prm.nf - 2)]
    prec = minimum((x.prec for x in c); init=ed.E.prec)
    prec = min(prec, minimum((x.prec for x in lam.lambda[1:max(prm.nf - 1, 0)]); init=prec))
    ws = WScratch(R)
    carry = Vector{Vector{ZZModPolyRingElem}}()
    for k in 1:length(c)
        ck = _f_series_polys(c[k], L)
        push!(carry, k == 1 ? ck : wpoly_mullow(R, carry[k - 1], ck, L; ws=ws))
    end
    # z_0^{n(j+1)} = z_0^n P_j B_j = P_j C_{k(j)} B_{j+1}
    zpre = [[d == 1 ? one(R.Znz) : zero(R.Znz) for d in 1:(R.f)]]
    for j in 1:(prm.i - 1)
        k = _f_carries(prm.p, j - 1)
        len = L - prm.n * j
        push!(zpre, k == 0 ? zpre[j] : wpoly_mullow(R, zpre[j], carry[k], len; ws=ws))
    end
    return FEnvelope(
        prm, R, ed, lam, c, prec, carry, zpre, OnceCache{Int,FElem}(), ScratchPool(R)
    )
end

# The coefficients of z_0^0, …, z_0^{len-1} of the series s as a W-polynomial (s must be known
# modulo z_0^len).
function _f_series_polys(s::WSeries, len::Integer)
    s.zprec >= len || throw(
        ArgumentError(
            "the f-envelope needs a series known modulo z^$len, not z^$(s.zprec)"
        ),
    )
    return _aligned(s, 0, len)
end

# The number k(s) of trailing digits p - 1 of s in base p: the carries of s + 1.
function _f_carries(p::Int, s::Int)
    k = 0
    while s % p == p - 1
        k += 1
        s ÷= p
    end
    return k
end

# ---------------------------------------------------------------------------------------------
# Elementary operations on FElem
# ---------------------------------------------------------------------------------------------

"""
    f_zero(env) -> FElem
"""
f_zero(env::FEnvelope) = FElem(wpoly_zero(env.R), env.R.N)

"""
    f_basis_element(env, w) -> FElem

The basis monomial `b_w = z_0^c ∏ f_u^{e_u}` of F-weight `w`
([AKN24, Prop. 3.34(2), p. 24]).
"""
function f_basis_element(env::FEnvelope, w::Integer)
    R = env.R
    v = [d == 1 ? shift_left(one(R.Znz), w) : zero(R.Znz) for d in 1:(R.f)]
    return FElem(v, R.N)
end

"""
    f_coefficient(x, w) -> WElem

The coefficient of the basis monomial of F-weight `w`.
"""
f_coefficient(x::FElem, w::Integer) = ZZRingElem[poly_coeff_lift(c, w) for c in x.v]

# ---------------------------------------------------------------------------------------------
# Normal forms (the reduce subalgorithm for R = O_K/ϖ^n)
# ---------------------------------------------------------------------------------------------

# The coordinates of the normal form of Q(z_0)·B_s ([AKN24, §7.1, reduce subalgorithm, p. 55]), as
# the W-polynomial in X of an FElem: at level s' = s, s + 1, …, the coefficients of z_0^c, c < n,
# of Q are those of b_{c+n·s'}, and Q ← (Q div z_0^n)·C_{k(s')} (see the comment at the top of
# the file). Q has length at most L - n·s; it is overwritten (the caller owns it). `ws` provides
# scratch space.
function _f_reduce!(env::FEnvelope, Q::Vector{ZZModPolyRingElem}, s::Int, ws::WScratch)
    R = env.R
    prm = env.prm
    n, L = prm.n, prm.L
    out = wpoly_zero(R)
    P = ws.prods
    while true
        len = L - n * s                          # Q is needed modulo z_0^len
        len <= 0 && break
        for d in 1:(R.f)
            out[d] = poly_set_block!(out[d], n * s, Q[d], min(n, len))
        end
        len <= n && break
        nonzero = false
        for d in 1:(R.f)
            Q[d] = poly_shift_right!(Q[d], n)
            nonzero |= !iszero(Q[d])
        end
        nonzero || break
        k = _f_carries(prm.p, s)
        if k > 0                                 # z_0^n B_s = C_k B_{s+1}
            wpoly_mul!(R, P, Q, env.carry[k], len - n, ws)
            for d in 1:(R.f)
                Q[d], P[d] = P[d], Q[d]
            end
        end
        s += 1
    end
    return out
end

"""
    f_monomial_weight(prm, a, e) -> Int

The F-weight `a + n Σ_u p^u e_u` of `z_0^a ∏ f_u^{e_u}` ([AKN24, Prop. 3.34(2), p. 24]).
"""
f_monomial_weight(prm::SyntomicParams, a::Integer, e::AbstractVector{<:Integer}) =
    f_weight(prm, a, e)

# The monomial z_0^a ∏ f_u^{e_u} of F-weight < L as z_0^{a'} ∏_u c_u^{m_u} · B_s: z_0^n = f_0,
# then f_u^p = c_u f_{u+1} ([AKN24, Lemma 3.24, p. 21]) bring the exponents to digits < p. Returns
# (s, a', m) with a' < n and m[u+1] = m_u, the number of times the relation for f_u^p is used.
function _f_carry_exponents(prm::SyntomicParams, a::Integer, e::AbstractVector{<:Integer})
    (a >= 0 && all(>=(0), e)) || throw(ArgumentError("negative exponent"))
    q, a2 = divrem(Int(a), prm.n)
    m = zeros(Int, prm.nf)
    s = 0
    for u in 0:(prm.nf - 1)
        q, digit = divrem(e[u + 1] + q, prm.p)
        m[u + 1] = q
        s += digit * prm.p^u
    end
    q == 0 || throw(ArgumentError("the monomial has F-weight at least L"))
    return s, a2, m
end

# The normal form of z_0^a ∏ f_u^{e_u} times the series Q (a W-polynomial of length at most the
# L - weight of the monomial, overwritten) at precision prec.
function _f_monomial_times!(
    env::FEnvelope, Q::Vector{ZZModPolyRingElem}, a::Integer, e, prec::Int, ws::WScratch
)
    R = env.R
    s, a2, m = _f_carry_exponents(env.prm, a, e)
    len = env.prm.L - env.prm.n * s
    for d in 1:(R.f)
        Q[d] = poly_shift_left!(Q[d], a2)
    end
    for u in 1:length(m), _ in 1:m[u]
        Q = wpoly_mullow(R, Q, _f_series_polys(env.c[u], len), len; ws=ws)
    end
    return FElem(_f_reduce!(env, Q, s, ws), prec)
end

"""
    f_monomial_normal_form(env, a, e) -> FElem

The normal form of the monomial `z_0^a ∏ f_u^{e_u}` (arbitrary exponents) in the basis of
[AKN24, Prop. 3.34(2), p. 24], modulo `F^{≥L}` (reduce subalgorithm, [AKN24, §7.1, p. 55]):
use `z_0^n = f_0` and `f_u^p = c_u f_{u+1}` ([AKN24, Lemma 3.24, p. 21]) until all exponents
are reduced, then the relations `z_0^n B_s = C_{k(s)} B_{s+1}` for the resulting power
series (see [`f_series_times_monomial`](@ref)).
"""
function f_monomial_normal_form(env::FEnvelope, a::Integer, e::AbstractVector{<:Integer})
    prm = env.prm
    length(e) == prm.nf || throw(ArgumentError("expected $(prm.nf) f-exponents"))
    w = f_monomial_weight(prm, a, e)
    w >= prm.L && return f_zero(env)
    _, _, m = _f_carry_exponents(prm, a, e)
    all(iszero, m) && return f_basis_element(env, w)          # z_0^n = f_0 only
    return with_scratch(env.scratch) do ws
        Q = [d == 1 ? one(env.R.Znz) : zero(env.R.Znz) for d in 1:(env.R.f)]
        _f_monomial_times!(env, Q, a, e, env.prec, ws)
    end
end

"""
    f_series_times_monomial(env, s::WSeries, a, e) -> FElem

The normal form of `s(z_0) · z_0^a ∏ f_u^{e_u}` for a power series `s`: after the exponents are
reduced (see [`f_monomial_normal_form`](@ref)), the element `Q(z_0)·B_s` is written as
`Σ_{c<n} Q_c z_0^c B_s + (Q div z_0^n)·C_{k(s)}·B_{s+1}`, where `z_0^n B_s = f_0 B_s = C_{k(s)} B_{s+1}`,
`C_k = c_0 ⋯ c_{k-1}` and `k(s)` is the number of trailing digits `p - 1` of `s` in base `p`
([AKN24, Lemma 3.24, p. 21]; [AKN24, §7.1, reduce subalgorithm, p. 55]).
"""
function f_series_times_monomial(
    env::FEnvelope, s::WSeries, a::Integer, e::AbstractVector{<:Integer}
)
    prm = env.prm
    needed = prm.L - f_monomial_weight(prm, a, e)     # coefficients z^k with k < needed matter
    needed <= 0 && return FElem(wpoly_zero(env.R), env.R.N)
    s.zprec >= needed || throw(
        ArgumentError("f_series_times_monomial: series known only modulo z^$(s.zprec)")
    )
    prec = min(s.prec, env.prec)
    return with_scratch(env.scratch) do ws
        _f_monomial_times!(env, _aligned(s, 0, needed), a, e, prec, ws)
    end
end

"""
    f_z_power(env, k) -> FElem

The normal form of `z_0^k` (used for the reduction maps of [AKN24, §7.1, Step 6, p. 56]):
with `k = n·j + r`, `r < n`, it is that of `z_0^r P_j B_j`, where
`z_0^{nj} = f_0^j = P_j B_j` ([AKN24, Lemma 3.24, p. 21]). Results are memoized.
"""
function f_z_power(env::FEnvelope, k::Integer)
    prm = env.prm
    k >= 0 || throw(ArgumentError("negative power of z_0"))
    k >= prm.L && return f_zero(env)
    return get_or_compute!(env.zpow, Int(k)) do
        R = env.R
        j, r = divrem(Int(k), prm.n)
        len = prm.L - prm.n * j
        v = with_scratch(env.scratch) do ws
            Q = [
                poly_shift_left!(poly_set_trunc!(zero(R.Znz), a, len - r), r) for
                a in env.zpre[j + 1]
            ]
            _f_reduce!(env, Q, j, ws)
        end
        # z_0^r f_0^j is a basis monomial for j < p; otherwise the relations for f_u^p are used
        FElem(v, j < prm.p ? R.N : env.prec)
    end
end

# ---------------------------------------------------------------------------------------------
# The images needed for the matrices of Steps 6, 8 and 9
# ---------------------------------------------------------------------------------------------

"""
    f_E_powers(env, jmax) -> Vector{WSeries}

The powers `E(z_0)^j`, `0 ≤ j ≤ jmax`, modulo `z_0^L` (entry `j + 1`), each one the previous one
times `E`.
"""
function f_E_powers(env::FEnvelope, jmax::Integer)
    L = env.prm.L
    E = ws_truncate(env.ed.E, L)
    out = [ws_one(env.R, L; prec=E.prec)]
    for _ in 1:jmax
        push!(out, ws_mul(out[end], E; zprec=L))
    end
    return out
end

"""
    f_can_image(env, w, j; Ej=E(z_0)^j) -> FElem

`can(d̃^j · b_w) = E(z_0)^j · b_w` in `Δ^(1)_{R/A_0}`, for the basis monomial `b_w` of F-weight `w`
([AKN24, §7.1, Step 9, p. 57]: "reducing each of the basis elements d̃^{i-Σp^u e_u} z_0^k ∏ f_u^{e_u}
… (replacing d̃ to E(z_0))"). The power `Ej` modulo `z_0^L` can be passed.
"""
function f_can_image(
    env::FEnvelope, w::Integer, j::Integer; Ej::WSeries=ws_pow(env.ed.E, j; zprec=env.prm.L)
)
    c, e = f_basis_exponents(env.prm, w)
    return f_series_times_monomial(env, Ej, c, e)
end

"""
    f_phi_series(env, e; zprec=L) -> WSeries

`∏_u λ_u^{e_u}` modulo `z_0^zprec` for the exponents `e` of a basis monomial, the series in
`φ_{p^u}(f_u)^{e_u} = (λ_u f_{u+1})^{e_u}` ([AKN24, Lemma 3.26(1), p. 21]).
"""
function f_phi_series(
    env::FEnvelope, e::AbstractVector{<:Integer}; zprec::Integer=env.prm.L
)
    prm = env.prm
    lam = ws_one(env.R, zprec)
    for u in 1:(prm.nf - 1)
        e[u] > 0 || continue
        lam = ws_mul(lam, ws_pow(env.lam.lambda[u], e[u]; zprec=zprec); zprec=zprec)
    end
    return lam
end

"""
    f_phi_image(env, w; lam=f_phi_series(env, e)) -> FElem

`φ_i(d̃^{i-Σp^u e_u} z_0^c ∏ f_u^{e_u}) = z_0^{pc} ∏ φ_{p^u}(f_u)^{e_u} = z_0^{pc} ∏ (λ_u f_{u+1})^{e_u}`
for the basis monomial of F-weight `w` ([AKN24, §7.1, Step 9, p. 57]; [AKN24, Lemma 3.26(1), p. 21]).
The series `∏_u λ_u^{e_u}` ([`f_phi_series`](@ref)) can be passed.
"""
function f_phi_image(env::FEnvelope, w::Integer; lam::Union{Nothing,WSeries}=nothing)
    prm = env.prm
    c, e = f_basis_exponents(prm, w)
    # f_{nf} has F-weight n·p^{nf} ≥ L, so a nonzero top exponent gives 0.
    prm.nf > 0 && e[end] > 0 && return f_zero(env)
    prm.p * w >= prm.L && return f_zero(env)             # φ multiplies F-weights by p
    shifted = zeros(Int, prm.nf)
    for u in 1:(prm.nf - 1)
        shifted[u + 1] = e[u]
    end
    lam === nothing && (lam = f_phi_series(env, e; zprec=prm.L - prm.p * w))
    return f_series_times_monomial(env, lam, prm.p * c, shifted)
end
