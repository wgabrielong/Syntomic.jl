# The prismatic envelope of R = O_K/ϖ^n over the Breuil–Kisin prism A_0 = W(F_q)[[z_0]]
# ("the f-envelope"), truncated at F^{≥L}, L = n·i.
#
# [AKN, §7.1, Step 3, p. 55]: "For Δ^(1)_{R/W(F_q)[[z_0]]}, find elements λ_u ∈ W(F_q)[[z_0]] as in
# Lemma 3.24 and create a lookup table of relations for f_u^p in the prismatic envelope in terms of
# those, noting that here the terms R'_u are all zero (Remark 3.29). … Also create a lookup table for
# the values of φ_{p^u}(f_u) following Lemma 3.26."
#
# With d = E(z_0) and r = z_0^n (so δ(r) = 0 and R'_u = 0, [AKN, Remark 3.29, p. 22]):
#   * relations (after d̃ ↦ d):  z_0^n = f_0  and  f_u^p = c_u f_{u+1},  c_u = -p + λ_u E^{p^{u+1}}
#     ([AKN, Lemma 3.24, p. 21; Cor. 3.31, p. 23]);
#   * divided Frobenius:  φ_{p^u}(f_u) = λ_u f_{u+1}  ([AKN, Lemma 3.26(1), p. 21]);
#   * a W(F_q)-basis of F^{[0,L-1]}Δ^(1)_{R/A_0} is given by the monomials z_0^c ∏ f_u^{e_u} with
#     c < n, e_u < p and F-weight c + n Σ p^u e_u < L ([AKN, Prop. 3.34(2), p. 24]).
# There is exactly one basis monomial b_w in each F-weight 0 ≤ w < L (see `f_basis_exponents`), so
# an element is a vector of L coefficients in W(F_q).
#
# The normal form ("reduce subalgorithm", [AKN, §7.1, p. 55]) of a monomial z_0^a ∏ f_u^{e_u} is
# computed by carrying: z_0^n → f_0, then f_u^p → c_u f_{u+1}; both preserve the F-weight up to the
# z-adic valuation of the coefficient c_u, so the recursion only reaches weights ≥ the starting
# weight, and terms of weight ≥ L are dropped.

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

The f-envelope `Δ^(1)_{R/A_0}` of `R = O_K/ϖ^n` truncated at `F^{≥L}` ([AKN, §7.1, Step 3]),
with the relation coefficients `c_u = -p + λ_u E^{p^{u+1}}` ([AKN, Lemma 3.24]) and a memo table of
normal forms of monomials.
"""
struct FEnvelope
    prm::SyntomicParams
    R::GaloisRing
    ed::EisensteinData
    lam::LambdaData
    c::Vector{WSeries}
    prec::Int
    cache::OnceCache{Tuple{Int,Vector{Int}},FElem}
end

"""
    f_relation_coefficient(ed, lam, u) -> WSeries

`c_u = -p + λ_u E^{p^{u+1}}`, the coefficient in the relation `f_u^p = c_u f_{u+1}` in
`Δ^(1)_{R/A_0}` ([AKN, Lemma 3.24, p. 21] with `R'_u = 0` by [AKN, Remark 3.29, p. 22]).
"""
function f_relation_coefficient(ed::EisensteinData, lam::LambdaData, u::Integer)
    zp = ed.zprec
    t = ws_mul(lam.lambda[u + 1], lam.Epow[u + 2]; zprec=zp)
    return ws_sub(t, ws_const(ed.R, w_int(ed.R, ed.R.p), zp))
end

function FEnvelope(prm::SyntomicParams, R::GaloisRing, ed::EisensteinData, lam::LambdaData)
    c = [f_relation_coefficient(ed, lam, u) for u in 0:(prm.nf - 2)]
    prec = minimum((x.prec for x in c); init=ed.E.prec)
    prec = min(prec, minimum((x.prec for x in lam.lambda[1:max(prm.nf - 1, 0)]); init=prec))
    return FEnvelope(prm, R, ed, lam, c, prec, OnceCache{Tuple{Int,Vector{Int}},FElem}())
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

The basis monomial `b_w = z_0^c ∏ f_u^{e_u}` of F-weight `w` ([AKN, Prop. 3.34(2), p. 24]).
"""
function f_basis_element(env::FEnvelope, w::Integer)
    R = env.R
    v = [d == 1 ? shift_left(one(R.Znz), w) : zero(R.Znz) for d in 1:(R.f)]
    return FElem(v, R.N)
end

# y + a·x for a scalar a ∈ W(F_q) (y is a W-polynomial in the auxiliary variable X)
function _f_addmul(env::FEnvelope, y::Vector{ZZModPolyRingElem}, a::WElem, x::FElem)
    return y .+ wpoly_scale(env.R, x.v, a)
end

"""
    f_coefficient(x, w) -> WElem

The coefficient of the basis monomial of F-weight `w`.
"""
f_coefficient(x::FElem, w::Integer) = ZZRingElem[lift(coeff(c, w)) for c in x.v]

# ---------------------------------------------------------------------------------------------
# Normal forms (the reduce subalgorithm for R = O_K/ϖ^n)
# ---------------------------------------------------------------------------------------------

"""
    f_monomial_weight(prm, a, e) -> Int

The F-weight `a + n Σ_u p^u e_u` of `z_0^a ∏ f_u^{e_u}` ([AKN, Prop. 3.34(2), p. 24]).
"""
f_monomial_weight(prm::SyntomicParams, a::Integer, e::AbstractVector{<:Integer}) =
    f_weight(prm, a, e)

"""
    f_monomial_normal_form(env, a, e) -> FElem

The normal form of the monomial `z_0^a ∏ f_u^{e_u}` (arbitrary exponents) in the basis of
[AKN, Prop. 3.34(2), p. 24], modulo `F^{≥L}` (reduce subalgorithm, [AKN, §7.1, p. 55]): use
`z_0^n = f_0` and `f_u^p = c_u f_{u+1}` ([AKN, Lemma 3.24, p. 21]) until all exponents are
reduced. Results are memoized.
"""
function f_monomial_normal_form(env::FEnvelope, a::Integer, e::AbstractVector{<:Integer})
    prm = env.prm
    length(e) == prm.nf || throw(ArgumentError("expected $(prm.nf) f-exponents"))
    w = f_monomial_weight(prm, a, e)
    w >= prm.L && return f_zero(env)
    key = (Int(a), Vector{Int}(e))
    return get_or_compute!(env.cache, key) do
        _f_monomial_normal_form_uncached(env, key[1], key[2], w)
    end
end

function _f_monomial_normal_form_uncached(env::FEnvelope, a::Int, e::Vector{Int}, w::Int)
    prm = env.prm
    # relation z_0^n = f_0 (the image of τ f_0 - r, [AKN, Cor. 3.31, p. 23])
    if a >= prm.n
        e2 = copy(e)
        e2[1] += 1
        return f_monomial_normal_form(env, a - prm.n, e2)
    end
    # relation f_u^p = c_u f_{u+1} for the smallest u with e_u ≥ p ([AKN, Lemma 3.24, p. 21])
    u = findfirst(>=(prm.p), e)
    if u === nothing
        return f_basis_element(env, w)
    end
    # u is 1-based here; f_{u-1}^p = c_{u-1} f_u
    e2 = copy(e)
    e2[u] -= prm.p
    e2[u + 1] += 1          # u < nf: otherwise the weight would be ≥ L (checked by the caller)
    return f_series_times_monomial(env, env.c[u], a, e2)
end

"""
    f_series_times_monomial(env, s::WSeries, a, e) -> FElem

The normal form of `s(z_0) · z_0^a ∏ f_u^{e_u}` for a power series `s`.
"""
function f_series_times_monomial(
    env::FEnvelope, s::WSeries, a::Integer, e::AbstractVector{<:Integer}
)
    prm = env.prm
    needed = prm.L - f_monomial_weight(prm, a, e)     # coefficients z^k with k < needed matter
    out = wpoly_zero(env.R)
    needed <= 0 && return FElem(out, env.R.N)
    s.zprec >= needed || throw(
        ArgumentError("f_series_times_monomial: series known only modulo z^$(s.zprec)")
    )
    prec = min(s.prec, env.prec)
    for k in (s.zval):(needed - 1)
        sk = ws_coefficient(s, k)
        all(iszero, sk) && continue
        t = f_monomial_normal_form(env, a + k, e)
        prec = min(prec, t.prec)
        out = _f_addmul(env, out, sk, t)
    end
    return FElem(out, prec)
end

"""
    f_z_power(env, k) -> FElem

The normal form of `z_0^k` (used for the reduction maps of [AKN, §7.1, Step 6, p. 56]).
"""
f_z_power(env::FEnvelope, k::Integer) =
    f_monomial_normal_form(env, k, zeros(Int, env.prm.nf))

# ---------------------------------------------------------------------------------------------
# The images needed for the matrices of Steps 6, 8 and 9
# ---------------------------------------------------------------------------------------------

"""
    f_can_image(env, w, j) -> FElem

`can(d̃^j · b_w) = E(z_0)^j · b_w` in `Δ^(1)_{R/A_0}`, for the basis monomial `b_w` of F-weight `w`
([AKN, §7.1, Step 9, p. 57]: "reducing each of the basis elements d̃^{i-Σp^u e_u} z_0^k ∏ f_u^{e_u}
… (replacing d̃ to E(z_0))").
"""
function f_can_image(env::FEnvelope, w::Integer, j::Integer)
    c, e = f_basis_exponents(env.prm, w)
    Ej = ws_pow(env.ed.E, j; zprec=env.prm.L)
    return f_series_times_monomial(env, Ej, c, e)
end

"""
    f_phi_image(env, w) -> FElem

`φ_i(d̃^{i-Σp^u e_u} z_0^c ∏ f_u^{e_u}) = z_0^{pc} ∏ φ_{p^u}(f_u)^{e_u} = z_0^{pc} ∏ (λ_u f_{u+1})^{e_u}`
for the basis monomial of F-weight `w` ([AKN, §7.1, Step 9, p. 57]; [AKN, Lemma 3.26(1), p. 21]).
"""
function f_phi_image(env::FEnvelope, w::Integer)
    prm = env.prm
    c, e = f_basis_exponents(prm, w)
    # f_{nf} has F-weight n·p^{nf} ≥ L, so a nonzero top exponent gives 0.
    prm.nf > 0 && e[end] > 0 && return f_zero(env)
    lam = ws_one(env.R, prm.L)
    for u in 1:(prm.nf - 1)
        e[u] > 0 || continue
        lam = ws_mul(lam, ws_pow(env.lam.lambda[u], e[u]; zprec=prm.L); zprec=prm.L)
    end
    shifted = zeros(Int, prm.nf)
    for u in 1:(prm.nf - 1)
        shifted[u + 1] = e[u]
    end
    return f_series_times_monomial(env, lam, prm.p * c, shifted)
end
