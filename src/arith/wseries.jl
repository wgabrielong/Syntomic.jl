# Truncated power series in z over W(F_q)/p^N: the Breuil–Kisin ring A = W(F_q)[[z]] of the
# algorithm, truncated at F-weight L (the F-filtration on A is the z-adic filtration).
#
# The δ-ring structure used in [AKN, §7.1, Step 3] is the one with
#     φ(Σ a_k z^k) = Σ σ(a_k) z^{pk},     δ(x) = (φ(x) - x^p)/p,
# where σ is the Witt vector Frobenius of W(F_q): this is the δ-structure of the Breuil–Kisin prism
# W(F_q)[[z_0]], with δ(z_0) = 0 [AKN, §2.3, p. 11].

"""
    WSeries

A power series `Σ_k a_k z^k` over `W(F_q)/p^N`, known modulo `(p^prec, z^zprec)`.

Fields:
- `R::GaloisRing`;
- `c::Vector{ZZModPolyRingElem}`: `f` polynomials; `c[d+1]` holds the `x^d`-coordinates of the
  coefficients `a_{zval}, a_{zval+1}, …` (so the series is `z^zval · Σ_d c[d+1](z) x^d`);
- `zval::Int`: all coefficients below `z^zval` are (known to be) zero;
- `zprec::Int`: the series is known modulo `z^zprec` (coefficients of `z^k`, `k ≥ zprec`, are
  unknown and not stored);
- `prec::Int`: the coefficients are known modulo `p^prec` (`prec ≤ N`).
"""
struct WSeries
    R::GaloisRing
    c::Vector{ZZModPolyRingElem}
    zval::Int
    zprec::Int
    prec::Int
end

function Base.show(io::IO, s::WSeries)
    return print(
        io, "WSeries over $(s.R) (zval=$(s.zval), zprec=$(s.zprec), prec=$(s.prec))"
    )
end

"""
    ws_length(s) -> Int

The number `zprec - zval` of stored coefficient positions (at least 0).
"""
ws_length(s::WSeries) = max(s.zprec - s.zval, 0)

"""
    ws_zero(R, zprec; prec=R.N) -> WSeries

The series `0 + O(z^zprec)`.
"""
ws_zero(R::GaloisRing, zprec::Integer; prec::Integer=R.N) =
    WSeries(R, wpoly_zero(R), zprec, zprec, prec)

"""
    ws_const(R, a::WElem, zprec) -> WSeries

The constant series `a + O(z^zprec)`.
"""
function ws_const(R::GaloisRing, a::WElem, zprec::Integer; prec::Integer=R.N)
    zprec <= 0 && return ws_zero(R, zprec; prec=prec)
    c = [R.Znz([R.Zn(a[d])]) for d in 1:(R.f)]
    return WSeries(R, c, 0, zprec, prec)
end

"""
    ws_one(R, zprec; prec=R.N) -> WSeries
"""
ws_one(R::GaloisRing, zprec::Integer; prec::Integer=R.N) =
    ws_const(R, w_one(R), zprec; prec=prec)

"""
    ws_monomial(R, a::WElem, k, zprec; prec=R.N) -> WSeries

The series `a z^k + O(z^zprec)` (`k ≥ 0`), with `a` known modulo `p^prec`.
"""
function ws_monomial(R::GaloisRing, a::WElem, k::Integer, zprec::Integer; prec::Integer=R.N)
    k >= 0 || throw(ArgumentError("ws_monomial: negative exponent $k"))
    k >= zprec && return ws_zero(R, zprec; prec=prec)
    c = [R.Znz([R.Zn(a[d])]) for d in 1:(R.f)]
    return WSeries(R, c, k, zprec, min(prec, R.N))
end

"""
    ws_from_coefficients(R, coeffs::Vector{WElem}, zprec; zval=0, prec=R.N) -> WSeries

The series `Σ_k coeffs[k+1] z^{zval+k} + O(z^zprec)` (coefficients beyond `zprec` are dropped),
with coefficients known modulo `p^prec`.
"""
function ws_from_coefficients(
    R::GaloisRing,
    coeffs::AbstractVector{WElem},
    zprec::Integer;
    zval::Integer=0,
    prec::Integer=R.N,
)
    zval >= 0 ||
        throw(ArgumentError("ws_from_coefficients: negative valuation offset $zval"))
    len = max(min(length(coeffs), zprec - zval), 0)
    c = [R.Znz([R.Zn(coeffs[k][d]) for k in 1:len]) for d in 1:(R.f)]
    return WSeries(R, c, min(zval, zprec), zprec, min(prec, R.N))
end

"""
    ws_coefficient(s, k) -> WElem

The coefficient of `z^k` (absolute index). Throws if `k ≥ s.zprec` (unknown coefficient).
"""
function ws_coefficient(s::WSeries, k::Integer)
    k < s.zprec ||
        throw(ArgumentError("coefficient of z^$k is unknown (zprec = $(s.zprec))"))
    k < s.zval && return w_zero(s.R)
    return ZZRingElem[lift(coeff(s.c[d], k - s.zval)) for d in 1:(s.R.f)]
end

"""
    ws_coefficients(s) -> Vector{WElem}

All known coefficients `a_0, …, a_{zprec-1}`.
"""
ws_coefficients(s::WSeries) = [ws_coefficient(s, k) for k in 0:(s.zprec - 1)]

# The components of s re-expressed relative to z^zv and truncated to length zp - zv
# (requires zv ≤ s.zval or s being zero there).
function _aligned(s::WSeries, zv::Integer, zp::Integer)
    len = zp - zv
    len <= 0 && return wpoly_zero(s.R)
    shift = s.zval - zv
    shift >= 0 || throw(ArgumentError("_aligned: cannot lower the valuation offset"))
    shift >= len && return wpoly_zero(s.R)
    return [truncate(shift_left(c, shift), len) for c in s.c]
end

"""
    ws_truncate(s, zprec) -> WSeries

`s` modulo `z^zprec` (only lowers the z-precision).
"""
function ws_truncate(s::WSeries, zprec::Integer)
    zp = min(s.zprec, zprec)
    zv = min(s.zval, zp)
    return WSeries(s.R, _aligned(s, zv, zp), zv, zp, s.prec)
end

"""
    ws_add(a, b) -> WSeries
"""
function ws_add(a::WSeries, b::WSeries)
    zp = min(a.zprec, b.zprec)
    zv = min(a.zval, b.zval, zp)
    A = _aligned(a, zv, zp)
    B = _aligned(b, zv, zp)
    return WSeries(a.R, [A[d] + B[d] for d in 1:(a.R.f)], zv, zp, min(a.prec, b.prec))
end

"""
    ws_sub(a, b) -> WSeries
"""
function ws_sub(a::WSeries, b::WSeries)
    zp = min(a.zprec, b.zprec)
    zv = min(a.zval, b.zval, zp)
    A = _aligned(a, zv, zp)
    B = _aligned(b, zv, zp)
    return WSeries(a.R, [A[d] - B[d] for d in 1:(a.R.f)], zv, zp, min(a.prec, b.prec))
end

"""
    ws_neg(a) -> WSeries
"""
ws_neg(a::WSeries) = WSeries(a.R, [-c for c in a.c], a.zval, a.zprec, a.prec)

"""
    ws_scale(s, a::WElem; prec=s.R.N) -> WSeries

`a·s` for a scalar `a ∈ W(F_q)/p^N` known modulo `p^prec`.
"""
ws_scale(s::WSeries, a::WElem; prec::Integer=s.R.N) =
    WSeries(s.R, wpoly_scale(s.R, s.c, a), s.zval, s.zprec, min(s.prec, prec))

"""
    ws_scale_int(s, k) -> WSeries

`k·s` for an integer `k`.
"""
function ws_scale_int(s::WSeries, k::IntegerLike)
    kk = s.R.Zn(k)
    return WSeries(s.R, [kk * c for c in s.c], s.zval, s.zprec, s.prec)
end

"""
    ws_shift(s, k) -> WSeries

`z^k · s` for `k ≥ 0`.
"""
function ws_shift(s::WSeries, k::Integer)
    k >= 0 || throw(ArgumentError("ws_shift: negative shift"))
    return WSeries(s.R, copy(s.c), s.zval + k, s.zprec + k, s.prec)
end

"""
    ws_mul(a, b; zprec=typemax(Int)) -> WSeries

The product, known modulo `z^{min(a.zprec + b.zval, b.zprec + a.zval, zprec)}` and modulo
`p^{min(a.prec, b.prec)}`.
"""
function ws_mul(a::WSeries, b::WSeries; zprec::Integer=typemax(Int))
    zv = a.zval + b.zval
    zp = min(a.zprec + b.zval, b.zprec + a.zval, zprec)
    prec = min(a.prec, b.prec)
    zp <= zv && return ws_zero(a.R, max(zp, 0); prec=prec)
    return WSeries(a.R, wpoly_mullow(a.R, a.c, b.c, zp - zv), zv, zp, prec)
end

"""
    ws_pow(a, e; zprec=a.zprec) -> WSeries

`a^e` for an integer `e ≥ 0`, modulo `z^zprec` (binary powering).
"""
function ws_pow(a::WSeries, e::Integer; zprec::Integer=a.zprec)
    e >= 0 || throw(ArgumentError("ws_pow: negative exponent"))
    result = ws_one(a.R, zprec; prec=a.prec)
    base = ws_truncate(a, zprec)
    k = e
    first = true
    while k > 0
        if isodd(k)
            result = first ? base : ws_mul(result, base; zprec=zprec)
            first = false
        end
        k >>= 1
        k > 0 && (base = ws_mul(base, base; zprec=zprec))
    end
    return result
end

"""
    ws_frobenius(a; zprec=typemax(Int)) -> WSeries

The Frobenius lift `φ(Σ a_k z^k) = Σ σ(a_k) z^{pk}` on `W(F_q)[[z]]`
(`δ(z) = 0`, [AKN, §2.3, p. 11]).
If `a` is known modulo `z^m`, then `φ(a)` is known modulo `z^{pm}`.

[AKN, §7.1, Step 3, p. 55]: "We compute the δ operation on W(F_q)[[z_0]] by
δ(x) = (φ(x) - x^p)/p".
"""
function ws_frobenius(a::WSeries; zprec::Integer=typemax(Int))
    R = a.R
    p = R.p
    zv = p * a.zval
    zp = min(p * a.zprec, zprec)
    zp <= zv && return ws_zero(R, max(zp, 0); prec=a.prec)
    needed = cld(zp - zv, p)                 # coefficients of a (relative to zval) that matter
    comps = wpoly_sigma(R, [truncate(c, needed) for c in a.c])
    comps = [truncate(inflate(c, p), zp - zv) for c in comps]
    return WSeries(R, comps, zv, zp, a.prec)
end

"""
    ws_sigma(a) -> WSeries

Apply `σ` to the coefficients only (no substitution in `z`): `Σ σ(a_k) z^k`. This is the series
`E^σ` used in `φ(E) = E^σ(z^p)`.
"""
ws_sigma(a::WSeries) = WSeries(a.R, wpoly_sigma(a.R, a.c), a.zval, a.zprec, a.prec)

"""
    ws_divexact_p(a, k=1) -> WSeries

`a / p^k` for a series all of whose coefficients are divisible by `p^k`; the result is known
modulo `p^{prec-k}`. Throws [`PrecisionExhausted`](@ref) if `a.prec < k` or if some coefficient
is not divisible by `p^k`.

(`divexact` on `ZZModPolyRingElem` by `p` is not available since `p` is not a unit; we lift to
`Z[z]`, divide there and reduce back.)
"""
function ws_divexact_p(a::WSeries, k::Integer=1)
    k == 0 && return a
    a.prec >= k || throw(
        PrecisionExhausted(
            "ws_divexact_p: cannot divide by p^$k a series known to precision $(a.prec)"
        ),
    )
    R = a.R
    pk = ZZ(R.p)^k
    comps = map(a.c) do c
        cz = lift(R.Zz, c)
        for j in 0:degree(cz)
            iszero(mod(coeff(cz, j), pk)) || throw(
                PrecisionExhausted("ws_divexact_p: coefficient not divisible by $(R.p)^$k"),
            )
        end
        map_coefficients(R.Zn, divexact(cz, pk); parent=R.Znz)
    end
    return WSeries(R, comps, a.zval, a.zprec, a.prec - k)
end

"""
    ws_delta(a; zprec=a.zprec) -> WSeries

`δ(a) = (φ(a) - a^p)/p` ([AKN, §7.1, Step 3]). Loses one digit of p-adic precision ([AKN, §7.2.2]).
"""
function ws_delta(a::WSeries; zprec::Integer=a.zprec)
    phia = ws_frobenius(a; zprec=zprec)
    ap = ws_pow(a, a.R.p; zprec=zprec)
    return ws_divexact_p(ws_sub(phia, ap), 1)
end

"""
    ws_inv(a; zprec=a.zprec) -> WSeries

The inverse of a unit power series (constant term a unit of `W(F_q)`), by Newton iteration
`b ← b(2 - ab)`.
"""
function ws_inv(a::WSeries; zprec::Integer=a.zprec)
    R = a.R
    zp = min(zprec, a.zprec)
    (a.zval == 0 && zp >= 1) ||
        throw(DomainError(a, "ws_inv: the constant term is not a unit (or unknown)"))
    a0 = ws_coefficient(a, 0)
    w_isunit(R, a0) || throw(DomainError(a, "ws_inv: the constant term is not a unit"))
    B = [R.Znz([R.Zn(c)]) for c in w_inv(R, a0)]
    A = _aligned(a, 0, zp)
    k = 1
    while k < zp
        k = min(2 * k, zp)
        AB = wpoly_mullow(R, [truncate(c, k) for c in A], B, k)
        # E = 1 - A B;  B ← B + B E
        E = [-c for c in AB]
        E[1] += one(R.Znz)
        BE = wpoly_mullow(R, B, E, k)
        B = [B[d] + BE[d] for d in 1:(R.f)]
    end
    return WSeries(R, B, 0, zp, a.prec)
end

"""
    ws_iszero(s; prec=s.prec) -> Bool

Whether `s ≡ 0` modulo `(p^prec, z^s.zprec)`.
"""
function ws_iszero(s::WSeries; prec::Integer=s.prec)
    m = ZZ(s.R.p)^prec
    for c in s.c
        cz = lift(s.R.Zz, c)
        for j in 0:degree(cz)
            iszero(mod(coeff(cz, j), m)) || return false
        end
    end
    return true
end

"""
    ws_isequal(a, b; prec=min(a.prec, b.prec)) -> Bool

Whether `a ≡ b` modulo `(p^prec, z^{min(a.zprec, b.zprec)})`.
"""
ws_isequal(a::WSeries, b::WSeries; prec::Integer=min(a.prec, b.prec)) =
    ws_iszero(ws_sub(a, b); prec=prec)

"""
    ws_z_valuation(s) -> Int

The z-adic valuation of `s` modulo `(p^prec, z^zprec)` (equal to `zprec` if `s ≡ 0`).
"""
function ws_z_valuation(s::WSeries)
    m = ZZ(s.R.p)^s.prec
    v = s.zprec
    for c in s.c
        cz = lift(s.R.Zz, c)
        for j in 0:degree(cz)
            if !iszero(mod(coeff(cz, j), m))
                v = min(v, s.zval + j)
                break
            end
        end
    end
    return v
end

"""
    ws_change_ring(s, R2::GaloisRing) -> WSeries

The image of `s` in a Galois ring `R2` with the same `p` and `h̃` but a different precision
(used to reduce shared data computed at a higher precision).
"""
function ws_change_ring(s::WSeries, R2::GaloisRing)
    (R2.p == s.R.p && R2.htilde == s.R.htilde) ||
        throw(ArgumentError("ws_change_ring: incompatible Galois rings"))
    comps = [map_coefficients(R2.Zn, lift(s.R.Zz, c); parent=R2.Znz) for c in s.c]
    return WSeries(R2, comps, s.zval, s.zprec, min(s.prec, R2.N))
end
