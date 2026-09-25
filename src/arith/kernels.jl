# Low-level kernels for polynomials in z with coefficients in W(F_q)/p^N.
#
# A "W-polynomial" is a vector of f polynomials A = (A_0, …, A_{f-1}) in (Z/p^N)[z], standing for
# Σ_d A_d(z) x^d. These kernels are shared by the power-series type `WSeries` and by the
# envelope elements; they are the innermost loops of the whole computation.

"""
    wpoly_zero(R) -> Vector{ZZModPolyRingElem}
"""
wpoly_zero(R::GaloisRing) = [zero(R.Znz) for _ in 1:(R.f)]

# Reduce a vector of 2f-1 polynomials Σ_k P_k x^k (0 ≤ k ≤ 2f-2) modulo h̃.
function _wpoly_reduce_x(R::GaloisRing, prods::Vector{ZZModPolyRingElem})
    f = R.f
    out = prods[1:f]
    for k in (f + 1):(2 * f - 1)
        iszero(prods[k]) && continue
        for d in 1:f
            c = R.xpow[k, d]
            iszero(c) && continue
            out[d] += R.Zn(c) * prods[k]
        end
    end
    return out
end

"""
    wpoly_mullow(R, A, B, len) -> Vector{ZZModPolyRingElem}

The product of the W-polynomials `A` and `B` truncated to `z`-length `len` (i.e. modulo `z^len`).

For `f = 1` this is a single FLINT `mullow`. For `f = 2` it uses Karatsuba in `x` (3 products);
for `f ≥ 3` schoolbook multiplication in `x` (`f²` products), followed by reduction modulo `h̃`.
"""
function wpoly_mullow(
    R::GaloisRing, A::Vector{ZZModPolyRingElem}, B::Vector{ZZModPolyRingElem}, len::Integer
)
    f = R.f
    len <= 0 && return wpoly_zero(R)
    f == 1 && return [mullow(A[1], B[1], len)]
    if f == 2
        m0 = mullow(A[1], B[1], len)
        m2 = mullow(A[2], B[2], len)
        m1 = mullow(A[1] + A[2], B[1] + B[2], len) - m0 - m2
        return _wpoly_reduce_x(R, [m0, m1, m2])
    end
    prods = [zero(R.Znz) for _ in 1:(2 * f - 1)]
    for d in 1:f
        iszero(A[d]) && continue
        for e in 1:f
            iszero(B[e]) && continue
            prods[d + e - 1] += mullow(A[d], B[e], len)
        end
    end
    return _wpoly_reduce_x(R, prods)
end

"""
    wpoly_scale(R, A, a::WElem) -> Vector{ZZModPolyRingElem}

The W-polynomial `a·A` for a scalar `a ∈ W(F_q)/p^N`.
"""
function wpoly_scale(R::GaloisRing, A::Vector{ZZModPolyRingElem}, a::WElem)
    f = R.f
    f == 1 && return [R.Zn(a[1]) * A[1]]
    prods = [zero(R.Znz) for _ in 1:(2 * f - 1)]
    for d in 1:f
        iszero(a[d]) && continue
        for e in 1:f
            iszero(A[e]) && continue
            prods[d + e - 1] += R.Zn(a[d]) * A[e]
        end
    end
    return _wpoly_reduce_x(R, prods)
end

"""
    wpoly_sigma(R, A) -> Vector{ZZModPolyRingElem}

Apply the Witt vector Frobenius `σ` to the coefficients of the W-polynomial `A` (no change of
the variable `z`).
"""
function wpoly_sigma(R::GaloisRing, A::Vector{ZZModPolyRingElem})
    f = R.f
    f == 1 && return copy(A)
    out = [zero(R.Znz) for _ in 1:f]
    for e in 1:f
        iszero(A[e]) && continue
        for d in 1:f
            c = R.sigma[d, e]
            iszero(c) && continue
            out[d] += R.Zn(c) * A[e]
        end
    end
    return out
end
