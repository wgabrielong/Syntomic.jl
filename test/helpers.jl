# Shared helpers for the test suite.
using Nemo
using Hecke
using Random

const S = Syntomic

"""
Build a Galois ring together with the q-adic field presented by the same (Conway) polynomial.
"""
function galois_ring_and_qadic(p::Int, f::Int, N::Int)
    K, a = qadic_field(p, f; precision=N, cached=false)
    htilde = [lift(ZZ, c) for c in coefficients(defining_polynomial(K))]
    R = S.GaloisRing(p, f == 1 ? [0, 1] : htilde, N)
    return R, K, a
end

"""Coordinates of a q-adic integer in the basis 1, a, …, a^{f-1}, reduced modulo p^N."""
function qadic_to_welem(R::S.GaloisRing, x)
    return ZZRingElem[mod(lift(ZZ, coeff(x, d)), R.pN) for d in 0:(R.f - 1)]
end

"""The q-adic integer with the given coordinates."""
function welem_to_qadic(K, a, v::Vector{ZZRingElem})
    return sum(K(v[d + 1]) * a^d for d in 0:(length(v) - 1))
end

"""A uniformly random element of W(F_q)/p^N."""
rand_welem(rng::AbstractRNG, R::S.GaloisRing) =
    ZZRingElem[rand(rng, 0:(R.pN - 1)) for _ in 1:(R.f)]

"""A random unit of W(F_q)/p^N."""
function rand_unit(rng::AbstractRNG, R::S.GaloisRing)
    while true
        a = rand_welem(rng, R)
        S.w_isunit(R, a) && return a
    end
end

"""A random series with the given z-precision and z-valuation offset."""
function rand_wseries(rng::AbstractRNG, R::S.GaloisRing, zprec::Int; zval::Int=0)
    coeffs = [rand_welem(rng, R) for _ in 1:max(zprec - zval, 0)]
    return S.ws_from_coefficients(R, coeffs, zprec; zval=zval)
end

"""Naive truncated product of coefficient lists (oracle for `ws_mul`)."""
function naive_series_product(R::S.GaloisRing, a::Vector, b::Vector, len::Int)
    out = [S.w_zero(R) for _ in 1:len]
    for i in 1:min(length(a), len), j in 1:min(length(b), len - i + 1)
        out[i + j - 1] = S.w_add(R, out[i + j - 1], S.w_mul(R, a[i], b[j]))
    end
    return out
end

"""A random polynomial in z over Z/p^N of the given length."""
rand_poly(rng::AbstractRNG, R::S.GaloisRing, len::Int) =
    R.Znz([R.Zn(rand(rng, 0:(R.pN - 1))) for _ in 1:max(len, 0)])
