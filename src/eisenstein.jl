# The normalized Eisenstein polynomial and its Frobenius data in W(F_q)[[z]].
#
# [AKN24, §7.1, Input, p. 53]: E(z) is "normalized so that E(0) = p"; this is the hypothesis of
# [AKN24, Lemma 4.14, p. 29] (the filtered crystalline Breuil–Kisin orientation) used in Step 4.

"""
    EisensteinData

The data attached to the Eisenstein polynomial over a fixed [`GaloisRing`](@ref) `W(F_q)/p^N`,
truncated at `z`-precision `zprec`:

- `E`: the normalized Eisenstein polynomial `E/(E(0)/p)`, so that `E(0) = p`;
- `phiE = φ(E) = E^σ(z^p)` and `deltaE = δ(E) = (φ(E) - E^p)/p` (a unit, `δ(E)(0) = 1 - p^{p-1}`).
"""
struct EisensteinData
    R::GaloisRing
    zprec::Int
    E::WSeries
    phiE::WSeries
    deltaE::WSeries
end

"""
    eisenstein_series(R::GaloisRing, X::FiniteChainRing, zprec) -> WSeries

The (unnormalized) Eisenstein polynomial of `X` as a series over `R`, modulo `z^zprec`.
"""
function eisenstein_series(R::GaloisRing, X::FiniteChainRing, zprec::Integer)
    (R.p == X.p && R.htilde == ZZRingElem.(X.htilde)) || throw(
        ArgumentError("the Galois ring does not match the presentation of W(F_q) of X")
    )
    coeffs = [w_from_coordinates(R, c) for c in X.E]
    return ws_from_coefficients(R, coeffs, zprec)
end

"""
    eisenstein_unit(R::GaloisRing, X::FiniteChainRing) -> WElem

The unit `u = E(0)/p ∈ W(F_q)^×`. It is computed from the exact integer coordinates of `E(0)`,
which are all divisible by `p` because `E(0)` has valuation `1`, so `u` is known modulo `p^N`.
"""
function eisenstein_unit(R::GaloisRing, X::FiniteChainRing)
    return w_from_coordinates(R, [div(c, X.p) for c in X.E[1]])
end

"""
    normalize_eisenstein(R::GaloisRing, X::FiniteChainRing, zprec) -> WSeries

The normalized Eisenstein polynomial `E/u`, `u = E(0)/p`, whose constant term is `p`
([AKN24, §7.1, Input, p. 53]), modulo `z^zprec`.
"""
function normalize_eisenstein(R::GaloisRing, X::FiniteChainRing, zprec::Integer)
    E = eisenstein_series(R, X, zprec)
    return ws_scale(E, w_inv(R, eisenstein_unit(R, X)))
end

"""
    EisensteinData(R::GaloisRing, X::FiniteChainRing, zprec) -> EisensteinData

Normalize the Eisenstein polynomial of `X` over `R` and compute `φ(E)` and `δ(E)` modulo
`z^zprec` (the δ-structure of [AKN24, §2.3, p. 11]; `δ` loses one digit of precision).
"""
function EisensteinData(R::GaloisRing, X::FiniteChainRing, zprec::Integer)
    E = normalize_eisenstein(R, X, zprec)
    phiE = ws_frobenius(E; zprec=zprec)
    deltaE = ws_delta(E; zprec=zprec)
    return EisensteinData(R, zprec, E, phiE, deltaE)
end
