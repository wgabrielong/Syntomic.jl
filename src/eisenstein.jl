# The normalized Eisenstein polynomial and its Frobenius data in W(F_q)[[z]].
#
# [AKN, §7.1, Input, p. 53]: E(z) is "normalized so that E(0) = p"; this is the hypothesis of
# [AKN, Lemma 4.14, p. 29] (the filtered crystalline Breuil–Kisin orientation) used in Step 4.

"""
    EisensteinData

The data attached to the Eisenstein polynomial over a fixed [`GaloisRing`](@ref) `W(F_q)/p^N`,
truncated at `z`-precision `zprec`:

- `E`: the normalized Eisenstein polynomial `E/(E(0)/p)`, so that `E(0) = p`;
- `phiE = φ(E) = E^σ(z^p)` and `deltaE = δ(E) = (φ(E) - E^p)/p` (a unit, `δ(E)(0) = 1 - p^{p-1}`);
- `descends`: whether `E ∈ Z_p[z]`, in which case every W-matrix of the algorithm has entries in
  `Z_p` (so the computation can be done with `f = 1` arithmetic).
"""
struct EisensteinData
    R::GaloisRing
    zprec::Int
    E::WSeries
    phiE::WSeries
    deltaE::WSeries
    descends::Bool
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
    normalize_eisenstein(E::WSeries) -> WSeries

`E/u` with `u = E(0)/p ∈ W(F_q)^×`, so that the constant term is `p` ([AKN, §7.1, Input]).
"""
function normalize_eisenstein(E::WSeries)
    R = E.R
    u = w_divexact_p(R, ws_coefficient(E, 0), 1)
    w_isunit(R, u) || throw(ArgumentError("E(0) does not have valuation 1"))
    return ws_scale(E, w_inv(R, u))
end

"""
    EisensteinData(R::GaloisRing, X::FiniteChainRing, zprec) -> EisensteinData

Normalize the Eisenstein polynomial of `X` over `R` and compute `φ(E)` and `δ(E)` modulo
`z^zprec` (the δ-structure of [AKN, §2.3, p. 11]; `δ` loses one digit of precision).
"""
function EisensteinData(R::GaloisRing, X::FiniteChainRing, zprec::Integer)
    E = normalize_eisenstein(eisenstein_series(R, X, zprec))
    phiE = ws_frobenius(E; zprec=zprec)
    deltaE = ws_delta(E; zprec=zprec)
    descends = all(all(iszero, c[2:end]) for c in ws_coefficients(E))
    return EisensteinData(R, zprec, E, phiE, deltaE, descends)
end
