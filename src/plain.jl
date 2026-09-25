# Plain (FLINT-free) data, for sending intermediate results to other processes and for storing
# them. Objects with FLINT parents (Galois rings, polynomials, matrices) cannot be serialized
# across processes; these types hold only `Int`/`BigInt` data and are rebuilt on arrival.

"""
    PlainWMatrix

A W-matrix ([`WMatrix`](@ref)) as plain data: the prime `p`, the presentation `htilde` of
`W(F_q) = Z_p[x]/(h̃)`, the coordinate vectors of the entries, and the precision `prec`.
"""
struct PlainWMatrix
    p::Int
    htilde::Vector{BigInt}
    entries::Matrix{Vector{BigInt}}
    prec::Int
end

Base.size(P::PlainWMatrix) = size(P.entries)
Base.size(P::PlainWMatrix, d::Integer) = size(P.entries, d)

"""
    to_plain(W::WMatrix) -> PlainWMatrix
"""
function to_plain(W::WMatrix)
    entries = [BigInt.(W.A[i, j]) for i in 1:size(W.A, 1), j in 1:size(W.A, 2)]
    return PlainWMatrix(W.R.p, BigInt.(W.R.htilde), entries, W.prec)
end

"""
    from_plain(R::GaloisRing, P::PlainWMatrix) -> WMatrix

The W-matrix over `R` (reducing the entries modulo `p^N`); `R` must present `W(F_q)` by the same
`h̃`. The precision is capped at `N`.
"""
function from_plain(R::GaloisRing, P::PlainWMatrix)
    (R.p == P.p && BigInt.(R.htilde) == P.htilde) ||
        throw(ArgumentError("from_plain: the Galois ring does not match the data"))
    A = [w_from_coordinates(R, P.entries[i, j]) for i in 1:size(P, 1), j in 1:size(P, 2)]
    return WMatrix(R, A, min(P.prec, R.N))
end

"""
    plain_top_left(P::PlainWMatrix, r, c) -> PlainWMatrix

The top-left `r × c` block.
"""
plain_top_left(P::PlainWMatrix, r::Integer, c::Integer) =
    PlainWMatrix(P.p, P.htilde, P.entries[1:r, 1:c], P.prec)
