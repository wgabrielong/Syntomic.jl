# Matrices over W(F_q)/p^N and Z/p^N, their expansion to Z_p-matrices, and triangular solves.
#
# [AKN, §7.1, Strategy, p. 54]: "The maps in the diagram are not W(F_q)-linear
# thanks to Frobenius, but they are of course Z_p-linear, so we use W(F_q) ≅ Z_p^f to compute
# matrix representatives for this square in Z_p-modules." Step 2 fixes the Z_p-basis x^d·b_m
# (0 ≤ d < f). We order it W-major with the x-power fastest: the Z_p-index of x^d·b_m is
# f·(m-1) + d + 1. Then
#   * a W(F_q)-linear map with W-matrix A has Z_p-matrix 𝓜(A) = (M(A_ij))_{ij}, where M(a) is
#     the multiplication matrix of a (see `mult_matrix`);
#   * a σ-semilinear map v ↦ Φ·σ(v) has Z_p-matrix 𝓜(Φ)·(I ⊗ S), where S is the matrix of σ.

"""
    PMatrix(M::ZZModMatrix, prec)

A `Z_p`-matrix known modulo `p^prec`, stored as a matrix over `Z/p^N` (`prec ≤ N`).
"""
struct PMatrix
    M::ZZModMatrix
    prec::Int
end

Base.size(A::PMatrix) = (nrows(A.M), ncols(A.M))
Base.size(A::PMatrix, d::Integer) = size(A)[d]

function Base.show(io::IO, A::PMatrix)
    r, c = size(A)
    return print(io, "$r×$c PMatrix known modulo p^$(A.prec)")
end

"""
    pm_mul(A, B) -> PMatrix

The product, known modulo `p^{min(A.prec, B.prec)}`.
"""
pm_mul(A::PMatrix, B::PMatrix) = PMatrix(A.M * B.M, min(A.prec, B.prec))

"""
    pm_sub(A, B) -> PMatrix
"""
pm_sub(A::PMatrix, B::PMatrix) = PMatrix(A.M - B.M, min(A.prec, B.prec))

"""
    pm_add(A, B) -> PMatrix
"""
pm_add(A::PMatrix, B::PMatrix) = PMatrix(A.M + B.M, min(A.prec, B.prec))

"""
    pm_neg(A) -> PMatrix
"""
pm_neg(A::PMatrix) = PMatrix(-A.M, A.prec)

"""
    pm_vcat(A, B) -> PMatrix
"""
pm_vcat(A::PMatrix, B::PMatrix) = PMatrix(vcat(A.M, B.M), min(A.prec, B.prec))

"""
    pm_hcat(A, B) -> PMatrix
"""
pm_hcat(A::PMatrix, B::PMatrix) = PMatrix(hcat(A.M, B.M), min(A.prec, B.prec))

"""
    pm_integer_matrix(A) -> Matrix{ZZRingElem}

The entries of `A`, reduced into `[0, p^prec)`.
"""
function pm_integer_matrix(A::PMatrix, p::Integer)
    m = ZZ(p)^A.prec
    L = lift(A.M)
    return [mod(L[i, j], m) for i in 1:nrows(L), j in 1:ncols(L)]
end

"""
    pm_isequal(A, B, p; prec=min(A.prec, B.prec)) -> Bool

Whether `A ≡ B (mod p^prec)`.
"""
function pm_isequal(A::PMatrix, B::PMatrix, p::Integer; prec::Integer=min(A.prec, B.prec))
    size(A) == size(B) || return false
    m = ZZ(p)^prec
    D = lift(A.M - B.M)
    return all(iszero(mod(D[i, j], m)) for i in 1:nrows(D), j in 1:ncols(D))
end

# ---------------------------------------------------------------------------------------------
# W-matrices and their expansion to Z_p
# ---------------------------------------------------------------------------------------------

"""
    WMatrix(R, A::Matrix{WElem}, prec)

A matrix with entries in `W(F_q)/p^N`, known modulo `p^prec`. Column `j` holds the coordinates
(with respect to a `W(F_q)`-basis) of the image of the `j`-th basis vector.
"""
struct WMatrix
    R::GaloisRing
    A::Matrix{WElem}
    prec::Int
end

Base.size(A::WMatrix) = size(A.A)

"""
    wm_zero(R, r, c; prec=R.N) -> WMatrix
"""
wm_zero(R::GaloisRing, r::Integer, c::Integer; prec::Integer=R.N) =
    WMatrix(R, [w_zero(R) for _ in 1:r, _ in 1:c], prec)

"""
    zp_expand(W::WMatrix; semilinear=false) -> PMatrix

The `Z_p`-matrix (size `f·r × f·c`) of the `W(F_q)`-linear map with W-matrix `W.A`
(`semilinear = false`, the matrix 𝓜(A)), or of the σ-semilinear map `v ↦ A·σ(v)`
(`semilinear = true`, the matrix 𝓜(A)·(I ⊗ S)). The Z_p-index of `x^d·b_m` is `f(m-1)+d+1`.
"""
function zp_expand(W::WMatrix; semilinear::Bool=false)
    R = W.R
    f = R.f
    r, c = size(W.A)
    M = zero_matrix(R.Zn, f * r, f * c)
    S = semilinear ? R.sigma : nothing
    for j in 1:c, i in 1:r
        a = W.A[i, j]
        all(iszero, a) && continue
        B = mult_matrix(R, a)
        semilinear && (B = B * S)
        for e in 1:f, d in 1:f
            M[f * (i - 1) + d, f * (j - 1) + e] = R.Zn(B[d, e])
        end
    end
    return PMatrix(M, W.prec)
end

"""
    zp_expand_descent(R, A::PMatrix; semilinear=false) -> PMatrix

For a matrix `A` with entries in `Z_p` (computed with `f = 1` arithmetic), the `Z_p`-matrix over
`W(F_q)` of the corresponding `W(F_q)`-linear map, `A ⊗ I_f` (`semilinear = false`), or of the
σ-semilinear map `v ↦ A·σ(v)`, `A ⊗ S` (`semilinear = true`).

This is the descent optimization: if the Eisenstein polynomial has coefficients in `Z_p`, every
W-matrix of the algorithm has entries in `Z_p` and can be computed with `f = 1` arithmetic.
"""
function zp_expand_descent(R::GaloisRing, A::PMatrix; semilinear::Bool=false)
    f = R.f
    r, c = size(A)
    L = lift(A.M)
    M = zero_matrix(R.Zn, f * r, f * c)
    for j in 1:c, i in 1:r
        a = L[i, j]
        iszero(a) && continue
        for e in 1:f, d in 1:f
            b = semilinear ? a * R.sigma[d, e] : (d == e ? a : zero(ZZ))
            iszero(b) && continue
            M[f * (i - 1) + d, f * (j - 1) + e] = R.Zn(b)
        end
    end
    return PMatrix(M, min(A.prec, R.N))
end

# ---------------------------------------------------------------------------------------------
# Block-triangular solves
# ---------------------------------------------------------------------------------------------

# Solve D·Y ≡ X (mod p^N) for a diagonal block D = p^v·D' with D' invertible modulo p.
function _solve_diagonal_block(D::ZZModMatrix, X::ZZModMatrix, p::Int)
    Zn = base_ring(D)
    Dz = lift(D)
    v = typemax(Int)
    for j in 1:ncols(Dz), i in 1:nrows(Dz)
        x = Dz[i, j]
        iszero(x) || (v = min(v, valuation(x, p)))
    end
    v == typemax(Int) && throw(
        PrecisionExhausted("triangular solve: a diagonal block is ≡ 0 (singular matrix)"),
    )
    pv = ZZ(p)^v
    Dp = matrix(Zn, divexact(Dz, pv))
    is_unit(det(Dp)) || throw(
        ArgumentError(
            "triangular solve: diagonal block is not p^v times an invertible matrix"
        ),
    )
    Xz = lift(X)
    for j in 1:ncols(Xz), i in 1:nrows(Xz)
        iszero(mod(Xz[i, j], pv)) || throw(
            PrecisionExhausted("triangular solve: right-hand side not divisible by $p^$v"),
        )
    end
    return inv(Dp) * matrix(Zn, divexact(Xz, pv))
end

function _solve_block_lower(A::ZZModMatrix, X::ZZModMatrix, f::Int, p::Int)
    n = nrows(A)
    nb = div(n, f)
    nb <= 1 && return _solve_diagonal_block(A, X, p)
    h = f * div(nb, 2)
    Y1 = _solve_block_lower(A[1:h, 1:h], X[1:h, :], f, p)
    X2 = X[(h + 1):n, :] - A[(h + 1):n, 1:h] * Y1
    Y2 = _solve_block_lower(A[(h + 1):n, (h + 1):n], X2, f, p)
    return vcat(Y1, Y2)
end

function _solve_block_upper(A::ZZModMatrix, X::ZZModMatrix, f::Int, p::Int)
    n = nrows(A)
    nb = div(n, f)
    nb <= 1 && return _solve_diagonal_block(A, X, p)
    h = f * div(nb, 2)
    Y2 = _solve_block_upper(A[(h + 1):n, (h + 1):n], X[(h + 1):n, :], f, p)
    X1 = X[1:h, :] - A[1:h, (h + 1):n] * Y2
    Y1 = _solve_block_upper(A[1:h, 1:h], X1, f, p)
    return vcat(Y1, Y2)
end

"""
    solve_lower(A, X, f, p; exponent=nothing) -> PMatrix

Solve `A·Y = X` for a block lower-triangular `A` (square blocks of size `f` on the diagonal,
each `p^v` times an invertible matrix, as for the expansion of a lower-triangular W-matrix).

The computed `Y` satisfies `A·Y ≡ X (mod p^N)` for the stored representatives. Hence (proofs of
[AKN, Lemmas 7.6 and 7.8]) it agrees with the true solution modulo `p^{min(A.prec, X.prec) - e}`, where
`e` is the p-exponent of the cokernel of `A` ([`local_exponent`](@ref); computed if not given).
Throws [`PrecisionExhausted`](@ref) when a required division by a power of `p` fails.
"""
function solve_lower(
    A::PMatrix, X::PMatrix, f::Integer, p::Integer; exponent::Union{Nothing,Integer}=nothing
)
    size(A, 1) == size(A, 2) == size(X, 1) || throw(DimensionMismatch("solve_lower"))
    e = exponent === nothing ? local_exponent(A.M, p, A.prec) : exponent
    size(A, 1) == 0 && return PMatrix(X.M, min(A.prec, X.prec) - e)
    Y = _solve_block_lower(A.M, X.M, Int(f), Int(p))
    return PMatrix(Y, min(A.prec, X.prec) - e)
end

"""
    solve_right_lower(X, A, f, p; exponent=nothing) -> PMatrix

Solve `Y·A = X` for a block lower-triangular `A` (i.e. compute `X·A^{-1}`); precision as in
[`solve_lower`](@ref).
"""
function solve_right_lower(
    X::PMatrix, A::PMatrix, f::Integer, p::Integer; exponent::Union{Nothing,Integer}=nothing
)
    size(A, 1) == size(A, 2) == size(X, 2) || throw(DimensionMismatch("solve_right_lower"))
    e = exponent === nothing ? local_exponent(A.M, p, A.prec) : exponent
    size(A, 1) == 0 && return PMatrix(X.M, min(A.prec, X.prec) - e)
    Yt = _solve_block_upper(transpose(A.M), transpose(X.M), Int(f), Int(p))
    return PMatrix(transpose(Yt), min(A.prec, X.prec) - e)
end
