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

# The least number f²·r·c of entries of an expansion for which it is computed in parallel
# (benchmarked: an entry of `zp_expand` costs a multiplication matrix, one of `zp_expand_descent`
# only a copy).
const PARALLEL_EXPAND_ENTRIES = 1 << 11
const PARALLEL_DESCENT_ENTRIES = 1 << 14

# The f·r × f·c matrix over Zn whose columns f(j-1)+1, …, f·j are filled by `fill!(M, cols)` for
# the columns j ∈ cols of the W-matrix (written to the columns f(j-first(cols))+1, … of the matrix M
# it is given). With `threaded` and at least `minwork` entries, the column ranges are filled in
# parallel, each into its own matrix, and the parts are concatenated.
function _expand_columns(
    fill!, Zn::ZZModRing, f::Int, r::Int, c::Int, threaded::Bool, minwork::Int
)
    parallel = threaded && threads_available() && c > 1 && f^2 * r * c >= minwork
    k = parallel ? min(c, Threads.nthreads()) : 1
    k == 1 && return fill!(zero_matrix(Zn, f * r, f * c), 1:c)
    parts = tmap(split_range(c, k)) do cols
        fill!(zero_matrix(Zn, f * r, f * length(cols)), cols)
    end
    return reduce(hcat, parts)
end

"""
    zp_expand(W::WMatrix; semilinear=false, threaded=false) -> PMatrix

The `Z_p`-matrix (size `f·r × f·c`) of the `W(F_q)`-linear map with W-matrix `W.A`
(`semilinear = false`, the matrix 𝓜(A)), or of the σ-semilinear map `v ↦ A·σ(v)`
(`semilinear = true`, the matrix 𝓜(A)·(I ⊗ S)). The Z_p-index of `x^d·b_m` is `f(m-1)+d+1`, for
the `Z_p`-basis of `W(F_q)` lifting `1, x, …, x^{f-1}` ([AKN, §7.1, Step 2, p. 54]). With
`threaded = true` blocks of columns are expanded in parallel.
"""
function zp_expand(W::WMatrix; semilinear::Bool=false, threaded::Bool=false)
    R = W.R
    f = R.f
    r, c = size(W.A)
    prec = min(W.prec, R.N)
    f == 1 &&
        return PMatrix(matrix(R.Zn, ZZRingElem[W.A[i, j][1] for i in 1:r, j in 1:c]), prec)
    S = semilinear ? R.sigma : nothing
    M = _expand_columns(R.Zn, f, r, c, threaded, PARALLEL_EXPAND_ENTRIES) do M, cols
        for (jj, j) in enumerate(cols), i in 1:r
            a = W.A[i, j]
            all(iszero, a) && continue
            B = mult_matrix(R, a)
            semilinear && (B = B * S)
            for e in 1:f, d in 1:f
                M[f * (i - 1) + d, f * (jj - 1) + e] = R.Zn(B[d, e])
            end
        end
        return M
    end
    return PMatrix(M, prec)
end

"""
    zp_expand_descent(R, A::PMatrix; semilinear=false, threaded=false) -> PMatrix

For a matrix `A` with entries in `Z_p` (computed with `f = 1` arithmetic), the `Z_p`-matrix over
`W(F_q)` of the corresponding `W(F_q)`-linear map, `A ⊗ I_f` (`semilinear = false`), or of the
σ-semilinear map `v ↦ A·σ(v)`, `A ⊗ S` (`semilinear = true`).

This is the descent optimization: if the Eisenstein polynomial has coefficients in `Z_p`, every
W-matrix of the algorithm has entries in `Z_p` and can be computed with `f = 1` arithmetic; the
`Z_p`-bases are those of [AKN, §7.1, Step 2, p. 54]. With `threaded = true` blocks of columns
are expanded in parallel.
"""
function zp_expand_descent(
    R::GaloisRing, A::PMatrix; semilinear::Bool=false, threaded::Bool=false
)
    f = R.f
    r, c = size(A)
    L = lift(A.M)
    M = _expand_columns(R.Zn, f, r, c, threaded, PARALLEL_DESCENT_ENTRIES) do M, cols
        for (jj, j) in enumerate(cols), i in 1:r
            a = L[i, j]
            iszero(a) && continue
            for e in 1:f, d in 1:f
                b = semilinear ? a * R.sigma[d, e] : (d == e ? a : zero(ZZ))
                iszero(b) && continue
                M[f * (i - 1) + d, f * (jj - 1) + e] = R.Zn(b)
            end
        end
        return M
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

# Solve A·Y = X for a block lower-triangular A (blocks f×f), splitting A in halves. The diagonal
# blocks are solved from top to bottom; `count[]` is the number of them reached so far, so that a
# failure can be located in this order (see `_solve_by_columns`).
function _solve_block_lower(
    A::ZZModMatrix, X::ZZModMatrix, f::Int, p::Int, count::Base.RefValue{Int}=Ref(0)
)
    n = nrows(A)
    nb = div(n, f)
    if nb <= 1
        count[] += 1
        return _solve_diagonal_block(A, X, p)
    end
    h = f * div(nb, 2)
    Y1 = _solve_block_lower(A[1:h, 1:h], X[1:h, :], f, p, count)
    X2 = X[(h + 1):n, :] - A[(h + 1):n, 1:h] * Y1
    Y2 = _solve_block_lower(A[(h + 1):n, (h + 1):n], X2, f, p, count)
    return vcat(Y1, Y2)
end

# Solve A·Y = X for a block upper-triangular A; the diagonal blocks are solved from bottom to top
# (`count` as for `_solve_block_lower`).
function _solve_block_upper(
    A::ZZModMatrix, X::ZZModMatrix, f::Int, p::Int, count::Base.RefValue{Int}=Ref(0)
)
    n = nrows(A)
    nb = div(n, f)
    if nb <= 1
        count[] += 1
        return _solve_diagonal_block(A, X, p)
    end
    h = f * div(nb, 2)
    Y2 = _solve_block_upper(A[(h + 1):n, (h + 1):n], X[(h + 1):n, :], f, p, count)
    X1 = X[1:h, :] - A[1:h, (h + 1):n] * Y2
    Y1 = _solve_block_upper(A[1:h, 1:h], X1, f, p, count)
    return vcat(Y1, Y2)
end

"""
    PARALLEL_SOLVE_WORK

The least work `n²·c` (for an `n×n` matrix and `c` right-hand sides) for which a triangular
solve is split into blocks of columns that are solved in parallel (at least 8 columns per block);
below it the solve is serial. Benchmarked in the square of the algorithm: below about
`n = c = 100` the parallel solves do not pay.
"""
const PARALLEL_SOLVE_WORK = 10^6

# The number of column blocks for a solve with an n×n matrix and c right-hand sides.
function _solve_block_count(n::Int, c::Int, threaded::Bool)
    (threaded && threads_available() && n * n * c >= PARALLEL_SOLVE_WORK) || return 1
    return clamp(div(c, 8), 1, Threads.nthreads())
end

"""
    split_range(c, k) -> Vector{UnitRange{Int}}

`1:c` split into `k` consecutive ranges whose lengths differ by at most one.
"""
function split_range(c::Integer, k::Integer)
    k >= 1 || throw(ArgumentError("split_range: need at least one part, got $k"))
    q, r = divrem(Int(c), Int(k))
    out = UnitRange{Int}[]
    s = 1
    for b in 1:k
        len = q + (b <= r ? 1 : 0)
        push!(out, s:(s + len - 1))
        s += len
    end
    return out
end

# Solve A·Y = X column block by column block with `solve(A, X, count)` (the columns of Y only
# depend on the corresponding columns of X), in parallel when `threaded` is true and the work is
# large enough. The exception rethrown is the one a serial solve raises: the diagonal blocks are
# solved in the same order for every column block, and a failure at a diagonal block (a singular
# block, or a right-hand side not divisible by its power of p) has the same exception for every
# column block that fails there, so it is the failure at the earliest diagonal block.
function _solve_by_columns(solve, A::ZZModMatrix, X::ZZModMatrix; threaded::Bool=false)
    k = _solve_block_count(nrows(A), ncols(X), threaded)
    k <= 1 && return solve(A, X, Ref(0))
    parts = tmap(split_range(ncols(X), k)) do cols
        count = Ref(0)
        try
            (solve(A, X[:, cols], count), 0, nothing)
        catch err
            err isa Union{PrecisionExhausted,ArgumentError} || rethrow()
            (nothing, count[], err)
        end
    end
    failures = [(c, err) for (_, c, err) in parts if err !== nothing]
    isempty(failures) || throw(last(first(sort(failures; by=first))))
    return reduce(hcat, [Y for (Y, _, _) in parts])
end

# The precision of the solution of A·Y = X (proofs of [AKN, Lemmas 7.6 and 7.8]).
function _solve_precision(A::PMatrix, X::PMatrix, e::Integer)
    0 <= e < A.prec || throw(
        PrecisionExhausted(
            "triangular solve: exponent $e not below the precision $(A.prec) of the matrix",
        ),
    )
    P = min(A.prec, X.prec) - e
    P >= 1 || throw(
        PrecisionExhausted(
            "triangular solve: the solution would be known to precision $P ≤ 0"
        ),
    )
    return P
end

# Check that A is square, of size divisible by f, and block lower-triangular (blocks f×f).
function _check_block_triangular(A::PMatrix, f::Integer; lower::Bool=true)
    n = size(A, 1)
    n % f == 0 || throw(
        ArgumentError("triangular solve: size $n is not divisible by the block size $f")
    )
    M = A.M
    for j in 1:n, i in 1:n
        (lower ? div(i - 1, f) < div(j - 1, f) : div(i - 1, f) > div(j - 1, f)) || continue
        iszero(M[i, j]) || throw(
            ArgumentError("triangular solve: the matrix is not block lower-triangular")
        )
    end
    return nothing
end

"""
    solve_lower(A, X, f, p; exponent=nothing, threaded=false) -> PMatrix

Solve `A·Y = X` for a block lower-triangular `A` (square blocks of size `f` on the diagonal,
each `p^v` times an invertible matrix, as for the expansion of a lower-triangular W-matrix).
With `threaded = true`, blocks of columns of `X` are solved in parallel when the matrices are
large enough (see `PARALLEL_SOLVE_WORK`); the result and any exception are the same as
for a serial solve.

The computed `Y` satisfies `A·Y ≡ X (mod p^N)` for the stored representatives. Hence (proofs of
[AKN, Lemmas 7.6 and 7.8]) it agrees with the true solution modulo `p^{min(A.prec, X.prec) - e}`, where
`e` is the p-exponent of the cokernel of `A` ([`local_exponent`](@ref); computed if not given).
Throws [`PrecisionExhausted`](@ref) when a required division by a power of `p` fails or the
resulting precision would not be positive.
"""
function solve_lower(
    A::PMatrix,
    X::PMatrix,
    f::Integer,
    p::Integer;
    exponent::Union{Nothing,Integer}=nothing,
    threaded::Bool=false,
)
    size(A, 1) == size(A, 2) == size(X, 1) || throw(DimensionMismatch("solve_lower"))
    _check_block_triangular(A, f; lower=true)
    e = exponent === nothing ? local_exponent(A.M, p, A.prec) : exponent
    P = _solve_precision(A, X, e)
    size(A, 1) == 0 && return PMatrix(X.M, P)
    solve(B, Y, count) = _solve_block_lower(B, Y, Int(f), Int(p), count)
    return PMatrix(_solve_by_columns(solve, A.M, X.M; threaded=threaded), P)
end

"""
    solve_right_lower(X, A, f, p; exponent=nothing, threaded=false) -> PMatrix

Solve `Y·A = X` for a block lower-triangular `A` (i.e. compute `X·A^{-1}`); precision as in
[`solve_lower`](@ref). With `threaded = true`, blocks of rows of `X` are solved in parallel.
"""
function solve_right_lower(
    X::PMatrix,
    A::PMatrix,
    f::Integer,
    p::Integer;
    exponent::Union{Nothing,Integer}=nothing,
    threaded::Bool=false,
)
    size(A, 1) == size(A, 2) == size(X, 2) || throw(DimensionMismatch("solve_right_lower"))
    _check_block_triangular(A, f; lower=true)
    e = exponent === nothing ? local_exponent(A.M, p, A.prec) : exponent
    P = _solve_precision(A, X, e)
    size(A, 1) == 0 && return PMatrix(X.M, P)
    solve(B, Y, count) = _solve_block_upper(B, Y, Int(f), Int(p), count)
    Yt = _solve_by_columns(solve, transpose(A.M), transpose(X.M); threaded=threaded)
    return PMatrix(transpose(Yt), P)
end
