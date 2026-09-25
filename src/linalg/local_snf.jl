# Smith normal form over the local rings Z/p^M (and hence over W(F_q)/p^M after expansion).
#
# Used for the final cohomology computation ([AKN, §7.1, Return]; [AKN, Lemma 7.1]) and to compute the
# exponents of the matrices that are inverted in Steps 7, 8 and 10 (the precision lost when
# solving A·Y = X is the largest elementary-divisor valuation of A; proofs of [AKN, Lemmas 7.6 and 7.8]).
#
# Nemo's `snf`/`elementary_divisors` on `ZZModMatrix` are too slow for our sizes and the generic
# p-adic linear algebra loses precision, so we use a direct local elimination: pivot on an entry
# of minimal valuation, clear its column by row operations. Since only the diagonal is needed,
# column operations are never performed (the pivot row's other entries are multiples of the
# pivot and could be cleared by column operations touching only the pivot row).
#
# The elimination has two implementations, which share the driver below and hence choose the
# same pivots:
# * the word-size path, for moduli p^M ≤ 2^62, on `UInt64` residues. For p = 2 a reduction is a
#   bit mask. For odd p, the row operation x ↦ x - t·y reduces t·y with Shoup's precomputed
#   quotient ⌊t·2^64/p^M⌋ of the row's multiplier t: one widening multiplication and at most one
#   correction per entry;
# * the big path, on `ZZRingElem` entries updated in place.
# Both store the transpose of the matrix, so that a row operation runs over contiguous memory.
#
# For M beyond the word size, the precisions k, 4k, 16k, … ≤ M/4 are tried first, where k is the
# largest word-size precision, and then M. The Smith normal form modulo p^K (K ≤ M) is the
# reduction of the one modulo p^M, whose diagonal entries p^a are reduced to p^min(a, K); so if all
# min(r, c) diagonal entries modulo p^K are nonzero, their valuations are those modulo p^M as
# well. The valuations are usually far below M (an a priori precision), and the cost of the big
# path grows with the size of p^K.

"""
    WORD_MODULUS_MAX

The largest modulus `p^M` of the word-size elimination, `2^62`.
"""
const WORD_MODULUS_MAX = UInt64(1) << 62

"""
    WORD_MIN_WORK

The number of entry updates in one elimination step of the word-size path from which its row
operations run in parallel (with `threaded = true`); smaller steps run serially, since the
tasks would cost more than they save.
"""
const WORD_MIN_WORK = 1 << 18

"""
    BIG_MIN_WORK

The number of entry updates in one elimination step of the `ZZRingElem` path from which its row
operations run in parallel (with `threaded = true`).
"""
const BIG_MIN_WORK = 1 << 12

"""
    snf_word_precision(p) -> Int

The largest `k` with `p^k ≤ 2^62` (the precision of the word-size Smith normal form for `p`);
`0` if `p > 2^62`.
"""
function snf_word_precision(p::Integer)
    k, q = 0, Int128(1)
    while q * p <= WORD_MODULUS_MAX
        q *= p
        k += 1
    end
    return k
end

"""
    snf_dims(A) -> (r, c)

The numbers of rows and columns of a `Matrix{ZZRingElem}`, `ZZMatrix` or `ZZModMatrix`.
"""
snf_dims(A::AbstractMatrix) = size(A)
snf_dims(A::Union{ZZMatrix,ZZModMatrix}) = (nrows(A), ncols(A))

"""
    snf_entry(A, i, j)

The entry `A[i, j]` as a `ZZRingElem`, or, for a `ZZMatrix` or `ZZModMatrix`, as a pointer to its
FLINT storage (valid while `A` is preserved from garbage collection); both can be passed to
Nemo's `mod` and `mod!`. For a `ZZModMatrix` this is the representative in `[0, modulus)`.
"""
snf_entry(A::AbstractMatrix{ZZRingElem}, i::Int, j::Int) = A[i, j]
snf_entry(A::Union{ZZMatrix,ZZModMatrix}, i::Int, j::Int) = Nemo.mat_entry_ptr(A, i, j)

# ---------------------------------------------------------------------------------------------
# Word-size arithmetic modulo p^k ≤ 2^62
# ---------------------------------------------------------------------------------------------

"""
    WordSNFModulus(p, k)

Arithmetic modulo `m = p^k ≤ 2^62` on `UInt64` residues in `[0, m)`. For odd `p` it stores
`pinv = p^{-1} mod 2^64` and `plim = ⌊(2^64 - 1)/p⌋`: `x` is divisible by `p` if and only if
`x·pinv mod 2^64 ≤ plim`, and then `x/p = x·pinv mod 2^64`.
"""
struct WordSNFModulus
    p::UInt64
    k::Int
    m::UInt64
    pinv::UInt64
    plim::UInt64
end

function WordSNFModulus(p::Integer, k::Integer)
    (p >= 2 && 1 <= k <= snf_word_precision(p)) ||
        throw(ArgumentError("$p^$k is not a word-size modulus (at most 2^62)"))
    P = UInt64(p)
    return WordSNFModulus(
        P, Int(k), P^k, isodd(P) ? inv_mod_2_64(P) : zero(UInt64), typemax(UInt64) ÷ P
    )
end

"""
    inv_mod_2_64(a) -> UInt64

The inverse of the odd number `a` modulo `2^64`, by Newton's iteration `x ↦ x·(2 - a·x)`, which
doubles the number of correct bits (`a` is its own inverse modulo `2^3`).
"""
function inv_mod_2_64(a::UInt64)
    isodd(a) || throw(ArgumentError("$a is not invertible modulo 2^64"))
    x = a
    for _ in 1:5
        x *= 2 - a * x
    end
    return x
end

"""
    snf_valuation(W::WordSNFModulus, x, lower, bound) -> Int

`min(v_p(x), bound)` for a residue `x`, given that `v_p(x) ≥ lower`.
"""
function snf_valuation(W::WordSNFModulus, x::UInt64, lower::Int, bound::Int)
    W.p == 2 && return min(trailing_zeros(x), bound)
    v = 0
    while v < bound
        y = x * W.pinv
        y <= W.plim || break
        x = y
        v += 1
    end
    return v
end

"""
    word_divexact_ppow(W::WordSNFModulus, x, v) -> UInt64

`x / p^v` for a residue `x` divisible by `p^v` (as an integer in `[0, m)`).
"""
word_divexact_ppow(W::WordSNFModulus, x::UInt64, v::Int) = W.p == 2 ? x >> v : x * W.pinv^v

"""
    word_mulmod(W::WordSNFModulus, a, b) -> UInt64

`a·b mod m` for residues `a`, `b`.
"""
function word_mulmod(W::WordSNFModulus, a::UInt64, b::UInt64)
    W.p == 2 && return (a * b) & (W.m - 1)
    return (widemul(a, b) % W.m) % UInt64
end

"""
    shoup_quotient(W::WordSNFModulus, t) -> UInt64

`⌊t·2^64/m⌋` for a residue `t`: then `t·x - ⌊shoup_quotient·x/2^64⌋·m ∈ [0, 2m)` for every
`x < 2^64` (this needs `m ≤ 2^63`).
"""
shoup_quotient(W::WordSNFModulus, t::UInt64) = ((UInt128(t) << 64) ÷ W.m) % UInt64

"""
    snf_matrix(A, W::WordSNFModulus) -> Matrix{UInt64}

The transpose of `A` with entries reduced modulo `W.m`.
"""
function snf_matrix(A, W::WordSNFModulus)
    r, c = snf_dims(A)
    Bt = Matrix{UInt64}(undef, c, r)
    GC.@preserve A for i in 1:r, j in 1:c
        Bt[j, i] = mod(snf_entry(A, i, j), W.m)
    end
    return Bt
end

"""
    SNFPivotStep

The data of one elimination step with pivot `p^v·u` (`u` a unit): `v` and `u^{-1}` modulo the
modulus, and for `ZZRingElem` entries also `p^v`.
"""
struct SNFPivotStep{T}
    v::Int
    pv::T
    uinv::T
end

"""
    snf_pivot_step(W::WordSNFModulus, x, v) -> SNFPivotStep

The data of an elimination step whose pivot is the residue `x` of valuation `v`.
"""
function snf_pivot_step(W::WordSNFModulus, x::UInt64, v::Int)
    u = word_divexact_ppow(W, x, v)
    return SNFPivotStep(v, zero(UInt64), invmod(u % Int, W.m % Int) % UInt64)
end

"""
    snf_row_operation!(Bt::Matrix{UInt64}, W, step, s, i)

Subtract `t` times the pivot row `s` from row `i` (rows are the columns of the transposed
storage `Bt`), where `t = (B[i, s]/p^v)·u^{-1}` clears the entry of row `i` below the pivot
`p^v·u`; only the columns after `s` are updated. For odd `p` the products are reduced with
[`Syntomic.shoup_quotient`](@ref).
"""
function snf_row_operation!(
    Bt::Matrix{UInt64}, W::WordSNFModulus, step::SNFPivotStep{UInt64}, s::Int, i::Int
)
    c = size(Bt, 1)
    m = W.m
    t = word_mulmod(W, word_divexact_ppow(W, Bt[s, i], step.v), step.uinv)
    if W.p == 2
        mask = m - 1
        @inbounds for j in (s + 1):c
            Bt[j, i] = (Bt[j, i] - t * Bt[j, s]) & mask
        end
    else
        tq = shoup_quotient(W, t)
        @inbounds for j in (s + 1):c
            x = Bt[j, s]
            y = t * x - (widemul(tq, x) >> 64) % UInt64 * m      # t·x mod m, in [0, 2m)
            y = y >= m ? y - m : y
            b = Bt[j, i]
            Bt[j, i] = b >= y ? b - y : b + (m - y)
        end
    end
    return nothing
end

"""
    snf_row_chunk(W::WordSNFModulus, nrows, width) -> Int

The number of rows per task in a parallel elimination step of the word-size path with `nrows`
rows of `width` entries to update: at least `WORD_MIN_WORK ÷ 4` entry updates per task.
"""
function snf_row_chunk(::WordSNFModulus, nrows::Int, width::Int)
    return max(1, cld(nrows, 4 * Threads.nthreads()), cld(WORD_MIN_WORK ÷ 4, max(width, 1)))
end

# ---------------------------------------------------------------------------------------------
# ZZRingElem arithmetic modulo p^M
# ---------------------------------------------------------------------------------------------

"""
    BigSNFModulus(p, prec)

Arithmetic modulo `m = p^prec` on `ZZRingElem` residues in `[0, m)`, updated in place. For odd
`p` it stores the powers `pk[a + 1] = p^a`, `0 ≤ a ≤ prec`, for the valuations.
"""
struct BigSNFModulus
    p::Int
    k::Int
    m::ZZRingElem
    pk::Vector{ZZRingElem}
end

function BigSNFModulus(p::Integer, prec::Integer)
    pk = isodd(p) ? [ZZ(p)^a for a in 0:prec] : ZZRingElem[]
    return BigSNFModulus(Int(p), Int(prec), ZZ(p)^prec, pk)
end

"""
    snf_valuation(W::BigSNFModulus, x, lower, bound) -> Int

`min(v_p(x), bound)` for a nonzero residue `x`, given that `v_p(x) ≥ lower`.
"""
function snf_valuation(W::BigSNFModulus, x::ZZRingElem, lower::Int, bound::Int)
    W.p == 2 && return min(trailing_zeros(x), bound)
    is_divisible_by(x, W.pk[bound + 1]) && return bound
    v = lower
    while is_divisible_by(x, W.pk[v + 2])
        v += 1
    end
    return v
end

"""
    snf_matrix(A, W::BigSNFModulus) -> Matrix{ZZRingElem}

The transpose of `A` with entries reduced modulo `W.m` (new objects, which the elimination
modifies in place).
"""
function snf_matrix(A, W::BigSNFModulus)
    r, c = snf_dims(A)
    Bt = Matrix{ZZRingElem}(undef, c, r)
    GC.@preserve A for i in 1:r, j in 1:c
        Bt[j, i] = mod!(ZZRingElem(), snf_entry(A, i, j), W.m)
    end
    return Bt
end

"""
    snf_pivot_step(W::BigSNFModulus, x, v) -> SNFPivotStep

The data of an elimination step whose pivot is the residue `x` of valuation `v`.
"""
function snf_pivot_step(W::BigSNFModulus, x::ZZRingElem, v::Int)
    pv = ZZ(W.p)^v
    return SNFPivotStep(v, pv, invmod(divexact(x, pv), W.m))
end

"""
    snf_row_operation!(Bt::Matrix{ZZRingElem}, W, step, s, i)

Subtract `t` times the pivot row `s` from row `i` (rows are the columns of the transposed
storage `Bt`), where `t = (B[i, s]/p^v)·u^{-1}` clears the entry of row `i` below the pivot
`p^v·u`; only the columns after `s` are updated, in place, skipping the zero entries of the
pivot row.
"""
function snf_row_operation!(
    Bt::Matrix{ZZRingElem}, W::BigSNFModulus, step::SNFPivotStep{ZZRingElem}, s::Int, i::Int
)
    c = size(Bt, 1)
    m = W.m
    t = divexact(Bt[s, i], step.pv)
    mul!(t, t, step.uinv)
    mod!(t, t, m)
    for j in (s + 1):c
        x = Bt[j, s]
        iszero(x) && continue
        y = Bt[j, i]
        submul!(y, t, x)
        mod!(y, y, m)
    end
    return nothing
end

"""
    snf_row_chunk(W::BigSNFModulus, nrows, width) -> Int

The number of rows per task in a parallel elimination step of the `ZZRingElem` path.
"""
snf_row_chunk(::BigSNFModulus, nrows::Int, width::Int) =
    max(1, cld(nrows, 4 * Threads.nthreads()))

# ---------------------------------------------------------------------------------------------
# The elimination
# ---------------------------------------------------------------------------------------------

"""
    snf_find_pivot(Bt, W, s, lower) -> (v, i, j)

An entry `B[i, j]` of minimal valuation `v` in the submatrix `B[s:r, s:c]` of the matrix whose
transpose is `Bt`, scanning the rows in order; stops early at valuation `lower`, a lower bound
for all valuations in the submatrix. Returns `v = W.k` if the submatrix is zero.
"""
function snf_find_pivot(
    Bt::AbstractMatrix, W::Union{WordSNFModulus,BigSNFModulus}, s::Int, lower::Int
)
    c, r = size(Bt)
    best_v, bi, bj = W.k, 0, 0
    for i in s:r, j in s:c
        x = Bt[j, i]
        iszero(x) && continue
        v = snf_valuation(W, x, lower, best_v)
        if v < best_v
            best_v, bi, bj = v, i, j
            best_v <= lower && return best_v, bi, bj
        end
    end
    return best_v, bi, bj
end

"""
    snf_swap_pivot!(Bt, s, bi, bj)

Move the entry `B[bi, bj]` to position `(s, s)` by swapping rows `s`, `bi` and columns `s`, `bj`
of the submatrix `B[s:r, s:c]` (the matrix `B` is stored as its transpose `Bt`).
"""
function snf_swap_pivot!(Bt::AbstractMatrix, s::Int, bi::Int, bj::Int)
    c, r = size(Bt)
    if bi != s
        for j in s:c
            Bt[j, s], Bt[j, bi] = Bt[j, bi], Bt[j, s]
        end
    end
    if bj != s
        for i in s:r
            Bt[s, i], Bt[bj, i] = Bt[bj, i], Bt[s, i]
        end
    end
    return nothing
end

"""
    snf_eliminate_below!(Bt, W, s, v; threaded, min_work)

Clear the entries of column `s` below the pivot `B[s, s]` of valuation `v` by row operations
([`Syntomic.snf_row_operation!`](@ref)), in parallel over the rows when `threaded` is true and the
step updates at least `min_work` entries. The entries of column `s` below the pivot are left
as they are: they are not read again.
"""
function snf_eliminate_below!(
    Bt::AbstractMatrix, W, s::Int, v::Int; threaded::Bool, min_work::Int
)
    c, r = size(Bt)
    step = snf_pivot_step(W, Bt[s, s], v)
    rows = [i for i in (s + 1):r if !iszero(Bt[s, i])]
    width = c - s
    parallel = threaded && length(rows) * width >= min_work
    chunk = snf_row_chunk(W, length(rows), width)
    tforeach(rows; threaded=parallel, chunksize=chunk) do i
        snf_row_operation!(Bt, W, step, s, i)
    end
    return nothing
end

"""
    snf_eliminate!(Bt, W; threaded, min_work) -> (vals, nzero)

The local elimination on the transposed matrix `Bt` (modified in place) modulo `W.m = p^k`: the
sorted valuations of the nonzero diagonal entries of the Smith normal form and the number of
zero ones (see [`local_elementary_divisors`](@ref)).
"""
function snf_eliminate!(Bt::AbstractMatrix, W; threaded::Bool, min_work::Int)
    c, r = size(Bt)
    vals = Int[]
    lower = 0
    for s in 1:min(r, c)
        v, bi, bj = snf_find_pivot(Bt, W, s, lower)
        v >= W.k && break
        snf_swap_pivot!(Bt, s, bi, bj)
        push!(vals, v)
        lower = v
        snf_eliminate_below!(Bt, W, s, v; threaded=threaded, min_work=min_work)
    end
    return vals, min(r, c) - length(vals)
end

"""
    snf_word(A, p, prec; threaded=false, min_work=WORD_MIN_WORK) -> (vals, nzero)

[`local_elementary_divisors`](@ref) by the word-size elimination; requires `p^prec ≤ 2^62`.
"""
function snf_word(
    A, p::Integer, prec::Integer; threaded::Bool=false, min_work::Integer=WORD_MIN_WORK
)
    W = WordSNFModulus(p, prec)
    return snf_eliminate!(snf_matrix(A, W), W; threaded=threaded, min_work=Int(min_work))
end

"""
    snf_big(A, p, prec; threaded=false, min_work=BIG_MIN_WORK) -> (vals, nzero)

[`local_elementary_divisors`](@ref) by the elimination on `ZZRingElem` entries, for any `prec ≥ 1`.
"""
function snf_big(
    A, p::Integer, prec::Integer; threaded::Bool=false, min_work::Integer=BIG_MIN_WORK
)
    prec >= 1 || throw(ArgumentError("the precision must be positive"))
    W = BigSNFModulus(p, prec)
    return snf_eliminate!(snf_matrix(A, W), W; threaded=threaded, min_work=Int(min_work))
end

"""
    snf_precisions(p, prec) -> Vector{Int}

The precisions at which [`Syntomic.local_snf`](@ref) tries the Smith normal form modulo `p^prec`:
the word-size precision `k` of `p` ([`Syntomic.snf_word_precision`](@ref)) if `1 ≤ k < prec`,
then `4k, 16k, …` up to `prec/4` (`4, 16, …` if `k = 0`), and `prec`. (The intermediate
precisions are at most `prec/4`, so that they add little to the cost when the result is only
decided at `prec`.)
"""
function snf_precisions(p::Integer, prec::Integer)
    k = snf_word_precision(p)
    precs = 1 <= k < prec ? [k] : Int[]
    K = 4 * max(k, 1)
    while 4 * K <= prec
        push!(precs, K)
        K *= 4
    end
    push!(precs, prec)
    return precs
end

"""
    local_snf(A, p, prec, threaded) -> (vals, nzero)

[`local_elementary_divisors`](@ref) for a `Matrix{ZZRingElem}`, `ZZMatrix` or `ZZModMatrix`
(whose modulus has already been checked). The Smith normal form is computed modulo `p^K` for
`K` in [`Syntomic.snf_precisions`](@ref)`(p, prec)`, on machine words when `p^K ≤ 2^62`, until no
diagonal entry vanishes modulo `p^K` or `K = prec`: the reductions modulo `p^K` of the diagonal
entries modulo `p^prec` are the diagonal entries modulo `p^K`.
"""
function local_snf(A, p::Integer, prec::Integer, threaded::Bool)
    p >= 2 || throw(ArgumentError("p = $p must be a prime"))
    r, c = snf_dims(A)
    prec >= 1 || return Int[], min(r, c)
    k = snf_word_precision(p)
    local vals, nzero
    for K in snf_precisions(p, prec)
        vals, nzero = if K <= k
            snf_word(A, p, K; threaded=threaded)
        else
            snf_big(A, p, K; threaded=threaded)
        end
        nzero == 0 && break
    end
    return vals, nzero
end

"""
    local_elementary_divisors(A, p, prec; threaded=false) -> (vals, nzero)

Smith normal form of an integer matrix `A` over the local ring `Z/p^prec`.

Returns `vals`, the sorted p-adic valuations `< prec` of the nonzero diagonal entries, and
`nzero`, the number of diagonal entries (among `min(size(A))`) that are `≡ 0 (mod p^prec)`.
If every nonzero elementary divisor of `A` over `Z_p` has valuation `< prec`, then `vals` are
exactly their valuations and `nzero` is the number of zero elementary divisors (see the proof of
[AKN, Lemma 7.1, pp. 57–58]).

`A` may be a `Matrix{ZZRingElem}`, a `ZZMatrix` or a `ZZModMatrix` (for which `prec` must not
exceed the precision of its modulus). The elimination runs on machine words when `p^prec ≤ 2^62`.
For larger `p^prec` it is first tried modulo the largest power `p^k` of `p` that fits and then
modulo `p^{4k}, p^{16k}, …` (up to `p^{prec/4}`), which decides the result as soon as no
elementary divisor vanishes there. With `threaded = true` the row operations of each elimination
step run in parallel when there is enough work.
"""
function local_elementary_divisors(
    A::AbstractMatrix{ZZRingElem}, p::Integer, prec::Integer; threaded::Bool=false
)
    return local_snf(A, p, prec, threaded)
end

function local_elementary_divisors(
    A::ZZMatrix, p::Integer, prec::Integer; threaded::Bool=false
)
    return local_snf(A, p, prec, threaded)
end

function local_elementary_divisors(
    A::ZZModMatrix, p::Integer, prec::Integer; threaded::Bool=false
)
    m = modulus(base_ring(A))
    cap = valuation(m, p)
    m == ZZ(p)^cap || throw(ArgumentError("the modulus $m is not a power of $p"))
    prec <= cap ||
        throw(ArgumentError("precision $prec exceeds the precision $cap of the modulus"))
    return local_snf(A, p, prec, threaded)
end

"""
    local_exponent(A, p, prec; threaded=false) -> Int

The largest p-adic valuation of an elementary divisor of the square matrix `A` over `Z_p`,
computed modulo `p^prec`; this is the p-exponent of the cokernel of `A`. Throws
[`PrecisionExhausted`](@ref) if some elementary divisor is `≡ 0 (mod p^prec)` (the matrix is
singular or `prec` is too small).
"""
function local_exponent(A, p::Integer, prec::Integer; threaded::Bool=false)
    vals, nzero = local_elementary_divisors(A, p, prec; threaded=threaded)
    nzero == 0 || throw(
        PrecisionExhausted(
            "local_exponent: $nzero elementary divisor(s) are ≡ 0 mod $p^$prec (singular matrix or precision too low)",
        ),
    )
    return isempty(vals) ? 0 : maximum(vals)
end
