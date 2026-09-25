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

# Find an entry of minimal valuation in B[s:r, s:c]; stop early at valuation `lower`, which is a
# lower bound for all valuations in the remaining submatrix.
function _find_pivot(B::Matrix{ZZRingElem}, s::Int, p::Int, prec::Int, lower::Int)
    r, c = size(B)
    best_v, bi, bj = prec, 0, 0
    for j in s:c, i in s:r
        x = B[i, j]
        iszero(x) && continue
        v = valuation(x, p)
        if v < best_v
            best_v, bi, bj = v, i, j
            best_v <= lower && return best_v, bi, bj
        end
    end
    return best_v, bi, bj
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
exceed the precision of its modulus). With `threaded = true` the row operations of each
elimination step run in parallel when there is enough work.
"""
function local_elementary_divisors(
    A::AbstractMatrix{ZZRingElem}, p::Integer, prec::Integer; threaded::Bool=false
)
    r, c = size(A)
    prec >= 1 || return Int[], min(r, c)
    m = ZZ(p)^prec
    B = Matrix{ZZRingElem}(undef, r, c)
    for j in 1:c, i in 1:r
        B[i, j] = mod(A[i, j], m)
    end
    vals = Int[]
    lower = 0
    for s in 1:min(r, c)
        v, bi, bj = _find_pivot(B, s, Int(p), Int(prec), lower)
        v >= prec && break
        if bi != s
            for j in 1:c
                B[s, j], B[bi, j] = B[bi, j], B[s, j]
            end
        end
        if bj != s
            for i in 1:r
                B[i, s], B[i, bj] = B[i, bj], B[i, s]
            end
        end
        push!(vals, v)
        lower = v
        pv = ZZ(p)^v
        uinv = invmod(divexact(B[s, s], pv), m)
        rows = [i for i in (s + 1):r if !iszero(B[i, s])]
        work = length(rows) * (c - s)
        chunk = max(1, cld(length(rows), 4 * max(Threads.nthreads(), 1)))
        tforeach(rows; threaded=threaded && work >= 4096, chunksize=chunk) do i
            t = mod(divexact(B[i, s], pv) * uinv, m)
            for j in (s + 1):c
                x = B[s, j]
                iszero(x) && continue
                B[i, j] = mod(B[i, j] - t * x, m)
            end
            B[i, s] = zero(ZZ)
        end
    end
    return vals, min(r, c) - length(vals)
end

function local_elementary_divisors(A::ZZMatrix, p::Integer, prec::Integer; kw...)
    return local_elementary_divisors(Matrix{ZZRingElem}(A), p, prec; kw...)
end

function local_elementary_divisors(A::ZZModMatrix, p::Integer, prec::Integer; kw...)
    m = modulus(base_ring(A))
    cap = valuation(m, p)
    m == ZZ(p)^cap || throw(ArgumentError("the modulus $m is not a power of $p"))
    prec <= cap ||
        throw(ArgumentError("precision $prec exceeds the precision $cap of the modulus"))
    return local_elementary_divisors(Matrix{ZZRingElem}(lift(A)), p, prec; kw...)
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
