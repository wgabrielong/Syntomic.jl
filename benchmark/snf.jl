# Benchmarks for the local Smith normal form (`local_elementary_divisors`).
#
#   julia --project=benchmark -t 8 benchmark/snf.jl
#   julia --project=benchmark -t 8 benchmark/snf.jl quick      # fewer repetitions
#
# Compares the current implementation (word-size path for p^M ≤ 2^62, in-place `ZZRingElem`
# path beyond) with the reference elimination `ReferenceSNF` below, the implementation of
# Syntomic.jl before the word-size path (commit 5796f80: `ZZRingElem` entries, one new object per
# arithmetic operation). Every timing is the minimum over several interleaved repetitions.
#
# Cases:
# 1. the matrices whose Smith normal forms `syntomic_cohomology(zpn(2, 2), i)` computes, for
#    i = 30 (K_59(Z/4)) and i = 40: the exponent phase (`red` and `can` on the ∇-terms at the
#    exponent precision), Steps 7, 8 and 10 (`red`, `can1`, `N^{≥i}∇` at the working precision)
#    and the two syntomic differentials; with the total run time, this gives the share of the
#    Smith normal forms in the total;
# 2. random 200×400 matrices at small and large moduli;
# 3. matrices with elementary divisors beyond the word size, at high precision.
using Syntomic, Nemo, Hecke
using Printf
using Random

const S = Syntomic
const QUICK = "quick" in ARGS
const REPS = QUICK ? 3 : 7

"""
    ReferenceSNF

The elimination of Syntomic.jl at commit 5796f80, for comparison: the same pivoting on a
`Matrix{ZZRingElem}`, every operation allocating a new `ZZRingElem`.
"""
module ReferenceSNF
using Nemo
using Syntomic: tforeach

"""Entry of minimal valuation in `B[s:r, s:c]`, scanning the columns; stops at `lower`."""
function find_pivot(B::Matrix{ZZRingElem}, s::Int, p::Int, prec::Int, lower::Int)
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

"""The valuations and the number of zero diagonal entries of the Smith form modulo `p^prec`."""
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
        v, bi, bj = find_pivot(B, s, Int(p), Int(prec), lower)
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

"""The reference elimination for a `ZZModMatrix` (entries lifted to `ZZRingElem`)."""
function local_elementary_divisors(A::ZZModMatrix, p::Integer, prec::Integer; kw...)
    return local_elementary_divisors(Matrix{ZZRingElem}(lift(A)), p, prec; kw...)
end
end

"""
    mintime(f, reps) -> Float64

The minimum over `reps` runs of the time of `f()` in seconds.
"""
mintime(f, reps::Int=REPS) = minimum(@elapsed(f()) for _ in 1:reps)

"""
    snf_path(A, p, prec) -> String

The precisions at which `local_elementary_divisors(A, p, prec)` computes the Smith normal form,
until one decides the result; `w` marks the word-size path.
"""
function snf_path(A, p, prec)
    k = S.snf_word_precision(p)
    tried = String[]
    for K in S.snf_precisions(p, prec)
        push!(tried, K <= k ? "w$K" : "$K")
        (K <= k ? S.snf_word(A, p, K) : S.snf_big(A, p, K))[2] == 0 && break
    end
    return join(tried, "→")
end

"""
    compare(label, A, p, prec) -> (old, new) serial times

Time the reference and the current Smith normal form of `A` modulo `p^prec`, serial and
threaded, interleaved, check that the results agree, and print a line. Beyond the word size
the `ZZRingElem` path alone (`Syntomic.snf_big`) is timed as well.
"""
function compare(label, A, p, prec)
    ref = ReferenceSNF.local_elementary_divisors(A, p, prec)
    big = prec > S.snf_word_precision(p)
    for threaded in (false, true)
        @assert local_elementary_divisors(A, p, prec; threaded=threaded) == ref
        big && @assert S.snf_big(A, p, prec; threaded=threaded) == ref
    end
    t = fill(Inf, 5)       # old, new, new threaded, big, big threaded
    for _ in 1:REPS
        t[1] = min(t[1], @elapsed ReferenceSNF.local_elementary_divisors(A, p, prec))
        t[2] = min(t[2], @elapsed local_elementary_divisors(A, p, prec))
        t[3] = min(t[3], @elapsed local_elementary_divisors(A, p, prec; threaded=true))
        big || continue
        t[4] = min(t[4], @elapsed S.snf_big(A, p, prec))
        t[5] = min(t[5], @elapsed S.snf_big(A, p, prec; threaded=true))
    end
    r, c = S.snf_dims(A)
    maxv = isempty(ref[1]) ? 0 : maximum(ref[1])
    bigcols = big ? @sprintf("%9.5f %9.5f", t[4], t[5]) : @sprintf("%9s %9s", "-", "-")
    @printf(
        "  %-22s %9s p=%-2d prec=%-5d v≤%-3d %-13s %9.5f %9.5f %9.5f %7.1fx %s\n",
        label,
        "$(r)×$(c)",
        p,
        prec,
        maxv,
        snf_path(A, p, prec),
        t[1],
        t[2],
        t[3],
        t[1] / t[2],
        bigcols
    )
    return t[1], t[2]
end

"""
    exponent_phase_matrices(X, i) -> (red, can1)

The `Z_p`-matrices whose exponents the exponent phase of the pipeline computes (this repeats the
construction in `Syntomic.exponent_phase`, which returns only the exponents).
"""
function exponent_phase_matrices(X, i)
    Xc = S.computing_ring(X, true)
    p, n = Xc.p, Xc.n
    prm = S.SyntomicParams(p, Xc.f, n, i)
    D = max(S.det_valuation_red(p, n, i), S.det_valuation_can1(n, i))
    R = S.GaloisRing(p, Xc.htilde, D + prm.nf + 3)
    ed = S.EisensteinData(R, Xc, prm.L)
    lam = S.LambdaData(ed, max(prm.nf, 2) - 2)
    fenv = S.FEnvelope(prm, R, ed, lam)
    red = S.zp_expand(S.reduction_matrix(fenv))
    can1 = S.zp_expand(S.can_nabla_matrix(fenv, i))
    return red, can1
end

"""
    pipeline_case(X, i)

Case 1 for `syntomic_cohomology(X, i)`.
"""
function pipeline_case(X, i)
    syntomic_cohomology(X, i)                                   # compile and warm up
    reps = max(3, REPS ÷ 2)
    total = mintime(() -> syntomic_cohomology(X, i), reps)
    serial = mintime(() -> syntomic_cohomology(X, i; threaded=false), reps)
    C = syntomic_complex(X, i; precision_policy=:adaptive)
    p = X.p
    T = S.target_precision(p, X.n, i)
    cases = Tuple{String,S.PMatrix,Int}[]
    try
        red0, can10 = exponent_phase_matrices(X, i)
        push!(cases, ("exponent phase: red", red0, red0.prec))
        push!(cases, ("exponent phase: can1", can10, can10.prec))
    catch err
        println("  (exponent phase matrices unavailable: ", sprint(showerror, err), ")")
    end
    for key in (:red, :can1, :nablaN)
        A = C.intermediates[key]
        push!(cases, ("Steps 7/8/10: $key", A, A.prec))
    end
    push!(cases, ("syn0", C.syn0, min(C.syn0.prec, T + 1)))
    push!(cases, ("syn1", C.syn1, min(C.syn1.prec, T + 1)))
    @printf(
        "syntomic_cohomology(%s, %d): N = %d, total %.3f s threaded, %.3f s serial\n",
        X,
        i,
        C.N,
        total,
        serial
    )
    header()
    sum_old, sum_new = 0.0, 0.0
    for (label, A, prec) in cases
        old, new = compare(label, A.M, p, prec)
        sum_old += old
        sum_new += new
    end
    # the serial total with the reference elimination is estimated as serial - new + old
    @printf(
        "  sum of the serial SNF times: old %.4f s (%.1f%% of the serial total), new %.4f s (%.1f%%)\n\n",
        sum_old,
        100 * sum_old / (serial - sum_new + sum_old),
        sum_new,
        100 * sum_new / serial
    )
end

"""Print the column headers of the comparison tables."""
function header()
    @printf(
        "  %-22s %9s %-4s %-10s %-5s %-13s %9s %9s %9s %8s %9s %9s\n",
        "matrix",
        "size",
        "",
        "",
        "",
        "precisions",
        "old",
        "new",
        "new thr.",
        "speedup",
        "big",
        "big thr."
    )
end

"""
    product_matrix(rng, p, a, r, c) -> Matrix{ZZRingElem}

`X·diag(p^a)·Y` for random `X` (`r × r`) and `Y` (`r × c`) with small entries.
"""
function product_matrix(rng, p, a, r, c)
    X = matrix(ZZ, [ZZ(rand(rng, -9:9)) for _ in 1:r, _ in 1:r])
    Y = matrix(ZZ, [ZZ(rand(rng, -9:9)) for _ in 1:r, _ in 1:c])
    return Matrix{ZZRingElem}(X * diagonal_matrix(ZZ, [ZZ(p)^e for e in a]) * Y)
end

"""
    random_case(rng, p, prec; r=200, c=400)

Case 2: a uniformly random `r × c` matrix modulo `p^prec`, and products `X·diag(p^a)·Y` with
valuations `a ≤ 12`, and, beyond the word size, with three valuations `a = k + 3` above the
word-size precision `k`, so that the word-size attempt does not decide the result.
"""
function random_case(rng, p, prec; r=200, c=400)
    m = ZZ(p)^prec
    compare("uniform", [rand(rng, ZZ(0):(m - 1)) for _ in 1:r, _ in 1:c], p, prec)
    a = rand(rng, 0:min(prec - 1, 12), r)
    compare("X·diag(p^a)·Y", product_matrix(rng, p, a, r, c), p, prec)
    k = S.snf_word_precision(p)
    prec > k + 3 || return nothing
    a[1:3] .= k + 3
    compare("X·diag(p^a)·Y, a ≥ k", product_matrix(rng, p, a, r, c), p, prec)
    return nothing
end

println("threads: ", Threads.nthreads(), ", repetitions: ", REPS)
println("\n1. Smith normal forms of syntomic_cohomology(zpn(2, 2), i)\n")
for i in (30, 40)
    pipeline_case(zpn(2, 2), i)
end
println("2. random 200×400 matrices\n")
header()
rng = Xoshiro(20260925)
for (p, prec) in ((2, 20), (3, 13), (2, 62), (3, 39), (2, 100), (3, 64), (2, 400))
    random_case(rng, p, prec)
end
println("\n3. far beyond the word size (precisions as in the exponent phase)\n")
header()
for (p, prec, top, n) in ((2, 1564, 70, 80), (2, 3000, 90, 120), (3, 600, 45, 80))
    a = rand(rng, 0:12, n)
    a[1:2] .= top
    compare("X·diag(p^a)·Y, a ≤ $top", product_matrix(rng, p, a, n, n), p, prec)
end
