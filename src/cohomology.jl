# The syntomic complex and its cohomology.
#
# [AKN24, §7.1, Return, p. 57]: syn^0 = (can - φ ; N^{≥i}∇) and syn^1 = (∇, -can + φ^∇), defining a complex
#     0 → Z_p^{f(in-1)} --syn^0--> Z_p^{2f(in-1)} --syn^1--> Z_p^{f(in-1)} → 0
# quasi-isomorphic to Z_p(i)(O_K/ϖ^n) ([AKN24, Thm. 2.27, p. 13]). Its cohomology is computed from
# the elementary divisors of syn^0 and syn^1 ([AKN24, Lemma 7.1, p. 57]): both differentials have rank
# f(in-1) ([AKN24, Lemma 7.2, p. 58]), H^0 = 0, H^1 is the torsion of coker(syn^0) and H^2 = coker(syn^1).

"""
    SyntomicComplex

The complex `Z_p^{f(in-1)} --syn0--> Z_p^{2f(in-1)} --syn1--> Z_p^{f(in-1)}` computing
`Z_p(i)(O_K/ϖ^n)` ([AKN24, §7.1, Return, p. 57]).

Fields: `ring`, `i`, the working precision `N`, the `Z_p`-matrices `syn0` and `syn1` (each known
modulo `p^prec`), the p-exponents `exponents` of the matrices inverted in Steps 7, 8 and 10, the
precision `policy` that chose `N` (`:fixed` if `N` was given), and `intermediates` (the
`Z_p`-matrices of Steps 5–10, keyed by `:red`, `:redN`, `:nablaOK`, `:can0`, `:can1`, `:phi`,
`:nablaR`, `:nablaN`, `:canmphi` and `:canmphiN`).

Like a matrix factorization of `LinearAlgebra`, the complex destructures into its two
differentials: `syn0, syn1 = syntomic_complex(X, i)` (iteration yields `C.syn0`, then `C.syn1`).
"""
struct SyntomicComplex
    ring::FiniteChainRing
    i::Int
    N::Int
    syn0::PMatrix
    syn1::PMatrix
    exponents::NamedTuple{(:red, :can1, :nablaN),NTuple{3,Int}}
    policy::Symbol
    intermediates::Dict{Symbol,Any}
end

"""
    iterate(C::SyntomicComplex[, state])

Destructuring `syn0, syn1 = C` of a [`SyntomicComplex`](@ref): yields `C.syn0`, then `C.syn1`.
"""
Base.iterate(C::SyntomicComplex) = (C.syn0, Val(:syn1))
Base.iterate(C::SyntomicComplex, ::Val{:syn1}) = (C.syn1, Val(:done))
Base.iterate(::SyntomicComplex, ::Val{:done}) = nothing
Base.length(::SyntomicComplex) = 2
Base.eltype(::Type{SyntomicComplex}) = PMatrix

function Base.show(io::IO, C::SyntomicComplex)
    m = size(C.syn0, 2)
    return print(
        io,
        "Syntomic complex Z_$(C.ring.p)^$m → Z_$(C.ring.p)^$(2m) → Z_$(C.ring.p)^$m for i = $(C.i) of $(C.ring) " *
        "(precisions $(C.syn0.prec), $(C.syn1.prec))",
    )
end

"""
    SyntomicCohomology

The cohomology of `Z_p(i)(O_K/ϖ^n)`: `h1` and `h2` are the sorted exponent lists `[a_1, a_2, …]`
with `H^k ≅ ⊕_j Z/p^{a_j}` (`H^0 = 0`). Also records the working precision `N`, the certified
precisions `(prec0, prec1)` of the two differentials, the precision policy used, the number of
restarts, and timings.
"""
struct SyntomicCohomology
    ring::FiniteChainRing
    i::Int
    h1::Vector{Int}
    h2::Vector{Int}
    N::Int
    prec0::Int
    prec1::Int
    policy::Symbol
    restarts::Int
    timings::Dict{Symbol,Float64}
end

function Base.show(io::IO, H::SyntomicCohomology)
    fmt(v) = isempty(v) ? "0" : join(("Z/$(H.ring.p)^$a" for a in v), " ⊕ ")
    return print(
        io, "H*(Z_$(H.ring.p)($(H.i))($(H.ring))): H^1 = $(fmt(H.h1)), H^2 = $(fmt(H.h2))"
    )
end

"""
    certified_elementary_divisors(A::PMatrix, p, rank, bound) -> Vector{Int}

The p-adic valuations of the `rank` elementary divisors of the differential `A`, certified by
[AKN24, Lemma 7.1, p. 57] and [AKN24, Remark 7.3, p. 58]: the Smith normal form is computed modulo
`p^M` with `M = min(A.prec, bound + 1)`, where `bound` is an a priori bound for the exponents
(the target precision of [AKN24, Lemma 7.2]); all `rank` diagonal entries must be nonzero modulo `p^M`.
Throws [`PrecisionExhausted`](@ref) otherwise.
"""
function certified_elementary_divisors(
    A::PMatrix, p::Integer, rank::Integer, bound::Integer; threaded::Bool=false
)
    rank == 0 && return Int[]
    M = min(A.prec, bound + 1)
    M >= 1 ||
        throw(PrecisionExhausted("a syntomic differential is known to precision $(A.prec)"))
    vals, nzero = local_elementary_divisors(A.M, p, M; threaded=threaded)
    (nzero == 0 && length(vals) == rank) || throw(
        PrecisionExhausted(
            "$nzero elementary divisor(s) of a syntomic differential are ≡ 0 mod $p^$M; the precision is insufficient",
        ),
    )
    return vals
end

"""
    syntomic_cohomology(C::SyntomicComplex; check=true, threaded=true) -> SyntomicCohomology

The cohomology of the syntomic complex `C` ([AKN24, §7.1, Return, p. 57]; [AKN24, Lemma 7.1, p. 57]).
With `check = true`, verifies `Σ h1 - Σ h2 = f·i·(n-1)` ([AKN24, Prop. 2.18, p. 9]) and throws a
[`CertificateError`](@ref) otherwise.
"""
function syntomic_cohomology(
    C::SyntomicComplex;
    check::Bool=true,
    threaded::Bool=true,
    policy::Symbol=C.policy,
    restarts::Integer=0,
    timings::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
)
    X = C.ring
    p, f, n, i = X.p, X.f, X.n, C.i
    m = f * (n * i - 1)
    T = target_precision(p, n, i)
    t0 = time()
    v0, v1 = concurrently(
        () -> certified_elementary_divisors(C.syn0, p, m, T; threaded=threaded),
        () -> certified_elementary_divisors(C.syn1, p, m, T; threaded=threaded);
        threaded=threaded,
    )
    timings[:snf] = time() - t0
    h1 = sort(filter(>(0), v0))
    h2 = sort(filter(>(0), v1))
    if check && sum(h1; init=0) - sum(h2; init=0) != f * i * (n - 1)
        throw(
            CertificateError(
                "Σ h1 - Σ h2 = $(sum(h1; init=0) - sum(h2; init=0)), expected f·i·(n-1) = $(f * i * (n - 1)) ([AKN24, Prop. 2.18])",
            ),
        )
    end
    return SyntomicCohomology(
        X, i, h1, h2, C.N, C.syn0.prec, C.syn1.prec, policy, restarts, timings
    )
end
