# K-groups from syntomic cohomology.
#
# For R = O_K/ϖ^n and i ≥ 1: K_{2i-1}(R; Z_p) ≅ H^1(Z_p(i)(R)), and K_{2i-2}(R; Z_p) ≅ H^2(Z_p(i)(R))
# for i ≥ 2, while K_0(R; Z_p) ≅ Z_p ([AKN, Cor. 2.16, p. 9]; [AKN, §1, p. 3]). The prime-to-p part of
# K_*(R) is that of K_*(F_q): K_{2i-1}(F_q) = Z/(q^i - 1) and K_{2i-2}(F_q) = 0 for i ≥ 2
# ([AKN, §1.1, (1), p. 2]).

"""
    k_groups(X::FiniteChainRing, i; integral=false, kwargs...) -> NamedTuple

The K-groups `K_{2i-1}(X)` and `K_{2i-2}(X)` of the finite chain ring `X = O_K/ϖ^n` as Hecke
`FinGenAbGroup`s, computed from [`syntomic_cohomology`](@ref) (keyword arguments are passed on):
`K_{2i-1}(X; Z_p) ≅ H^1(Z_p(i)(X))` for `i ≥ 1` and `K_{2i-2}(X; Z_p) ≅ H^2(Z_p(i)(X))` for
`i ≥ 2` ([AKN, Cor. 2.16, p. 9]).

With `integral = false` (default) these are the p-adic K-groups `K_*(X; Z_p)`; with
`integral = true` the integral groups, adding the prime-to-p part `Z/(q^i - 1)` of
`K_{2i-1}(F_q)` ([AKN, §1.1, (1), p. 2]). For `i = 1`, `K_0 = Z` (its p-completion `Z_p` when
`integral = false`, which is returned as the group `Z`).

Returns `(; odd = K_{2i-1}, even = K_{2i-2}, degrees = (2i-1, 2i-2), cohomology = H)`.
"""
function k_groups(X::FiniteChainRing, i::Integer; integral::Bool=false, kwargs...)
    H = syntomic_cohomology(X, i; kwargs...)
    p = ZZ(X.p)
    odd_orders = ZZRingElem[p^a for a in H.h1]
    if integral
        pushfirst!(odd_orders, ZZ(p)^(X.f * i) - 1)
    end
    odd = abelian_group(isempty(odd_orders) ? ZZRingElem[1] : odd_orders)
    even = if i == 1
        abelian_group([0])
    else
        abelian_group(isempty(H.h2) ? ZZRingElem[1] : ZZRingElem[p^a for a in H.h2])
    end
    return (; odd=odd, even=even, degrees=(2 * i - 1, 2 * i - 2), cohomology=H)
end

# ---------------------------------------------------------------------------------------------
# Tables of K-groups
# ---------------------------------------------------------------------------------------------

"""
    KTheoryRow

The element type of a [`KTheoryTable`](@ref): a named tuple
`(degree = r, exponents = [a_1, …], prime_to_p_order = m)` meaning
`K_r ≅ Z/m ⊕ ⊕_j Z/p^{a_j}`, where `exponents === nothing` stands for `K_0` (`Z`, or `Z_p` for
p-adic K-theory).
"""
const KTheoryRow = NamedTuple{
    (:degree, :exponents, :prime_to_p_order),Tuple{Int,Union{Nothing,Vector{Int}},BigInt}
}

"""
    KTheoryTable <: AbstractVector

The K-groups `K_r(X)` of a finite chain ring `X = O_K/ϖ^n` for a range of degrees, as returned by
[`k_theory_table`](@ref): a vector, sorted by degree, of the rows
`(degree = r, exponents = [a_1, …], prime_to_p_order = m)` meaning `K_r ≅ Z/m ⊕ ⊕_j Z/p^{a_j}`
([AKN, Cor. 2.16, p. 9]); `exponents === nothing` stands for `K_0`. For p-adic K-theory
(`integral = false`) every `m` is `1`; for integral K-theory `m = q^i - 1` in degree `2i - 1`
([AKN, §1.1, (1), p. 2]).

Rows are accessed as for any vector (`tab[2].exponents`, `[r.degree for r in tab]`). The
`text/plain` display shows the primary decomposition of each group, such as
`K_3(Z/4) ≅ Z/8 ⊕ Z/3`: the cyclic prime-to-p part `Z/(q^i - 1)` is split into its primary
components (see [`group_summands`](@ref)).

Fields: `ring`, `integral` and `rows`.
"""
struct KTheoryTable <: AbstractVector{KTheoryRow}
    ring::FiniteChainRing
    integral::Bool
    rows::Vector{KTheoryRow}
end

Base.size(tab::KTheoryTable) = size(tab.rows)
Base.getindex(tab::KTheoryTable, k::Int) = tab.rows[k]
Base.IndexStyle(::Type{KTheoryTable}) = IndexLinear()

"""
    cyclic_primary_decomposition(m; max_digits=50) -> Vector{Tuple{BigInt,Int}}

The primary decomposition `Z/m ≅ ⊕ Z/ℓ^k` of a finite cyclic group, as the pairs `(ℓ, k)` of
the prime factorization of `m ≥ 1`, sorted by `ℓ` (empty for `m = 1`).

The factorization is bounded so that it always returns quickly: after trial division by the
first `10^5` primes, the remaining cofactor `c` is factored completely only if it is a probable
prime or has at most `max_digits` decimal digits. Otherwise it is returned as a single pair
`(c, 1)` with `c` composite, so that `Z/c` is a cyclic summand that is not split further.
"""
function cyclic_primary_decomposition(m::Integer; max_digits::Integer=50)
    m >= 1 || throw(ArgumentError("the order of a cyclic group must be positive, got $m"))
    isone(m) && return Tuple{BigInt,Int}[]
    small, _ = Nemo.factor_trial_range(ZZ(m))
    out = [(BigInt(l), Int(k)) for (l, k) in small]
    c = ZZ(m)
    for (l, k) in small
        c = divexact(c, l^k)
    end
    if isone(c)
        nothing
    elseif Nemo.is_probable_prime(c)
        push!(out, (BigInt(c), 1))
    elseif ndigits(c) <= max_digits
        append!(out, [(BigInt(l), Int(k)) for (l, k) in Nemo.factor(c)])
    else
        push!(out, (BigInt(c), 1))
    end
    return sort!(out)
end

"""
    group_summands(row::KTheoryRow, p) -> Vector{Tuple{BigInt,Int,Int}}

The primary decomposition of the finite group `K_r ≅ Z/m ⊕ ⊕_j Z/p^{a_j}` of a row of a
[`KTheoryTable`](@ref), as triples `(ℓ, k, μ)` meaning `(Z/ℓ^k)^μ`: first the p-part by
increasing `k`, then the primary components of `Z/m` by increasing `ℓ` (see
[`cyclic_primary_decomposition`](@ref)). Empty for the zero group; `K_0`
(`exponents === nothing`) is not finite and throws an `ArgumentError`.
"""
function group_summands(row::KTheoryRow, p::Integer)
    row.exponents === nothing &&
        throw(ArgumentError("K_$(row.degree) is not a finite group"))
    out = Tuple{BigInt,Int,Int}[]
    for a in sort(unique(row.exponents))
        push!(out, (BigInt(p), a, count(==(a), row.exponents)))
    end
    for (l, k) in cyclic_primary_decomposition(row.prime_to_p_order)
        push!(out, (l, k, 1))
    end
    return out
end

# `Z/ℓ^k`, with ℓ^k written out when it is small.
function _cyclic_group_string(l::BigInt, k::Int)
    (k == 1 || l^k < 1000) && return "Z/$(l^k)"
    return "Z/$l^$k"
end

"""
    group_string(row::KTheoryRow, p; integral=true) -> String

The primary decomposition of the group of a row of a [`KTheoryTable`](@ref) as text, such as
`"Z/8 ⊕ Z/3"` or `"(Z/2)^2 ⊕ Z/2^9"`; `"0"` for the zero group and `"Z"` (`"Z_p"` if
`integral = false`) for `K_0`.
"""
function group_string(row::KTheoryRow, p::Integer; integral::Bool=true)
    row.exponents === nothing && return integral ? "Z" : "Z_$p"
    parts = map(group_summands(row, p)) do (l, k, mu)
        g = _cyclic_group_string(l, k)
        mu == 1 ? g : "($g)^$mu"
    end
    return isempty(parts) ? "0" : join(parts, " ⊕ ")
end

function Base.show(io::IO, ::MIME"text/plain", tab::KTheoryTable)
    X = tab.ring
    name = isempty(X.label) ? "R" : X.label
    ringtext = isempty(X.label) ? "R = $X" : X.label
    print(io, tab.integral ? "K-groups of " : "$(X.p)-adic K-groups of ", ringtext)
    isempty(tab) && return print(io, ": none")
    print(io, ":")
    coefficients = tab.integral ? "" : "; Z_$(X.p)"
    lhs = ["K_$(r.degree)($name$coefficients)" for r in tab]
    width = maximum(length, lhs)
    for (l, r) in zip(lhs, tab)
        print(
            io, "\n  ", rpad(l, width), " ≅ ", group_string(r, X.p; integral=tab.integral)
        )
    end
    return nothing
end
