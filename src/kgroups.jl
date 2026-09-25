# K-groups from syntomic cohomology.
#
# For R = O_K/ϖ^n and i ≥ 1: K_{2i-1}(R; Z_p) ≅ H^1(Z_p(i)(R)), and K_{2i-2}(R; Z_p) ≅ H^2(Z_p(i)(R))
# for i ≥ 2, while K_0(R; Z_p) ≅ Z_p ([AKN, Cor. 2.16, p. 9]; [AKN, §1, p. 3]). The prime-to-p part of
# K_*(R) is that of K_*(F_q): K_{2i-1}(F_q) = Z/(q^i - 1) and K_{2i-2}(F_q) = 0 for i ≥ 2.

"""
    k_groups(X::FiniteChainRing, i; integral=false, kwargs...) -> NamedTuple

The K-groups `K_{2i-1}(X)` and `K_{2i-2}(X)` of the finite chain ring `X = O_K/ϖ^n` as Hecke
`FinGenAbGroup`s, computed from [`syntomic_cohomology`](@ref) (keyword arguments are passed on).

With `integral = false` (default) these are the p-adic K-groups `K_*(X; Z_p)`; with
`integral = true` the integral groups, adding the prime-to-p part `Z/(q^i - 1)` of
`K_{2i-1}(F_q)`. For `i = 1`, `K_0 = Z` (its p-completion `Z_p` when `integral = false`, which
is returned as the group `Z`).

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
