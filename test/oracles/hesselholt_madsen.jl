# Hesselholt–Madsen oracle: the K-groups of truncated polynomial algebras over F_q.
#
# This file does NOT depend on Syntomic.jl. It uses only Base (no Nemo, Hecke or Oscar).
#
# What it computes. `hm_exponents(p, f, n, i)` returns the exponent list of
#
#     K_{2i-1}(F_q[z]/(z^n), (z); Z_p) = W_{ni}(F_q) / V_n W_i(F_q),   q = p^f,
#
# where W_m denotes big Witt vectors with truncation set {1, ..., m} and V_n is the
# Verschiebung. This is Hesselholt–Madsen, "Cyclic polytopes and the K-theory of
# truncated polynomial algebras", Invent. Math. 130 (1997). The relative even groups
# vanish. K_{2i-1}(F_q) = Z/(q^i - 1) is prime to p, so this is also
# K_{2i-1}(F_q[z]/z^n; Z_p), and K_{2i-2}(F_q[z]/z^n; Z_p) = 0 for i >= 2.
#
# Use in the package tests. Let E be Eisenstein of degree e over W(F_q) (e.g. z^e + p),
# and let n <= e. Mod z^n, E = p * (unit), so O_K/pi^n = W[z]/(E, z^n) = F_q[z]/z^n. The
# syntomic answer must then be H^1(Z_p(i)) = hm_exponents(p, f, n, i) and H^2 = 0.
#
# ---------------------------------------------------------------------------------------
# Derivation of the p-typical decomposition
# ---------------------------------------------------------------------------------------
# (1) For a Z_(p)-algebra k and a truncation set S, idempotents split the big Witt
#     vectors as W_S(k) = prod_{d in S, p not dividing d} W_{l_S(d)}(k), where the
#     factors are p-typical Witt vectors of length l_S(d) = #{j >= 0 : d p^j in S}.
#     For S = {1..m}, l_m(d) = #{j >= 0 : d p^j <= m}.
# (2) Write n = p^a * n' with p not dividing n'. Then V_n = V_{n'} o V_{p^a}, with
#     W_i --V_{p^a}--> W_{p^a i} --V_{n'}--> W_{ni}.
#     * V_{p^a} preserves the d-factor, where it is the p-typical V^a:
#       W_{l_i(d)} -> W_{l_{p^a i}(d)}. If d <= i then l_{p^a i}(d) = l_i(d) + a; if
#       i < d <= p^a i then l_i(d) = 0 and l_{p^a i}(d) <= a. V^a is injective, and its
#       image V^a W_l inside W_{l+a} is the kernel of the restriction W_{l+a} -> W_a (the
#       first a components). Hence the d-factor of W_{p^a i}/V_{p^a} W_i is
#       W_{min(a, l_{p^a i}(d))}.
#     * V_{n'} (n' a unit in Z_(p)) satisfies F_{n'} V_{n'} = n', which is invertible. It
#       maps the d-factor of W_{p^a i} isomorphically onto the (n'd)-factor of W_{ni}; the
#       lengths agree, since l_{ni}(n'd) = l_{p^a i}(d). The factors d' with n' not
#       dividing d' are not in its image.
# (3) Therefore, for 1 <= d <= ni with p not dividing d,
#         h(d) = l_{ni}(d)              if n' does not divide d,
#         h(d) = min(a, l_{ni}(d))      if n' divides d,
#     and W_{ni}(F_q)/V_n W_i(F_q) = sum_d W_{h(d)}(F_q). As an abelian group each
#     W_h(F_q) = W(F_q)/p^h is (Z/p^h)^f.
#     Check: the total order is q^{ni} / q^i, so sum_d h(d) = i(n-1) (times f in Z_p-length).
#
# ---------------------------------------------------------------------------------------
# Validation record (run 2026-09-25; every case agrees)
# ---------------------------------------------------------------------------------------
# (A) The SageMath implementation of [AKN] (antieau/kzpn, Sage 10.3, f = 1, E = z^e + p;
#     SHA-256 of prismatic_envelope.sage
#     c0e256cf30ada5a26a2c37fcf519518796720fa0d9f9cdcacbc97b66a9bb313e), at the working
#     precision of [AKN, §7.1, Step 1] plus 5. H^2 = [] in every case; H^1:
#       (p,e,n,i) = (2,3,3,2) [1,3]        (2,3,3,3) [1,1,4]      (2,4,4,2) [1,1,2,2]
#                   (3,2,2,2) [2]          (3,3,3,2) [1,1,1,1]    (5,2,2,3) [1,2]
#                   (2,2,2,2) [1,1]        (2,2,2,3) [1,1,1]      (2,2,2,4) [1,1,1,1]
#                   (3,2,2,3) [1,2]        (3,2,2,4) [1,1,2]      (2,3,3,4) [1,1,2,4]
#                   (2,4,4,3) [1,1,1,2,2,2]                       (3,3,3,3) [1,1,1,1,1,1]
#                   (5,3,3,2) [1,1,2]      (2,5,5,2) [1,1,2,4]    (2,6,6,2) [1,1,1,1,2,4]
#                   (3,4,4,2) [1,1,2,2]    (2,3,2,3) [1,1,1]      (2,5,3,2) [1,3]
#                   (2,4,3,3) [1,1,4]      (3,5,4,2) [1,1,2,2]    (2,6,6,3) [1,1,1,1,1,1,2,2,5]
#                   (3,6,6,2) [1,1,1,1,1,1,1,3]                   (2,8,8,2) [1,1,1,1,2,2,3,3]
#                   (5,5,5,2) [1,1,1,1,1,1,1,1]
#                   i = 1: (2,3,3,1) [2]  (3,2,2,1) [1]  (2,4,4,1) [1,2]
#     The runs cover n = p^a (a > 0), n prime to p, and mixed n = 6 = 2*3 with p = 2, 3,
#     as well as n < e.
# (B) [AKN, Fig. 1, p. 4], F_2[z]/z^3: K3 = Z/2+Z/8, K5 = (Z/2)^2+Z/16,
#     K7 = (Z/2)^2+Z/4+Z/16, K9 = (Z/2)^2+(Z/4)^2+Z/16 (2-parts), and K1 = Z/4.
# (C) sigma-patched Sage runs at f = 2 (test/fixtures/sage_fq.json; lower trust):
#     (p,n,i) = (2,2,2) [1,1,1,1]  (2,2,3) [1 x 6]  (2,3,2) [1,1,3,3]  (2,3,3) [1,1,1,1,4,4]
#               (2,4,2) [1,1,1,1,2,2,2,2]  (3,2,2) [2,2]  (3,2,3) [1,1,2,2]  (3,2,4) [1,1,1,1,2,2].
# (D) i = 1 agrees with the brute-force oracle brute_force_units.jl for E = z^e + p, n <= e.
# (E) sum(hm_exponents(p, f, n, i)) == f * i * (n - 1) for all p in {2,3,5,7},
#     1 <= n, i <= 12, f in 1:3.

"""
    hm_exponents(p, f, n, i) -> Vector{Int}

Return the sorted exponent list of the Hesselholt–Madsen group
`K_{2i-1}(F_q[z]/(z^n), (z); Z_p) = W_{ni}(F_q)/V_n W_i(F_q)`, with `q = p^f`. This
function is independent of Syntomic.jl.

It uses the p-typical decomposition derived in the header of this file. Put
`n = p^a n′` with `p ∤ n′`, and let `ℓ(d) = #{j ≥ 0 : d p^j ≤ ni}`. The group is then
`⊕_{1 ≤ d ≤ ni, p ∤ d} (Z/p^{h(d)})^f`, where `h(d) = ℓ(d)` if `n′ ∤ d` and
`h(d) = min(a, ℓ(d))` if `n′ | d`. The exponents sum to `f·i·(n-1)`, and
`n = 1` gives `Int[]`.

The even relative groups vanish. So for an Eisenstein `E` of degree `e ≥ n` over
`W(F_q)`, the syntomic cohomology of `O_K/π^n = F_q[z]/z^n` satisfies
`H^1(Z_p(i)) = hm_exponents(p, f, n, i)` and `H^2(Z_p(i)) = 0` for `i ≥ 1`.

# Examples
```julia-repl
julia> hm_exponents(2, 1, 3, 2)      # K_3(F_2[z]/z^3; Z_2) = Z/2 ⊕ Z/8   (Fig. 1)
2-element Vector{Int64}:
 1
 3

julia> hm_exponents(2, 1, 3, 3)      # K_5: (Z/2)^2 ⊕ Z/16  (Fig. 1; KZPN (2,3,3,3))
3-element Vector{Int64}:
 1
 1
 4

julia> hm_exponents(2, 1, 4, 2)      # KZPN (p,e,n,i) = (2,4,4,2)
4-element Vector{Int64}:
 1
 1
 2
 2

julia> hm_exponents(3, 1, 2, 2), hm_exponents(3, 1, 3, 2), hm_exponents(5, 1, 2, 3)
([2], [1, 1, 1, 1], [1, 2])

julia> hm_exponents(2, 2, 3, 2)      # f = 2: each summand appears twice
4-element Vector{Int64}:
 1
 1
 3
 3
```
"""
function hm_exponents(p::Int, f::Int, n::Int, i::Int)::Vector{Int}
    _hm_isprime(p) || throw(ArgumentError("p = $p is not a prime"))
    f >= 1 || throw(ArgumentError("f must be ≥ 1 (got f = $f)"))
    n >= 1 || throw(ArgumentError("n must be ≥ 1 (got n = $n)"))
    i >= 1 || throw(ArgumentError("i must be ≥ 1 (got i = $i)"))
    exps = Int[]
    for (_, h) in hm_components(p, n, i)
        append!(exps, fill(h, f))
    end
    return sort!(exps)
end

"""
    hm_components(p, n, i) -> Vector{Tuple{Int,Int}}

Return the pairs `(d, h(d))` with `h(d) > 0` from the p-typical decomposition
`W_{ni}(k)/V_n W_i(k) ≅ ⊕_{p ∤ d ≤ ni} W_{h(d)}(k)`, where `k` is a perfect field of
characteristic `p`. This is the `f`-independent core of [`hm_exponents`](@ref), and it
is independent of Syntomic.jl.

# Example
```julia-repl
julia> hm_components(2, 6, 2)        # n = 6 = 2 * 3: a = 1, n′ = 3
6-element Vector{Tuple{Int64, Int64}}:
 (1, 4)
 (3, 1)
 (5, 2)
 (7, 1)
 (9, 1)
 (11, 1)
```
"""
function hm_components(p::Int, n::Int, i::Int)::Vector{Tuple{Int,Int}}
    a = 0
    nprime = n
    while nprime % p == 0
        nprime = div(nprime, p)
        a += 1
    end
    top = n * i
    out = Tuple{Int,Int}[]
    for d in 1:top
        d % p == 0 && continue
        l = 0                                   # l = #{j >= 0 : d p^j <= ni}
        t = d
        while t <= top
            l += 1
            t = Base.checked_mul(t, p)
        end
        h = d % nprime == 0 ? min(a, l) : l
        h > 0 && push!(out, (d, h))
    end
    return out
end

"""
    HM_SAGE_VALIDATION

The validation cases `(p, e, n, i, h1)` from the original KZPN Sage code, at `f = 1` with
`E = z^e + p` and `n ≤ e`, where Sage returned `H^2 = []`. Each satisfies
`hm_exponents(p, 1, n, i) == h1`.
"""
const HM_SAGE_VALIDATION = [
    (2, 3, 3, 2, [1, 3]),
    (2, 3, 3, 3, [1, 1, 4]),
    (2, 4, 4, 2, [1, 1, 2, 2]),
    (3, 2, 2, 2, [2]),
    (3, 3, 3, 2, [1, 1, 1, 1]),
    (5, 2, 2, 3, [1, 2]),
    (2, 2, 2, 2, [1, 1]),
    (2, 2, 2, 3, [1, 1, 1]),
    (2, 2, 2, 4, [1, 1, 1, 1]),
    (3, 2, 2, 3, [1, 2]),
    (3, 2, 2, 4, [1, 1, 2]),
    (2, 3, 3, 4, [1, 1, 2, 4]),
    (2, 4, 4, 3, [1, 1, 1, 2, 2, 2]),
    (3, 3, 3, 3, [1, 1, 1, 1, 1, 1]),
    (5, 3, 3, 2, [1, 1, 2]),
    (2, 5, 5, 2, [1, 1, 2, 4]),
    (2, 6, 6, 2, [1, 1, 1, 1, 2, 4]),
    (3, 4, 4, 2, [1, 1, 2, 2]),
    (2, 3, 2, 3, [1, 1, 1]),
    (2, 5, 3, 2, [1, 3]),
    (2, 4, 3, 3, [1, 1, 4]),
    (3, 5, 4, 2, [1, 1, 2, 2]),
    (2, 6, 6, 3, [1, 1, 1, 1, 1, 1, 2, 2, 5]),
    (3, 6, 6, 2, [1, 1, 1, 1, 1, 1, 1, 3]),
    (2, 8, 8, 2, [1, 1, 1, 1, 2, 2, 3, 3]),
    (5, 5, 5, 2, [1, 1, 1, 1, 1, 1, 1, 1]),
    (2, 3, 3, 1, [2]),
    (3, 2, 2, 1, [1]),
    (2, 4, 4, 1, [1, 2]),
]

"""
    HM_F2_SIGMA_PATCHED_VALIDATION

The `f = 2` cases `(p, f, n, i, h1)` from the σ-patched Sage fixtures
(`test/fixtures/sage_fq.json`, lower trust), where `H^2 = []`. Each satisfies `hm_exponents(p, f, n, i) == h1`.
"""
const HM_F2_SIGMA_PATCHED_VALIDATION = [
    (2, 2, 2, 2, [1, 1, 1, 1]),
    (2, 2, 2, 3, [1, 1, 1, 1, 1, 1]),
    (2, 2, 3, 2, [1, 1, 3, 3]),
    (2, 2, 3, 3, [1, 1, 1, 1, 4, 4]),
    (2, 2, 4, 2, [1, 1, 1, 1, 2, 2, 2, 2]),
    (3, 2, 2, 2, [2, 2]),
    (3, 2, 2, 3, [1, 1, 2, 2]),
    (3, 2, 2, 4, [1, 1, 1, 1, 2, 2]),
]

function _hm_isprime(p::Int)
    p < 2 && return false
    d = 2
    while d * d <= p
        p % d == 0 && return false
        d += 1
    end
    return true
end
