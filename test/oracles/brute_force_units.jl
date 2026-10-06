# Brute-force oracle for weight i = 1: the principal unit group of O_K/pi^n.
#
# This file does NOT depend on Syntomic.jl. It uses only Base (no Nemo, Hecke or Oscar),
# so it can check the package's weight-1 results by a completely separate route.
#
# What it computes. Let W = W(F_q) = Z_p[x]/(htilde), q = p^f, and let
# O_K = W[z]/(E) for an Eisenstein polynomial E in W[z], with uniformizer pi = z.
# `principal_units_exponents` returns the exponent list [a_1 <= a_2 <= ...] of the
# finite abelian p-group
#
#     U^1/U^n = (1 + pi*O_K)/(1 + pi^n*O_K)  =  sum_j Z/p^{a_j},
#
# found by enumerating all q^(n-1) of its elements.
#
# Why it matters. For i = 1 the syntomic complex gives H^1(Z_p(1)(O_K/pi^n)) =
# K_1(O_K/pi^n; Z_p) = (R^x) completed at p = U^1/U^n, and H^2 = 0. Its order is q^(n-1).
#
# Method.
#   * O_K is free over W with basis 1, z, ..., z^(e-1), and
#     v_pi(sum_j a_j z^j) = min_j (e*v_p(a_j) + j), because the terms have distinct
#     pi-valuations mod e. Hence pi^n*O_K = sum_j p^(m_j) W z^j with
#     m_j = ceil((n - j)/e), and O_K/pi^n = sum_j (W/p^(m_j)) z^j.
#   * Products are computed in O_K/p^M = (W/p^M)[z]/(E) with M = ceil(n/e) = m_0.
#     This is legitimate because p^M*O_K = pi^(e*M)*O_K lies inside pi^n*O_K. The
#     reduction uses z^e = -E_e^(-1) (E_0 + ... + E_(e-1) z^(e-1)); the leading
#     coefficient E_e is a unit, and E_e^(-1) mod p^M comes from Newton iteration.
#   * U^1 consists of the classes with a_0 = 1 mod p. Each element x gets its order
#     p^(o(x)) by following the table x -> x^p, and then
#     |G[p^k]| = #{x : o(x) <= k} = p^(sum_j min(a_j, k)) determines the partition.
#
# Validation. These cases were run when the file was written (2026-09-25) and all
# agree. The Conway generators are x^2+x+1 for F_4, x^2+2x+2 for F_9 and x^3+x+1 for F_8.
#   * f = 1, E = z + 2: n = 2, 3, 4 gives [1], [1,1], [1,2]. This matches [AKN24, Appendix A] K_1 of
#     Z/2^n, and also K_1 of Z/32 = [1,3].
#   * f = 1, E = z + p: (Z/p^n)^x has p-part Z/p^(n-1) for p = 3, 5, 7 ([AKN24, Appendix A], K_1 rows).
#   * Independent brute-force computations, with a = omega the Conway generator:
#       p=2 f=2 z+2 and z+2a, n=2..5: [1,1] [1,1,2] [1,2,3] [1,3,4]
#       p=3 f=2 z+3, n=2..4: [1,1] [2,2] [3,3]
#       p=2 f=2 z^2+2 and z^2+2az+2a, n=2..6: [1,1] [2,2] [1,1,2,2] [1,1,1,2,3] [1,1,2,3,3]
#       p=3 f=2 z^2+3, n=2..4: [1,1] [1,1,1,1] [1,1,1,1,2]
#       p=2 f=3 z+2, n=2,3: [1,1,1] [1,1,2,2]
#   * Further brute-force cases: z^2+2az+2 at n=3,4 gives [2,2], [1,1,2,2]; z^2+2az+2a at
#     n=4,5 gives [1,1,2,2], [1,1,1,2,3]; p=3: z^2+3az+3 at n=3 gives [1,1,1,1], and
#     z^2+3az+3a at n=4 gives [1,1,2,2].
#   * The original KZPN Sage code (f = 1, H^1 at i = 1, Sage 10.3) gives the same as
#     this oracle for z^2+2 at n=3,5 ([2], [1,1,2]), z^3+2 at n=3,5 ([2], [1,3]),
#     z^4+2 at n=4 ([1,2]), z^2+3 at n=2 ([1]), z+2 at n=2,3,4 and z+3 at n=4 ([3]).
#   * For n <= e the ring is F_q[z]/z^n, and the result equals `hm_exponents(p, f, n, 1)`
#     in hesselholt_madsen.jl.
#   * The result is invariant under E -> E^sigma and E(z) -> E(u z) for a unit u, and it
#     does not depend on whether E is monic.

"""
    principal_units_exponents(p, htilde, E, n) -> Vector{Int}

Return the sorted exponent list of the finite abelian `p`-group
`(1 + π O_K)/(1 + π^n O_K)`, computed by brute-force enumeration. This function is
independent of Syntomic.jl.

# Arguments
- `p::Int`: the prime.
- `htilde::Vector{Int}`: the integer coefficients (low to high) of a monic lift `h̃ ∈ Z[x]`,
  of degree `f`, of an irreducible polynomial over `F_p`. Then `W = W(F_q) = Z_p[x]/(h̃)`
  with `q = p^f`. For `f = 1`, pass `[0, 1]`; the value of `x` plays no role.
- `E::Vector{Vector{Int}}`: an Eisenstein polynomial over `W`, listed from low to high
  degree. Each coefficient is a coordinate vector of length `f` in the basis
  `1, x, …, x^(f-1)`. The leading coefficient must be a unit, but need not be 1. The
  other coefficients must be divisible by `p`, and `E_0 / p` must be a unit.
- `n::Int ≥ 1`: the power of the uniformizer `π = z`.

`n = 1` gives `Int[]`. The group has order `q^(n-1)`, so the enumeration is instant
for `q^(n-1)` up to a few thousand and still practical for about `10^6`.

# Examples
```julia-repl
julia> principal_units_exponents(2, [0, 1], [[2], [1]], 4)          # (Z/16)^× ⊃ U^1 = {±1} × Z/4
2-element Vector{Int64}:
 1
 2

julia> principal_units_exponents(3, [0, 1], [[3], [1]], 4)          # Z/27 (cyclic, odd p)
1-element Vector{Int64}:
 3

julia> principal_units_exponents(2, [1, 1, 1], [[0, 2], [0, 2], [1, 0]], 5)   # E = z^2+2az+2a over W(F_4)
5-element Vector{Int64}:
 1
 1
 1
 2
 3

julia> principal_units_exponents(2, [1, 1, 0, 1], [[2, 0, 0], [1, 0, 0]], 3)  # E = z+2 over W(F_8)
4-element Vector{Int64}:
 1
 1
 2
 2
```
"""
function principal_units_exponents(
    p::Int, htilde::Vector{Int}, E::Vector{Vector{Int}}, n::Int
)::Vector{Int}
    p >= 2 && _bf_isprime(p) || throw(ArgumentError("p = $p is not a prime"))
    n >= 1 || throw(ArgumentError("n must be ≥ 1 (got n = $n)"))
    length(htilde) >= 2 || throw(ArgumentError("htilde must have degree ≥ 1"))
    htilde[end] == 1 || throw(ArgumentError("htilde must be monic (last entry 1)"))
    f = length(htilde) - 1
    e = length(E) - 1
    e >= 1 || throw(ArgumentError("E must have degree ≥ 1"))
    all(c -> length(c) == f, E) ||
        throw(ArgumentError("every coefficient of E must be a vector of length f = $f"))
    n == 1 && return Int[]

    M = cld(n, e)                          # working modulus p^M (= p^{m_0})
    P = _bf_checked_pow(p, M)
    m = [max(cld(n - j, e), 0) for j in 0:(e - 1)]   # a_j is needed modulo p^{m_j}
    pm = [_bf_checked_pow(p, mj) for mj in m]
    h = [mod(c, P) for c in htilde]

    # Validate the residue field and the Eisenstein conditions (all modulo p).
    _bf_check_field(p, f, h)
    Ered = [[mod(c, P) for c in coeff] for coeff in E]
    for j in 0:(e - 1)
        all(c -> mod(c, p) == 0, E[j + 1]) || throw(
            ArgumentError("E is not Eisenstein: coefficient of z^$j is not divisible by p"),
        )
    end
    all(c -> mod(c, p * p) == 0, E[1]) &&
        throw(ArgumentError("E is not Eisenstein: E(0) is divisible by p^2"))
    lead = Ered[e + 1]
    all(c -> mod(c, p) == 0, lead) &&
        throw(ArgumentError("the leading coefficient of E is not a unit"))
    leadinv = _bf_winv(lead, h, p, M, P, f)
    # z^e = sum_{k<e} red[k+1] z^k with red[k+1] = -E_e^{-1} E_k (mod p^M)
    red = [[mod(-c, P) for c in _bf_wmul(leadinv, Ered[k + 1], h, P)] for k in 0:(e - 1)]

    # Mixed-radix indexing of U^1/U^n: a_0 = 1 + p*t with t mod p^{m_0-1}; a_j mod p^{m_j}.
    radices = Int[]
    for j in 0:(e - 1), _ in 1:f
        push!(radices, j == 0 ? div(pm[1], p) : pm[j + 1])
    end
    total = foldl(_bf_checked_mul, radices; init=1)
    total == _bf_checked_pow(p, f * (n - 1)) ||
        error("internal error: |U^1/U^n| = $total, expected q^(n-1)")
    total <= 50_000_000 ||
        throw(ArgumentError("group too large for brute force (order $total)"))

    decode = function (idx::Int)
        x = [zeros(Int, f) for _ in 1:e]
        r = idx
        k = 0
        for j in 0:(e - 1), d in 1:f
            k += 1
            digit = r % radices[k]
            r = div(r, radices[k])
            x[j + 1][d] = j == 0 ? p * digit + (d == 1 ? 1 : 0) : digit
        end
        return x
    end
    encode = function (x::Vector{Vector{Int}})
        idx = 0
        mult = 1
        k = 0
        for j in 0:(e - 1), d in 1:f
            k += 1
            c = mod(x[j + 1][d], pm[j + 1])
            digit = j == 0 ? div(c - (d == 1 ? 1 : 0), p) : c
            (j == 0 && mod(c - (d == 1 ? 1 : 0), p) != 0) &&
                error("internal error: element left U^1")
            idx += digit * mult
            mult *= radices[k]
        end
        return idx
    end

    # Table of the p-th power map on indices, then orders by memoised descent.
    pmap = Vector{Int}(undef, total)
    for idx in 0:(total - 1)
        x = decode(idx)
        pmap[idx + 1] = encode(_bf_okpow(x, p, red, h, P, f, e))
    end
    one_idx = 0                                # a_0 = 1, all other coordinates 0
    ord = fill(-1, total)
    ord[one_idx + 1] = 0
    stack = Int[]
    for idx in 0:(total - 1)
        cur = idx
        while ord[cur + 1] < 0
            push!(stack, cur)
            cur = pmap[cur + 1]
            length(stack) > 64 * M + 64 &&
                error("internal error: p-power map does not reach 1")
        end
        o = ord[cur + 1]
        while !isempty(stack)
            o += 1
            ord[pop!(stack) + 1] = o
        end
    end

    # |G[p^k]| = p^{s_k}, s_k = sum_j min(a_j, k); r_k = s_k - s_{k-1} = #{j : a_j >= k}.
    maxo = maximum(ord)
    s = Int[]
    for k in 0:maxo
        c = count(<=(k), ord)
        sk = _bf_exact_log(c, p)
        sk === nothing && error("internal error: |G[p^$k]| = $c is not a power of p")
        push!(s, sk)
    end
    rk = [s[k + 1] - s[k] for k in 1:maxo]           # rk[k] = #{j : a_j >= k}
    exps = Int[]
    for k in 1:maxo
        nk = rk[k] - (k < maxo ? rk[k + 1] : 0)
        nk >= 0 || error("internal error: inconsistent p-torsion counts")
        append!(exps, fill(k, nk))
    end
    sum(exps) == f * (n - 1) || error("internal error: exponent sum is not f(n-1)")
    return sort!(exps)
end

"""
    kgroups_i1_expected(p, htilde, E, n) -> Vector{Int}

Return the expected weight-1 answer, the exponent list of
`H^1(Z_p(1)(O_K/π^n)) = K_1(O_K/π^n; Z_p)`. It simply calls
[`principal_units_exponents`](@ref) and uses the same arguments.

The companion group `H^2(Z_p(1)(O_K/π^n))` is zero, because the Picard and Brauer
groups of a local Artinian ring vanish. This function is independent of Syntomic.jl.

# Example
```julia-repl
julia> kgroups_i1_expected(2, [0, 1], [[2], [1]], 3)                # K_1(Z/8; Z_2) = (Z/2)^2
2-element Vector{Int64}:
 1
 1
```
"""
kgroups_i1_expected(p::Int, htilde::Vector{Int}, E::Vector{Vector{Int}}, n::Int) =
    principal_units_exponents(p, htilde, E, n)

"""
    BRUTE_FORCE_UNITS_CASES

The validated cases `(p, htilde, E, n, expected)` listed in the header of
`brute_force_units.jl`. Tests can loop over them. `a` denotes the Conway generator
`x` of `W(F_q)`.
"""
const BRUTE_FORCE_UNITS_CASES = let
    h1 = [0, 1]                 # f = 1
    h4 = [1, 1, 1]              # F_4: x^2 + x + 1
    h9 = [2, 2, 1]              # F_9: x^2 + 2x + 2
    h8 = [1, 1, 0, 1]           # F_8: x^3 + x + 1
    [
        # f = 1, E = z + p ([AKN24, Appendix A], K_1 rows)
        (2, h1, [[2], [1]], 2, [1]),
        (2, h1, [[2], [1]], 3, [1, 1]),
        (2, h1, [[2], [1]], 4, [1, 2]),
        (2, h1, [[2], [1]], 5, [1, 3]),
        (3, h1, [[3], [1]], 4, [3]),
        (5, h1, [[5], [1]], 4, [3]),
        (7, h1, [[7], [1]], 3, [2]),
        # f = 1, ramified (matches the original KZPN Sage code at i = 1)
        (2, h1, [[2], [0], [1]], 3, [2]),
        (2, h1, [[2], [0], [1]], 5, [1, 1, 2]),
        (2, h1, [[2], [0], [0], [1]], 3, [2]),
        (2, h1, [[2], [0], [0], [1]], 5, [1, 3]),
        (2, h1, [[2], [0], [0], [0], [1]], 4, [1, 2]),
        (3, h1, [[3], [0], [1]], 2, [1]),
        # independent brute-force computations
        (2, h4, [[2, 0], [1, 0]], 2, [1, 1]),
        (2, h4, [[2, 0], [1, 0]], 3, [1, 1, 2]),
        (2, h4, [[2, 0], [1, 0]], 4, [1, 2, 3]),
        (2, h4, [[2, 0], [1, 0]], 5, [1, 3, 4]),
        (2, h4, [[0, 2], [1, 0]], 4, [1, 2, 3]),
        (3, h9, [[3, 0], [1, 0]], 2, [1, 1]),
        (3, h9, [[3, 0], [1, 0]], 3, [2, 2]),
        (3, h9, [[3, 0], [1, 0]], 4, [3, 3]),
        (2, h4, [[2, 0], [0, 0], [1, 0]], 2, [1, 1]),
        (2, h4, [[2, 0], [0, 0], [1, 0]], 3, [2, 2]),
        (2, h4, [[2, 0], [0, 0], [1, 0]], 4, [1, 1, 2, 2]),
        (2, h4, [[2, 0], [0, 0], [1, 0]], 5, [1, 1, 1, 2, 3]),
        (2, h4, [[2, 0], [0, 0], [1, 0]], 6, [1, 1, 2, 3, 3]),
        (2, h4, [[0, 2], [0, 2], [1, 0]], 6, [1, 1, 2, 3, 3]),
        (3, h9, [[3, 0], [0, 0], [1, 0]], 2, [1, 1]),
        (3, h9, [[3, 0], [0, 0], [1, 0]], 3, [1, 1, 1, 1]),
        (3, h9, [[3, 0], [0, 0], [1, 0]], 4, [1, 1, 1, 1, 2]),
        (2, h8, [[2, 0, 0], [1, 0, 0]], 2, [1, 1, 1]),
        (2, h8, [[2, 0, 0], [1, 0, 0]], 3, [1, 1, 2, 2]),
        # further brute-force cases
        (2, h4, [[2, 0], [0, 2], [1, 0]], 3, [2, 2]),
        (2, h4, [[2, 0], [0, 2], [1, 0]], 4, [1, 1, 2, 2]),
        (2, h4, [[0, 2], [0, 2], [1, 0]], 4, [1, 1, 2, 2]),
        (2, h4, [[0, 2], [0, 2], [1, 0]], 5, [1, 1, 1, 2, 3]),
        (3, h9, [[3, 0], [0, 3], [1, 0]], 3, [1, 1, 1, 1]),
        (3, h9, [[0, 3], [0, 3], [1, 0]], 4, [1, 1, 2, 2]),
        # invariance: E = z^2+2az+2 (above), E^sigma = z^2+2a^2z+2, E(az) = a^2z^2+2a^2z+2
        # (non-monic), with a^2 = -a-1 in W(F_4)
        (2, h4, [[2, 0], [-2, -2], [1, 0]], 4, [1, 1, 2, 2]),
        (2, h4, [[2, 0], [-2, -2], [-1, -1]], 4, [1, 1, 2, 2]),
        (2, h4, [[2, 0], [-2, -2], [-1, -1]], 5, [1, 1, 1, 2, 3]),
    ]
end

# ---------------------------------------------------------------------------------------
# Internal helpers (prefixed _bf_ to avoid clashing with anything in the including module)
# ---------------------------------------------------------------------------------------

function _bf_isprime(p::Int)
    p < 2 && return false
    d = 2
    while d * d <= p
        p % d == 0 && return false
        d += 1
    end
    return true
end

function _bf_checked_pow(b::Int, k::Int)
    r = 1
    for _ in 1:k
        r = Base.checked_mul(r, b)
    end
    return r
end

_bf_checked_mul(a::Int, b::Int) = Base.checked_mul(a, b)

function _bf_exact_log(c::Int, p::Int)
    k = 0
    while c > 1
        c % p == 0 || return nothing
        c = div(c, p)
        k += 1
    end
    return c == 1 ? k : nothing
end

# Product in (Z/P)[x]/(h), where h is monic of degree f (coefficients low to high).
function _bf_wmul(a::Vector{Int}, b::Vector{Int}, h::Vector{Int}, P::Int)
    f = length(a)
    f == 1 && return [Int(mod(widemul(a[1], b[1]), P))]
    c = zeros(Int, 2 * f - 1)                 # index k <-> degree k-1
    for i in 1:f
        a[i] == 0 && continue
        for j in 1:f
            c[i + j - 1] = Int(mod(c[i + j - 1] + widemul(a[i], b[j]), P))
        end
    end
    for k in (2 * f - 1):-1:(f + 1)           # x^(k-1) = x^(k-1-f) * (-(h_0 + ... + h_{f-1} x^{f-1}))
        t = c[k]
        t == 0 && continue
        c[k] = 0
        for l in 0:(f - 1)
            c[k - f + l] = Int(mod(c[k - f + l] - widemul(t, h[l + 1]), P))
        end
    end
    return c[1:f]
end

function _bf_wadd(a::Vector{Int}, b::Vector{Int}, P::Int)
    return [mod(a[i] + b[i], P) for i in eachindex(a)]
end

function _bf_wone(f::Int)
    v = zeros(Int, f)
    v[1] = 1
    return v
end

function _bf_wpow(a::Vector{Int}, k::Integer, h::Vector{Int}, P::Int)
    result = _bf_wone(length(a))
    base = a
    while k > 0
        isodd(k) && (result = _bf_wmul(result, base, h, P))
        k >>= 1
        k > 0 && (base = _bf_wmul(base, base, h, P))
    end
    return result
end

# F_q = F_p[x]/(h mod p) must be a field: every nonzero y satisfies y^(q-1) = 1.
function _bf_check_field(p::Int, f::Int, h::Vector{Int})
    f == 1 && return nothing
    q = _bf_checked_pow(p, f)
    hp = [mod(c, p) for c in h]
    y = zeros(Int, f)
    for idx in 1:(q - 1)
        r = idx
        for d in 1:f
            y[d] = r % p
            r = div(r, p)
        end
        _bf_wpow(y, q - 1, hp, p) == _bf_wone(f) || throw(
            ArgumentError(
                "htilde is not irreducible modulo p, so Z_p[x]/(htilde) is not W(F_q)"
            ),
        )
    end
    return nothing
end

# Inverse of a unit u of W/p^M: start from u^(q-2) (an inverse modulo p), then Newton.
function _bf_winv(u::Vector{Int}, h::Vector{Int}, p::Int, M::Int, P::Int, f::Int)
    q = _bf_checked_pow(p, f)
    y = _bf_wpow(u, q - 2, h, P)
    prec = 1
    two = _bf_wone(f) .* 2
    while prec < M
        uy = _bf_wmul(u, y, h, P)
        y = _bf_wmul(y, [mod(two[i] - uy[i], P) for i in 1:f], h, P)
        prec *= 2
    end
    _bf_wmul(u, y, h, P) == _bf_wone(f) ||
        throw(ArgumentError("the leading coefficient of E is not invertible modulo p^$M"))
    return y
end

# Product in (W/P)[z]/(E), with z^e = sum_k red[k+1] z^k.
function _bf_okmul(x, y, red, h, P, f, e)
    c = [zeros(Int, f) for _ in 1:(2 * e - 1)]
    for i in 1:e
        all(iszero, x[i]) && continue
        for j in 1:e
            all(iszero, y[j]) && continue
            c[i + j - 1] = _bf_wadd(c[i + j - 1], _bf_wmul(x[i], y[j], h, P), P)
        end
    end
    for k in (2 * e - 1):-1:(e + 1)           # z^(k-1) = z^(k-1-e) * z^e
        t = c[k]
        all(iszero, t) && continue
        c[k] = zeros(Int, f)
        for l in 0:(e - 1)
            c[k - e + l] = _bf_wadd(c[k - e + l], _bf_wmul(t, red[l + 1], h, P), P)
        end
    end
    return c[1:e]
end

function _bf_okpow(x, k::Int, red, h, P, f, e)
    result = [l == 1 ? _bf_wone(f) : zeros(Int, f) for l in 1:e]
    base = x
    while k > 0
        isodd(k) && (result = _bf_okmul(result, base, red, h, P, f, e))
        k >>= 1
        k > 0 && (base = _bf_okmul(base, base, red, h, P, f, e))
    end
    return result
end
