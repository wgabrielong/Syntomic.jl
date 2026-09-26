# Step 4 for the weight i: the units w(v^i) = w(v)^i.
#
# ∇ in Breuil–Kisin weight i involves w(v^i) = w(v)^i ([AKN, Lemma 4.40, p. 36]; [AKN, §7.1,
# Step 5, p. 56]). It is computed in one of these ways, all giving the same result:
#
# * from w(v) (Step 4), by binary powering, or by the binomial expansion
#   w(v)^i = Σ_k C(i,k) y^k, y = w(v) - 1 ∈ F^{≥p}, from a table of the y^k with k·wt(y) < L
#   (wt the F-weight valuation); in a sweep over i also incrementally, w(v)^i = w(v)^{i-1}·w(v);
# * as a fixed point: raising w(v) = w(u)·φ(w(v)) ([AKN, §7.1, Step 4, p. 56]) to the i-th power
#   gives w(v)^i = w(u)^i·φ(w(v)^i), i.e. w(v^i) = ∏_{r ≥ 0} φ^r(w(u)^i) ([AKN, Lemma 4.15,
#   p. 29]). The unit w(u) = 1 + a is sparse (a = φ(Q)φ_1(g_0) only has components of F-weight
#   t ≤ e·p), so w(u)^i is cheap (incrementally, or by the binomial expansion in a), and the
#   fixed point needs about one product of general elements modulo F^{≥L}.
#
# The method is chosen by a cost model that counts products in the g-envelope at their
# F-truncations (see `g_mul_cost`).

# ---------------------------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------------------------

"""
    g_mul_cost(p, T; w=0, square=false, support=T) -> Float64

A model of the cost of a product in the g-envelope modulo `F^{≥T}`: `T^4` for general operands,
times
- `(1 - (w/T)^2)^2` if the result lies in `F^{≥w}` (a product of elements of high F-weight is
  cheaper only once `w` is a sizeable fraction of `T`),
- `1/2 + 8/(5p^2)` for a square (it saves less for small `p`, where most products of basis
  monomials carry and have to be reduced),
- `min(1, 4·support/T)` if one operand has only `support` nonzero components `c_t`.
The time of a product grows like `T^4`; the factors are calibrated by timing products
(`benchmark/gside.jl`).
"""
function g_mul_cost(
    p::Integer, T::Integer; w::Integer=0, square::Bool=false, support::Integer=T
)
    T <= 0 && return 0.0
    x = min(w / T, 1.0)
    c = float(T)^4 * (1 - x^2)^2 * min(1.0, 4 * support / T)
    return square ? c * (1 / 2 + 8 / (5 * p^2)) : c
end

"""
    g_frobenius_cost(p, T) -> Float64

The modelled cost of `φ` modulo `F^{≥T}` (with the table of the `φ(g^{μ(t)})` already built):
about `0.12` of a product (see `g_mul_cost`).
"""
g_frobenius_cost(p::Integer, T::Integer) = 0.12 * g_mul_cost(p, T)

"""
    binary_power_cost(p, i, T) -> Float64

The modelled cost (see `g_mul_cost`) of `x^i` modulo `F^{≥T}` by binary powering of a unit `x`:
`⌊log_2 i⌋` squares and `popcount(i) - 1` products.
"""
function binary_power_cost(p::Integer, i::Integer, T::Integer)
    i <= 1 && return 0.0
    nsq = ndigits(i; base=2) - 1
    return nsq * g_mul_cost(p, T; square=true) + (count_ones(i) - 1) * g_mul_cost(p, T)
end

"""
    binomial_table_size(wt, L, kmax) -> Int

The number of powers `y^k`, `1 ≤ k ≤ kmax`, of an element `y ∈ F^{≥wt}` that are nonzero modulo
`F^{≥L}`: `k·wt < L` (all `k ≤ kmax` if `wt = 0`).
"""
binomial_table_size(wt::Integer, L::Integer, kmax::Integer) =
    max(wt == 0 ? kmax : min(kmax, div(L - 1, wt)), 0)

"""
    binomial_table_cost(p, L, wt, K) -> Float64

The modelled cost (see `g_mul_cost`) of the table `y, y^2, …, y^K` modulo `F^{≥L}` for a general
`y ∈ F^{≥wt}` (`y^k = y^{k-1}·y`, a square for `k = 2`).
"""
function binomial_table_cost(p::Integer, L::Integer, wt::Integer, K::Integer)
    return sum((g_mul_cost(p, L; w=k * wt, square=k == 2) for k in 2:K); init=0.0)
end

"""
    bk_unit_v_cost(p, L, s) -> Float64

The modelled cost of `w(v)` modulo `F^{≥L}` by `bk_unit_v`: a product with `w(u)` (which has `s`
nonzero components) and a Frobenius for each F-truncation of `bk_unit_ladder`.
"""
function bk_unit_v_cost(p::Integer, L::Integer, s::Integer)
    return sum(
        (
            g_mul_cost(p, T; support=s) + g_frobenius_cost(p, T) for
            T in bk_unit_ladder(p, L)
        );
        init=0.0,
    )
end

"""
    bk_unit_fixed_point_cost(p, L) -> Float64

The modelled cost of `bk_unit_fixed_point` modulo `F^{≥L}`: a product and a Frobenius for each
F-truncation of `bk_unit_fixed_point_ladder` after the first.
"""
function bk_unit_fixed_point_cost(p::Integer, L::Integer)
    Ts = bk_unit_fixed_point_ladder(p, L)
    return sum((g_mul_cost(p, T) + g_frobenius_cost(p, T) for T in Ts[2:end]); init=0.0)
end

# The name of the smallest entry of a NamedTuple of costs (the first one on ties).
function _cheapest(costs::NamedTuple)
    best = first(keys(costs))
    for (k, c) in pairs(costs)
        c < costs[best] && (best = k)
    end
    return best
end

"""
    unit_power_cost(p, L, i, wt, s) -> Float64

The modelled cost (see `g_mul_cost`) of `x^i` modulo `F^{≥L}` by `bk_unit_u_power`, where
`a = x - 1 ∈ F^{≥wt}` has `s` nonzero components: the products `a^{k-1}·a`, `2 ≤ k ≤ K`, of the
binomial table if `K = binomial_table_size(wt, L, i) < i`, and `i - 1` products with `x`
otherwise.
"""
function unit_power_cost(p::Integer, L::Integer, i::Integer, wt::Integer, s::Integer)
    K = binomial_table_size(wt, L, i)
    K < i || return (i - 1) * g_mul_cost(p, L; support=s)
    return sum((g_mul_cost(p, L; w=k * wt, support=s) for k in 2:K); init=0.0)
end

# ---------------------------------------------------------------------------------------------
# Binomial expansions
# ---------------------------------------------------------------------------------------------

"""
    BinomialPowers

The powers `ypow[k] = y^k`, `1 ≤ k ≤ K`, of `y = x - 1` modulo `F^{≥L}` for an element `x` of the
g-envelope, with the F-weight valuation `wt` of `y` (so `y^k = 0` modulo `F^{≥L}` for
`k·wt ≥ L`) and the precision `prec` of `x`: the table for the binomial expansion
`x^i = Σ_k C(i, k) y^k`. Built by `binomial_powers`.
"""
struct BinomialPowers
    L::Int
    wt::Int
    prec::Int
    ypow::Vector{GElem}
end

"""
    binomial_powers(env, x, kmax; L=g_length(x)) -> BinomialPowers

The table `y^k` modulo `F^{≥L}`, `y = x - 1`, for `1 ≤ k ≤ kmax` with `k·wt(y) < L` (see
`binomial_table_size`), where `wt(y)` is the F-weight valuation of `y`.
"""
function binomial_powers(env::GEnvelope, x::GElem, kmax::Integer; L::Integer=g_length(x))
    L = _g_check_length(L, g_length(x))
    y = g_sub(env, x, g_one(env; L=L); L=L)
    wt = g_weight_valuation(y)
    ypow = GElem[]
    K = binomial_table_size(wt, L, kmax)
    K >= 1 && push!(ypow, y)
    for _ in 2:K
        push!(ypow, g_mul(env, ypow[end], y))
    end
    return BinomialPowers(L, wt, x.prec, ypow)
end

"""
    binomial_power(env, B::BinomialPowers, i; L=B.L) -> GElem

`x^i = Σ_{0 ≤ k ≤ i} C(i, k) y^k` modulo `F^{≥L}` (`L ≤ B.L`) from the table `B` of powers of
`y = x - 1`; it needs `y^k` for `k ≤ min(i, (L-1)/wt(y))` (so `i` up to the `kmax` of `B`, or any
`i` if all nonzero powers are tabulated).
"""
function binomial_power(env::GEnvelope, B::BinomialPowers, i::Integer; L::Integer=B.L)
    i >= 0 || throw(ArgumentError("binomial_power: negative exponent"))
    L = _g_check_length(L, B.L)
    K = binomial_table_size(B.wt, L, i)
    K <= length(B.ypow) || throw(ArgumentError("binomial_power: y^$K is not tabulated"))
    R = env.R
    out = g_one(env; L=L).c                      # new polynomials, updated in place
    prec = B.prec
    t = zero(R.Znz)
    for k in 1:K
        yk = B.ypow[k]
        prec = min(prec, yk.prec)
        c = mod(ZZ(binomial(big(i), k)), R.pN)
        iszero(c) && continue
        for s in 0:(L - 1)
            yk.vals[s + 1] >= L - s && continue
            o = out[s + 1]
            for d in 1:(R.f)
                a = yk.c[s + 1][d]
                iszero(a) && continue
                length(a) > L - s && (a = t = poly_set_trunc!(t, a, L - s))
                o[d] = poly_addmul!(o[d], a, c)
            end
        end
    end
    return GElem(out, prec)
end

# The F-weight valuation of x - 1 and the number of nonzero components of x - 1, modulo F^{≥L}.
function _unit_shape(env::GEnvelope, x::GElem, L::Integer)
    y = g_sub(env, x, g_one(env; L=L); L=L)
    return g_weight_valuation(y), count(c -> !all(iszero, c), y.c)
end

# ---------------------------------------------------------------------------------------------
# w(v^i) as a fixed point
# ---------------------------------------------------------------------------------------------

"""
    bk_unit_fixed_point_ladder(p, L) -> Vector{Int}

The F-truncations `T_k = ⌈L/p^{K-k}⌉`, `0 ≤ k ≤ K = bk_unit_iterations(p, L)`, of
`bk_unit_fixed_point`: `T_0 ≤ p^2`, `T_k ≤ p·T_{k-1}` and `T_K = L`.
"""
function bk_unit_fixed_point_ladder(p::Integer, L::Integer)
    K = bk_unit_iterations(p, L)
    return [Int(cld(big(L), big(p)^(K - k))) for k in 0:K]
end

"""
    bk_unit_fixed_point(env, U; L=g_length(U)) -> GElem

The solution `X = ∏_{r ≥ 0} φ^r(U)` of `X = U·φ(X)` modulo `F^{≥L}`, for `U - 1 ∈ F^{≥p}`. For
`U = w(u)^i` this is `w(v^i) = w(v)^i`: `w(v)` is the fixed point of `x ↦ w(u)·φ(x)`
([AKN, §7.1, Step 4, p. 56]; [AKN, Lemma 4.15, p. 29]) and `φ` is multiplicative. Since
`X ≡ U` modulo `F^{≥p^2}`, and `x ≡ X` modulo `F^{≥T}` implies `U·φ(x) ≡ X` modulo `F^{≥pT}`,
the `K = bk_unit_iterations(p, L)` iterations `x_k = U·φ(x_{k-1})` are computed modulo
`F^{≥T_k}` for the F-truncations `T_0 < T_1 < … < T_K = L` of `bk_unit_fixed_point_ladder`,
starting from `x_0 = U` modulo `F^{≥T_0}`. Throws an `ErrorException` if `x_K ≢ x_{K-1}` modulo
`F^{≥T_{K-1}}` (an internal consistency check: both are `X` modulo `F^{≥T_{K-1}}`).
"""
function bk_unit_fixed_point(env::GEnvelope, U::GElem; L::Integer=g_length(U))
    L = _g_check_length(L, g_length(U))
    p = env.R.p
    wt, _ = _unit_shape(env, U, L)
    wt >= min(p, L) || throw(
        ArgumentError(
            "bk_unit_fixed_point: U - 1 must lie in F^{≥p} (it lies in F^{≥$wt})"
        ),
    )
    Ts = bk_unit_fixed_point_ladder(p, L)
    x = g_truncate(env, U, Ts[1])
    prev = x
    for T in Ts[2:end]
        prev = x
        x = g_mul(env, U, g_frobenius(env, x; L=T); L=T)
    end
    K = length(Ts) - 1
    K >= 1 && _bk_unit_check_modulo(env, x, prev, K, Ts[K], "the fixed point")
    return x
end

"""
    bk_unit_u_power(env, wu, i; L=g_length(wu)) -> GElem

`w(u)^i = w(u^i)` modulo `F^{≥L}` for `i ≥ 1`: by the binomial expansion in `a = w(u) - 1` if
fewer than `i` powers of `a` are nonzero modulo `F^{≥L}` (see `binomial_powers`), and as
`w(u)^{k-1}·w(u)` otherwise; both use products with the sparse elements `a` or `w(u)` (see
`unit_power_cost`).
"""
function bk_unit_u_power(env::GEnvelope, wu::GElem, i::Integer; L::Integer=g_length(wu))
    i >= 1 || throw(ArgumentError("bk_unit_u_power: the exponent must be positive"))
    L = _g_check_length(L, g_length(wu))
    wt, _ = _unit_shape(env, wu, L)
    if binomial_table_size(wt, L, i) < i
        return binomial_power(env, binomial_powers(env, wu, i; L=L), i)
    end
    x = g_truncate(env, wu, L)
    wuL = x
    for _ in 2:i
        x = g_mul(env, x, wuL)
    end
    return x
end

# ---------------------------------------------------------------------------------------------
# One weight
# ---------------------------------------------------------------------------------------------

"""
    bk_unit_power_method(p, i, L, wt) -> Symbol

For `w(v)^i` modulo `F^{≥L}` from `w(v)`: `:binomial` if the modelled cost of the binomial table
(see `binomial_table_cost`; `wt` is the F-weight valuation of `w(v) - 1`) is below that of binary
powering (see `binary_power_cost`), and `:binary` otherwise.
"""
function bk_unit_power_method(p::Integer, i::Integer, L::Integer, wt::Integer)
    i <= 1 && return :binary
    K = binomial_table_size(wt, L, i)
    return if binomial_table_cost(p, L, wt, K) < binary_power_cost(p, i, L)
        :binomial
    else
        :binary
    end
end

"""
    bk_unit_power(env, wv, i; L=g_length(wv), method=:auto) -> GElem

`w(v^i) = w(v)^i` modulo `F^{≥L}` from `wv = w(v)`, the factor relating the Breuil–Kisin twists
in `η_R(w(s^i)) = w(v^i) w(s^i)` ([AKN, Lemma 4.40, p. 36]; used in [AKN, §7.1, Step 5, p. 56]),
by binary powering (`method = :binary`) or from the binomial expansion in `y = w(v) - 1`
(`:binomial`); `:auto` chooses by the cost model (see `bk_unit_power_method`). The result does not
depend on the method.
"""
function bk_unit_power(
    env::GEnvelope, wv::GElem, i::Integer; L::Integer=g_length(wv), method::Symbol=:auto
)
    L = _g_check_length(L, g_length(wv))
    if method === :auto
        method = bk_unit_power_method(env.R.p, i, L, first(_unit_shape(env, wv, L)))
    end
    method === :binary && return g_pow(env, wv, i; L=L)
    method === :binomial && return binomial_power(env, binomial_powers(env, wv, i; L=L), i)
    throw(ArgumentError("unknown method :$method (use :auto, :binary or :binomial)"))
end

"""
    bk_unit_v_power_method(p, i, L, s, wt) -> Symbol

The method of `bk_unit_v_power` with the smallest modelled cost (see `g_mul_cost`), where `w(u) - 1`
has `s` nonzero components and F-weight valuation `wt`: `:fixed_point` (`w(u)^i`, then
`bk_unit_fixed_point`), or `w(v)` by `bk_unit_v` followed by `:binary` or `:binomial` (see
`bk_unit_power`; the F-weight valuation of `w(v) - 1` is taken to be `min(wt, p^2)`, since
`w(v) ≡ w(u)` modulo `F^{≥p^2}`).
"""
function bk_unit_v_power_method(p::Integer, i::Integer, L::Integer, s::Integer, wt::Integer)
    fixed = unit_power_cost(p, L, i, wt, s) + bk_unit_fixed_point_cost(p, L)
    wv = bk_unit_v_cost(p, L, s)
    wty = min(wt, p^2)
    costs = (
        fixed_point=fixed,
        binary=wv + binary_power_cost(p, i, L),
        binomial=wv + binomial_table_cost(p, L, wty, binomial_table_size(wty, L, i)),
    )
    return _cheapest(costs)
end

"""
    bk_unit_v_power(env, wu, i; L=g_length(wu), method=:auto) -> GElem

`w(v^i) = w(v)^i` modulo `F^{≥L}` from `wu = w(u)`, for `i ≥ 1` ([AKN, §7.1, Steps 4–5,
pp. 55–56; Lemma 4.15, p. 29; Lemma 4.40, p. 36]): as the fixed point of `x ↦ w(u)^i·φ(x)`
(`method = :fixed_point`, see `bk_unit_fixed_point` and `bk_unit_u_power`), or from `w(v)`
(`bk_unit_v`) by `:binary` powering or the `:binomial` expansion (see `bk_unit_power`); `:auto`
chooses by the cost model (see `bk_unit_v_power_method`). The value does not depend on the
method; the recorded precision `prec` may be higher for `:fixed_point`, which reduces fewer
normal forms.
"""
function bk_unit_v_power(
    env::GEnvelope, wu::GElem, i::Integer; L::Integer=g_length(wu), method::Symbol=:auto
)
    i >= 1 || throw(ArgumentError("bk_unit_v_power: the weight must be positive"))
    L = _g_check_length(L, g_length(wu))
    if method === :auto
        wt, s = _unit_shape(env, wu, L)
        method = bk_unit_v_power_method(env.R.p, i, L, s, wt)
    end
    if method === :fixed_point
        return bk_unit_fixed_point(env, bk_unit_u_power(env, wu, i; L=L))
    elseif method in (:binary, :binomial)
        return bk_unit_power(env, bk_unit_v(env, wu; L=L), i; L=L, method=method)
    end
    throw(
        ArgumentError(
            "unknown method :$method (use :auto, :fixed_point, :binary or :binomial)"
        ),
    )
end

# ---------------------------------------------------------------------------------------------
# All weights of a sweep
# ---------------------------------------------------------------------------------------------

"""
    bk_unit_powers_method(p, is, Ls, wt) -> Symbol

For `w(v)^i`, `i ∈ is` (sorted), modulo `F^{≥Ls[i]}` from `w(v)`, the method of `bk_unit_powers`
with the smallest modelled cost (see `g_mul_cost`), `wt` the F-weight valuation of `w(v) - 1`:
- `:incremental`: `w(v)^i = w(v)^{i-1}·w(v)` for consecutive weights (binary powering otherwise),
  modulo `F^{≥max Ls}`;
- `:binary`: binary powering modulo `F^{≥Ls[i]}` for each `i`;
- `:binomial`: the binomial table modulo `F^{≥max Ls}`, then a linear combination for each `i`.
"""
function bk_unit_powers_method(p::Integer, is, Ls, wt::Integer)
    return _cheapest(_bk_unit_powers_costs(p, is, Ls, wt))
end

# The modelled costs of the methods of `bk_unit_powers`.
function _bk_unit_powers_costs(p::Integer, is, Ls, wt::Integer)
    Lmax = maximum(Ls[i] for i in is)
    incremental = 0.0
    prev = 0
    for i in is
        incremental += if i == prev + 1 && prev > 0
            g_mul_cost(p, Lmax)
        else
            binary_power_cost(p, i, Lmax)
        end
        prev = i
    end
    binary = sum(binary_power_cost(p, i, Ls[i]) for i in is)
    binomial = binomial_table_cost(p, Lmax, wt, binomial_table_size(wt, Lmax, maximum(is)))
    return (incremental=incremental, binary=binary, binomial=binomial)
end

"""
    bk_unit_powers(env, wv, is, Ls; method=:auto) -> Dict{Int,GElem}

`w(v^i) = w(v)^i` modulo `F^{≥Ls[i]}` from `wv = w(v)` for all `i ∈ is` (a sweep;
`Ls[i] ≤ g_length(wv)`), by the method `:incremental`, `:binary` or `:binomial` (see
`bk_unit_powers_method`, which `:auto` uses) ([AKN, Lemma 4.40, p. 36]; [AKN, §7.1, Step 5,
p. 56]). The results do not depend on the method.
"""
function bk_unit_powers(env::GEnvelope, wv::GElem, is, Ls; method::Symbol=:auto)
    sorted = sort(unique(collect(is)))
    out = Dict{Int,GElem}()
    isempty(sorted) && return out
    Lmax = maximum(Ls[i] for i in sorted)
    if method === :auto
        method = bk_unit_powers_method(
            env.R.p, sorted, Ls, first(_unit_shape(env, wv, Lmax))
        )
    end
    if method === :incremental
        wvL = g_truncate(env, wv, Lmax)
        x = nothing
        prev = 0
        for i in sorted
            x = if x !== nothing && i == prev + 1
                g_mul(env, x, wvL)
            else
                g_pow(env, wv, i; L=Lmax)
            end
            prev = i
            out[i] = g_truncate(env, x, Ls[i])
        end
    elseif method === :binary
        for i in sorted
            out[i] = g_pow(env, wv, i; L=Ls[i])
        end
    elseif method === :binomial
        B = binomial_powers(env, wv, last(sorted); L=Lmax)
        for i in sorted
            out[i] = binomial_power(env, B, i; L=Ls[i])
        end
    else
        throw(
            ArgumentError(
                "unknown method :$method (use :auto, :incremental, :binary or :binomial)"
            ),
        )
    end
    return out
end

"""
    bk_unit_v_powers_method(p, is, Ls, s, wt) -> Symbol

The method of `bk_unit_v_powers` with the smallest modelled cost (see `g_mul_cost`), where
`w(u) - 1` has `s` nonzero components and F-weight valuation `wt`: `:fixed_point` (`w(u)^i` for
all `i` from one table, then `bk_unit_fixed_point` modulo `F^{≥Ls[i]}` for each `i`), or `w(v)`
by `bk_unit_v` followed by a method of `bk_unit_powers` (`:incremental`, `:binary`,
`:binomial`).
"""
function bk_unit_v_powers_method(p::Integer, is, Ls, s::Integer, wt::Integer)
    Lmax = maximum(Ls[i] for i in is)
    fixed = unit_power_cost(p, Lmax, maximum(is), wt, s)
    fixed += sum(bk_unit_fixed_point_cost(p, Ls[i]) for i in is)
    wv = bk_unit_v_cost(p, Lmax, s)
    others = map(c -> wv + c, _bk_unit_powers_costs(p, is, Ls, min(wt, p^2)))
    return _cheapest(merge((fixed_point=fixed,), others))
end

"""
    bk_unit_v_powers(env, wu, is, Ls; method=:auto) -> Dict{Int,GElem}

`w(v^i) = w(v)^i` modulo `F^{≥Ls[i]}` from `wu = w(u)` for all `i ∈ is` (a sweep;
`Ls[i] ≤ g_length(wu)`) ([AKN, §7.1, Steps 4–5, pp. 55–56; Lemma 4.15, p. 29; Lemma 4.40,
p. 36]): by `:fixed_point` (the powers `w(u)^i` modulo `F^{≥max Ls}` from one table, see
`bk_unit_u_power`, then `bk_unit_fixed_point` modulo `F^{≥Ls[i]}`), or from `w(v)` by a method
of `bk_unit_powers` (`:incremental`, `:binary`, `:binomial`); `:auto` chooses by the cost model
(see `bk_unit_v_powers_method`). The values do not depend on the method; the recorded precision
`prec` may be higher for `:fixed_point`, which reduces fewer normal forms.
"""
function bk_unit_v_powers(env::GEnvelope, wu::GElem, is, Ls; method::Symbol=:auto)
    sorted = sort(unique(collect(is)))
    isempty(sorted) && return Dict{Int,GElem}()
    all(>=(1), sorted) || throw(ArgumentError("bk_unit_v_powers: weights must be positive"))
    Lmax = maximum(Ls[i] for i in sorted)
    if method === :auto
        wt, s = _unit_shape(env, wu, Lmax)
        method = bk_unit_v_powers_method(env.R.p, sorted, Ls, s, wt)
    end
    method === :fixed_point ||
        return bk_unit_powers(env, bk_unit_v(env, wu; L=Lmax), sorted, Ls; method=method)
    imax = last(sorted)
    wt, _ = _unit_shape(env, wu, Lmax)
    upow = if binomial_table_size(wt, Lmax, imax) < imax
        B = binomial_powers(env, wu, imax; L=Lmax)
        i -> binomial_power(env, B, i; L=Ls[i])
    else
        table = Dict{Int,GElem}()
        wuL = g_truncate(env, wu, Lmax)
        x = wuL
        for k in 1:imax
            k > 1 && (x = g_mul(env, x, wuL))
            k in sorted && (table[k] = x)
        end
        i -> g_truncate(env, table[i], Ls[i])
    end
    # the fixed points are independent: largest first, in parallel when `env.threaded`
    order = reverse(sorted)
    xs = tmap(i -> bk_unit_fixed_point(env, upow(i)), order; threaded=env.threaded)
    return Dict(zip(order, xs))
end
