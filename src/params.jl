# Integer parameters of a computation and the combinatorics of the linear bases.
#
# [AKN, §7.1, Step 2 ("initialize linear bases"), pp. 54–55].

"""
    IntegerLike

Machine or FLINT integers.
"""
const IntegerLike = Union{Integer,ZZRingElem}

"""
    ilog(p, x) -> Int

The exact integer `⌊log_p(x)⌋` for integers `p ≥ 2`, `x ≥ 1` (integer arithmetic only).
"""
function ilog(p::Integer, x::Integer)
    p >= 2 || throw(ArgumentError("ilog: base must be at least 2, got $p"))
    x >= 1 || throw(ArgumentError("ilog: argument must be positive, got $x"))
    k = 0
    y = BigInt(p)
    while y <= x
        k += 1
        y *= p
    end
    return k
end

"""
    base_p_digits(x, p, len) -> Vector{Int}

The base-`p` digits `[d_0, d_1, …, d_{len-1}]` of `0 ≤ x < p^len` (least significant first).
These are the exponents `e_u` of the basis monomials of [AKN, §7.1, Step 2].
"""
function base_p_digits(x::Integer, p::Integer, len::Integer)
    x >= 0 || throw(ArgumentError("base_p_digits: negative argument $x"))
    d = zeros(Int, len)
    y = x
    for j in 1:len
        y, d[j] = divrem(y, p)
    end
    y == 0 || throw(ArgumentError("base_p_digits: $x does not fit in $len base-$p digits"))
    return d
end

"""
    num_f_variables(p, i) -> Int

The number `M_f + 1` of divided-power-type generators `f_0, …, f_{M_f}` of the prismatic envelope
of `R = O_K/ϖ^n` that are needed in `F^{[0, in-1]}` ([AKN, §7.1, Step 2]: `M_f = ⌊log_p(i-1)⌋` for
`i > 1`). For `i = 1` no `f`-variables are needed (`f_0` has F-weight `n = in`).
"""
num_f_variables(p::Integer, i::Integer) = i == 1 ? 0 : ilog(p, i - 1) + 1

"""
    num_g_variables(p, L) -> Int

The number `M_g + 1` of generators `g_0, …, g_{M_g}` of the prismatic envelope of `O_K` over
`W(F_q)[[z_0, z_1]]` needed in `F^{[0, L-1]}`, where `M_g = ⌊log_p(L-1)⌋` ([AKN, §7.1, Step 2, p. 55]).
Zero when `L = 1`.
"""
num_g_variables(p::Integer, L::Integer) = L <= 1 ? 0 : ilog(p, L - 1) + 1

"""
    SyntomicParams(p, f, n, i)

The integer parameters of the computation of `Z_p(i)(O_K/ϖ^n)` for `K/Q_p` with residue field
`F_q`, `q = p^f` ([AKN, §7.1, Input]).

Fields: `p`, `f`, `n`, `i`, `L = n*i` (the F-precision: everything is computed in
`F^{[0, L-1]}`), `nf = M_f + 1` and `ng = M_g + 1` (numbers of `f`- and `g`-variables).
The p-adic working precision is *not* part of the parameters (see [`working_precision`](@ref)).
"""
struct SyntomicParams
    p::Int
    f::Int
    n::Int
    i::Int
    L::Int
    nf::Int
    ng::Int
    function SyntomicParams(p::IntegerLike, f::Integer, n::Integer, i::Integer)
        Nemo.is_prime(ZZ(p)) || throw(ArgumentError("p = $p is not prime"))
        f >= 1 || throw(ArgumentError("the residue degree f must be at least 1, got $f"))
        n >= 1 || throw(
            ArgumentError("the power n of the uniformizer must be at least 1, got $n")
        )
        i >= 1 || throw(ArgumentError("the weight i must be at least 1, got $i"))
        pp = Int(p)
        L = n * i
        return new(pp, f, n, i, L, num_f_variables(pp, i), num_g_variables(pp, L))
    end
end

function Base.show(io::IO, prm::SyntomicParams)
    return print(
        io, "SyntomicParams(p=$(prm.p), f=$(prm.f), n=$(prm.n), i=$(prm.i); L=$(prm.L))"
    )
end

"""
    f_basis_exponents(prm, w) -> (c, e)

The reduced basis monomial `z_0^c ∏_u f_u^{e_u}` of the prismatic envelope of `O_K/ϖ^n` of
F-weight `w` ([AKN, §7.1, Step 2]): `c = w mod n` and `e` are the base-`p` digits of `⌊w/n⌋`
(`0 ≤ e_u < p`), using that `z_0` has F-weight 1 and `f_u` has F-weight `n p^u`.
There is exactly one such monomial in each F-weight `0 ≤ w < L`.
"""
function f_basis_exponents(prm::SyntomicParams, w::Integer)
    0 <= w < prm.L || throw(ArgumentError("F-weight $w outside [0, $(prm.L))"))
    s, c = divrem(w, prm.n)
    return c, base_p_digits(s, prm.p, prm.nf)
end

"""
    g_basis_exponents(prm, w) -> e

The reduced `W(F_q)[[z_0]]`-basis monomial `∏_u g_u^{e_u}` of the prismatic envelope of `O_K`
over `W(F_q)[[z_0, z_1]]` of F-weight `w` ([AKN, §7.1, Step 2]): the base-`p` digits of `w`
(`g_u` has F-weight `p^u`).
"""
function g_basis_exponents(prm::SyntomicParams, w::Integer)
    0 <= w < prm.L || throw(ArgumentError("F-weight $w outside [0, $(prm.L))"))
    return base_p_digits(w, prm.p, prm.ng)
end

"""
    f_weight(prm, c, e) -> Int

The F-weight `c + n Σ_u p^u e_u` of the monomial `z_0^c ∏_u f_u^{e_u}` ([AKN, §7.1, Step 2]).
"""
function f_weight(prm::SyntomicParams, c::Integer, e::AbstractVector{<:Integer})
    return c + prm.n * sum((e[u + 1] * prm.p^u for u in 0:(length(e) - 1)); init=0)
end

"""
    g_weight(p, e) -> Int

The F-weight `Σ_u p^u e_u` of the monomial `∏_u g_u^{e_u}`.
"""
function g_weight(p::Integer, e::AbstractVector{<:Integer})
    return sum((e[u + 1] * p^u for u in 0:(length(e) - 1)); init=0)
end
