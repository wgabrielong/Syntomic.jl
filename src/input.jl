# Input: the finite chain ring R = O_K/ϖ^n = W(F_q)[z]/(E(z), z^n).
#
# [AKN24, §7.1, Input, p. 53]: integers q = p^f, i, n and an Eisenstein polynomial E(z) ∈ W(F_q)[z].
#
# All accepted input forms are converted once into exact, plain integer data (so that a
# `FiniteChainRing` can be sent to other processes and rebuilt at any p-adic precision):
# the monic integer polynomial h̃ presenting W(F_q) = Z_p[x]/(h̃) ([AKN24, §7.1, Step 2, p. 54]) and
# the integer coordinates of the coefficients of E. Since p = (unit)·z^e in O_K, the ring
# R = W(F_q)[z]/(E, z^n) only depends on E modulo p^⌈n/e⌉; an input coefficient known to lower
# precision is rejected, and a coefficient known to higher precision is replaced by an integer lift.
# The stored integers define an Eisenstein polynomial presenting the same ring R; the precision to
# which they agree with the input is recorded, so that the ring is never used for a larger n.

"""
    FiniteChainRing

The finite chain ring `R = O_K/ϖ^n = W(F_q)[z]/(E(z), z^n)` for an Eisenstein polynomial
`E ∈ W(F_q)[z]` of degree `e` ([AKN24, §7.1, Input, p. 53]), stored as exact integer data:

- `p`, `f` (so `q = p^f`), `n`, `e = deg E`;
- `htilde::Vector{BigInt}`: the monic polynomial `h̃ = h_0 + … + x^f` with `W(F_q) = Z_p[x]/(h̃)`;
- `E::Vector{Vector{BigInt}}`: `E[k+1]` holds the coordinates, in the basis `1, x, …, x^{f-1}`, of
  the coefficient of `z^k` in `E` (not normalized);
- `Eprec::Int`: the p-adic precision to which the stored `E` agrees with the input (`typemax(Int)`
  for exact input). Since `R` depends only on `E` modulo `p^⌈n/e⌉`, `Eprec ≥ max(⌈n/e⌉, 2)` is
  required (see [`with_power`](@ref));
- `label::String`: an optional description.

Construct it with [`finite_chain_ring`](@ref), [`zpn`](@ref) or [`truncated_witt_vectors`](@ref).
"""
struct FiniteChainRing
    p::Int
    f::Int
    n::Int
    e::Int
    htilde::Vector{BigInt}
    E::Vector{Vector{BigInt}}
    Eprec::Int
    label::String
end

"""
    prime(R::FiniteChainRing) -> Int

The residue characteristic `p`.
"""
Hecke.prime(R::FiniteChainRing) = R.p

"""
    ramification_index(R::FiniteChainRing) -> Int

The ramification index `e = deg E` of `K/Q_p`.
"""
Hecke.ramification_index(R::FiniteChainRing) = R.e

"""
    inertia_degree(R::FiniteChainRing) -> Int

The residue degree `f` of `K/Q_p` (so the residue field of `R` is `F_q`, `q = p^f`).
"""
Hecke.inertia_degree(R::FiniteChainRing) = R.f

"""
    residue_field(R::FiniteChainRing) -> (F, red)

The residue field `F_q = F_p[x]/(h̃ mod p)` of `R = O_K/ϖ^n`, `q = p^f`, where
`W(F_q) = Z_p[x]/(h̃)` is the presentation of `R` ([AKN24, §7.1, Step 2, p. 54]).

As for the `residue_field` of Hecke's p-adic and q-adic fields, a pair is returned:

- `F`, a Nemo `FqField` of order `q` with defining polynomial `h̃ mod p`, whose generator
  `gen(F)` is the image of `x` (so `gen(F) = 0` for `zpn(p, n)`, where `h̃ = x`);
- `red`, the reduction map `Z[x] → F_q`, `x ↦ gen(F)`, from the polynomial ring
  `polynomial_ring(ZZ, :x)[1]`: an element of `W(F_q)` with integer coordinates `c` in the basis
  `1, x, …, x^{f-1}` (such as a coefficient `R.E[k+1]` of `E`) has residue `red(Zx(c))`, where
  `Zx = domain(red)`. `preimage(red, a)` is a lift of `a` of degree `< f`.

When `h̃` is the Conway polynomial (the default for `f > 1`), `F` has the same defining polynomial
as the residue field of `qadic_field(p, f)`.
"""
function Hecke.residue_field(R::FiniteChainRing)
    Fp = Nemo.GF(R.p)
    Fpx, _ = polynomial_ring(Fp, :x)
    F, x = finite_field(Fpx([Fp(c) for c in R.htilde]), :x)
    Zx, _ = polynomial_ring(ZZ, :x)
    reduce_mod_p(g) = sum((F(coeff(g, k)) * x^k for k in 0:degree(g)); init=zero(F))
    red = MapFromFunc(Zx, F, reduce_mod_p, a -> lift(Zx, a))
    return F, red
end

function Base.show(io::IO, R::FiniteChainRing)
    q = R.f == 1 ? "$(R.p)" : "$(R.p)^$(R.f)"
    s = "O_K/ϖ^$(R.n) with E of degree $(R.e) over W(F_$q)"
    isempty(R.label) || (s *= " [$(R.label)]")
    return print(io, s)
end

function Base.:(==)(R1::FiniteChainRing, R2::FiniteChainRing)
    return (R1.p, R1.f, R1.n, R1.htilde, R1.E) == (R2.p, R2.f, R2.n, R2.htilde, R2.E)
end

Base.hash(R::FiniteChainRing, h::UInt) = hash((R.p, R.f, R.n, R.htilde, R.E), h)

# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

"""
    required_input_precision(n, e) -> Int

The p-adic precision to which the coefficients of `E` must be known: `max(⌈n/e⌉, 2)`
(`R` depends only on `E` modulo `p^⌈n/e⌉`, and precision `2` is needed to see that the constant
term has valuation exactly `1`).
"""
required_input_precision(n::Integer, e::Integer) = max(cld(n, e), 2)

# The p-adic valuation of an element of W(F_q) given by integer coordinates (p is a uniformizer
# of W(F_q) and 1, x, …, x^{f-1} reduce to an F_p-basis of F_q).
function _coordinate_valuation(p::Integer, c::Vector{BigInt})
    all(iszero, c) && return typemax(Int)
    return minimum(padic_valuation(p, x) for x in c if !iszero(x))
end

function _validate_eisenstein(p::Integer, f::Integer, E::Vector{Vector{BigInt}}, n::Integer)
    e = length(E) - 1
    e >= 1 || throw(ArgumentError("the Eisenstein polynomial must have degree at least 1"))
    all(length(c) == f for c in E) ||
        throw(ArgumentError("every coefficient of E must have $f coordinates"))
    _coordinate_valuation(p, E[end]) == 0 ||
        throw(ArgumentError("E is not Eisenstein: its leading coefficient is not a unit"))
    _coordinate_valuation(p, E[1]) == 1 || throw(
        ArgumentError("E is not Eisenstein: its constant term does not have valuation 1"),
    )
    for k in 2:e
        _coordinate_valuation(p, E[k]) >= 1 || throw(
            ArgumentError(
                "E is not Eisenstein: the coefficient of z^$(k - 1) is not divisible by $p",
            ),
        )
    end
    n >= 1 ||
        throw(ArgumentError("the power n of the uniformizer must be at least 1, got $n"))
    return e
end

function _validate_htilde(p::Integer, htilde::Vector{BigInt})
    f = length(htilde) - 1
    f >= 1 || throw(ArgumentError("h̃ must have degree at least 1"))
    isone(htilde[end]) || throw(ArgumentError("h̃ must be monic"))
    if f > 1
        Fp = Nemo.Native.GF(p)
        Fpx, _ = polynomial_ring(Fp, :x; cached=false)
        Nemo.is_irreducible(Fpx([Fp(c) for c in htilde])) || throw(ArgumentError("h̃ = $htilde is not irreducible modulo $p"))
    end
    return f
end

"""
    finite_chain_ring(p, htilde, E, n; Eprec=typemax(Int), label="") -> FiniteChainRing

The ring `O_K/ϖ^n = W(F_q)[z]/(E, z^n)` of [AKN24, §7.1, Input, p. 53], from integer data
(the basic constructor): `h̃` as its coefficient list `[h_0, …, h_{f-1}, 1]` and `E` as the
list (low to high degree) of the coordinate vectors of its coefficients. For `f = 1` use
`htilde = [0, 1]`. If the coordinates of `E` are only known modulo `p^Eprec`, pass `Eprec`;
it must be at least `max(⌈n/e⌉, 2)`.
"""
function finite_chain_ring(
    p::Integer,
    htilde::AbstractVector{<:Integer},
    E::AbstractVector{<:AbstractVector{<:Integer}},
    n::Integer;
    Eprec::Integer=typemax(Int),
    label::AbstractString="",
)
    Nemo.is_prime(ZZ(p)) || throw(ArgumentError("p = $p is not prime"))
    h = BigInt[BigInt(c) for c in htilde]
    f = _validate_htilde(p, h)
    EE = [BigInt[BigInt(c) for c in coeff] for coeff in E]
    e = _validate_eisenstein(p, f, EE, n)
    req = required_input_precision(n, e)
    Eprec >= req || throw(
        ArgumentError(
            "E is known only modulo p^$Eprec, but n = $n requires precision $req " *
            "(R = W(F_q)[z]/(E, z^n) depends on E modulo p^⌈n/e⌉); give E to higher precision",
        ),
    )
    return FiniteChainRing(Int(p), f, Int(n), e, h, EE, Int(Eprec), String(label))
end

# ---------------------------------------------------------------------------------------------
# Conversions from Nemo/Hecke p-adic and q-adic data
# ---------------------------------------------------------------------------------------------

# A balanced integer lift of a p-adic number known modulo p^prec (so that -1 stays -1).
function _balanced_lift(x::PadicFieldElem)
    valuation(x) >= 0 ||
        throw(ArgumentError("the coefficients of E must be p-adic integers, got $x"))
    prec = precision(x)
    m = ZZ(prime(parent(x)))^prec
    r = mod(lift(ZZ, x), m)
    2 * r > m && (r -= m)
    return BigInt(r)
end

function _check_precision(x::PadicFieldElem, required::Integer, what::AbstractString)
    precision(x) >= required || throw(
        ArgumentError(
            "$what is known only to precision $(precision(x)), but precision $required is required " *
            "(R = W(F_q)[z]/(E, z^n) depends on E modulo p^⌈n/e⌉)",
        ),
    )
    return nothing
end

function _integer_htilde(K::QadicField)
    return BigInt[BigInt(lift(ZZ, c)) for c in coefficients(defining_polynomial(K))]
end

"""
    finite_chain_ring(E, n; label="") -> FiniteChainRing

The ring `O_K/ϖ^n` of [AKN24, §7.1, Input, p. 53] for `K = Frac(W(F_q))[z]/(E)`, where `E` is an
Eisenstein polynomial over a q-adic field `qadic_field(p, f)` (any defining polynomial; `W(F_q)`
is presented by it) or over a p-adic field `padic_field(p)`. Every coefficient must be known to
p-adic precision at least `max(⌈n/e⌉, 2)`.
"""
function finite_chain_ring(
    E::PolyRingElem{QadicFieldElem}, n::Integer; label::AbstractString=""
)
    K = base_ring(parent(E))
    p = Int(prime(K))
    f = degree(K)
    e = degree(E)
    e >= 1 || throw(ArgumentError("the Eisenstein polynomial must have degree at least 1"))
    req = required_input_precision(n, e)
    coords = Vector{Vector{BigInt}}(undef, e + 1)
    Eprec = typemax(Int)
    for k in 0:e
        c = coeff(E, k)
        coords[k + 1] = map(0:(f - 1)) do d
            x = coeff(c, d)
            _check_precision(x, req, "the coefficient of z^$k of E")
            _balanced_lift(x)
        end
        Eprec = min(Eprec, minimum(precision(coeff(c, d)) for d in 0:(f - 1)))
    end
    return finite_chain_ring(p, _integer_htilde(K), coords, n; Eprec=Eprec, label=label)
end

function finite_chain_ring(
    E::PolyRingElem{PadicFieldElem}, n::Integer; label::AbstractString=""
)
    K = base_ring(parent(E))
    p = Int(prime(K))
    e = degree(E)
    e >= 1 || throw(ArgumentError("the Eisenstein polynomial must have degree at least 1"))
    req = required_input_precision(n, e)
    coords = map(0:e) do k
        x = coeff(E, k)
        _check_precision(x, req, "the coefficient of z^$k of E")
        [_balanced_lift(x)]
    end
    Eprec = minimum(precision(coeff(E, k)) for k in 0:e)
    return finite_chain_ring(p, [0, 1], coords, n; Eprec=Eprec, label=label)
end

"""
    finite_chain_ring(L::Hecke.LocalField, n; label="") -> FiniteChainRing

The ring `O_L/ϖ^n` of [AKN24, §7.1, Input, p. 53] for a totally ramified extension `L` of a q-adic
or p-adic field given by an Eisenstein polynomial (as returned by `eisenstein_extension`).
"""
function finite_chain_ring(L::Hecke.LocalField, n::Integer; label::AbstractString="")
    return finite_chain_ring(defining_polynomial(L), n; label=label)
end

"""
    finite_chain_ring(E::ZZPolyRingElem, n; p, f=1, htilde=nothing, label="") -> FiniteChainRing

The ring `W(F_q)[z]/(E, z^n)` of [AKN24, §7.1, Input, p. 53] for an Eisenstein polynomial
`E` with integer coefficients (so `E ∈ Z_p[z] ⊆ W(F_q)[z]`). `W(F_q)` is presented by
`htilde` if it is given (then `f` defaults to its degree), and otherwise by the Conway
polynomial used by `qadic_field(p, f)`.
"""
function finite_chain_ring(
    E::ZZPolyRingElem,
    n::Integer;
    p::Integer,
    f::Union{Nothing,Integer}=nothing,
    htilde::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    label::AbstractString="",
)
    h = if htilde === nothing
        default_htilde(p, something(f, 1))
    else
        BigInt[BigInt(c) for c in htilde]
    end
    ff = length(h) - 1
    (f === nothing || f == ff) || throw(ArgumentError("h̃ has degree $ff, expected f = $f"))
    coords = [vcat(BigInt(coeff(E, k)), zeros(BigInt, ff - 1)) for k in 0:degree(E)]
    return finite_chain_ring(p, h, coords, n; label=label)
end

"""
    default_htilde(p, f) -> Vector{BigInt}

The monic integer polynomial presenting `W(F_q)`, `q = p^f`, used by default: `[0, 1]` for
`f = 1`, and the Conway polynomial used by `qadic_field(p, f)` otherwise.
"""
function default_htilde(p::Integer, f::Integer)
    Nemo.is_prime(ZZ(p)) || throw(ArgumentError("p = $p is not prime"))
    f >= 1 || throw(ArgumentError("the residue degree f must be at least 1, got $f"))
    f == 1 && return BigInt[0, 1]
    K, _ = qadic_field(p, f; precision=4, cached=false)
    return _integer_htilde(K)
end

"""
    finite_chain_ring(coeffs; p, f, n, basis=:conway, htilde=nothing, precision=nothing,
                      label="") -> FiniteChainRing

The ring `W(F_q)[z]/(E, z^n)` of [AKN24, §7.1, Input, p. 53] where the coefficient of `z^k`
in `E` is given by `coeffs[k+1]`, an integer, a `ZZPolyRingElem` or a vector of integers
`[a_0, a_1, …]` standing for `Σ_j a_j y^j`, where the generator `y` is

- `basis = :conway`: the generator `x` of `W(F_q) = Z_p[x]/(h̃)` (by default `h̃` is the Conway
  polynomial of `qadic_field(p, f)`); `E` is then exact;
- `basis = :teichmuller`: the Teichmüller lift `ζ = teichmuller(x)` of the image of `x` in
  `F_q`, a primitive `(q-1)`-st root of unity when `h̃` is a Conway polynomial. `ζ` is computed
  modulo `p^precision`, by default to the precision `max(⌈n/e⌉, 2)` needed for `n`; pass a larger
  `precision` to use the ring with larger powers `n` (see [`with_power`](@ref)).
"""
function finite_chain_ring(
    coeffs::AbstractVector;
    p::Integer,
    f::Integer,
    n::Integer,
    basis::Symbol=:conway,
    htilde::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    precision::Union{Nothing,Integer}=nothing,
    label::AbstractString="",
)
    basis in (:conway, :teichmuller) ||
        throw(ArgumentError("basis must be :conway or :teichmuller, got :$basis"))
    h = htilde === nothing ? default_htilde(p, f) : BigInt[BigInt(c) for c in htilde]
    length(h) - 1 == f ||
        throw(ArgumentError("h̃ has degree $(length(h) - 1), expected f = $f"))
    e = length(coeffs) - 1
    e >= 1 || throw(ArgumentError("the Eisenstein polynomial must have degree at least 1"))
    polys = [_as_integer_coefficients(c) for c in coeffs]
    basis == :conway && return finite_chain_ring(
        p, h, [_reduce_mod_htilde(a, h) for a in polys], n; label=label
    )
    P = something(precision, required_input_precision(n, e))
    R = GaloisRing(p, h, P)
    zeta = teichmuller_lift(R, w_gen(R))
    gens = [w_pow(R, zeta, j) for j in 0:(max(maximum(length, polys), 1) - 1)]
    coords = map(polys) do a
        acc = w_zero(R)
        for (j, aj) in enumerate(a)
            acc = w_add(R, acc, w_mul_int(R, gens[j], aj))
        end
        [_balanced_residue(c, R.pN) for c in acc]
    end
    return finite_chain_ring(p, h, coords, n; Eprec=P, label=label)
end

# The coordinates of Σ_j a_j x^j in W(F_q) = Z_p[x]/(h̃), computed exactly over Z (h̃ is monic).
function _reduce_mod_htilde(a::Vector{BigInt}, h::Vector{BigInt})
    f = length(h) - 1
    c = vcat(a, zeros(BigInt, max(f - length(a), 0)))
    for k in (length(c) - 1):-1:f              # x^k = -x^{k-f} (h_0 + … + h_{f-1} x^{f-1})
        t = c[k + 1]
        iszero(t) && continue
        c[k + 1] = 0
        for d in 0:(f - 1)
            c[k - f + d + 1] -= t * h[d + 1]
        end
    end
    return c[1:f]
end

_as_integer_coefficients(c::Integer) = BigInt[c]
_as_integer_coefficients(c::ZZRingElem) = BigInt[BigInt(c)]
function _as_integer_coefficients(c::ZZPolyRingElem)
    return BigInt[BigInt(coeff(c, j)) for j in 0:degree(c)]
end
_as_integer_coefficients(c::AbstractVector{<:Integer}) = BigInt[BigInt(x) for x in c]

function _balanced_residue(c::ZZRingElem, m::ZZRingElem)
    r = mod(c, m)
    2 * r > m && (r -= m)
    return BigInt(r)
end

"""
    teichmuller_lift(R::GaloisRing, a::WElem) -> WElem

The Teichmüller lift of the residue of `a` in `F_q`: the unique `(q-1)`-st root of unity (or
`0`) congruent to `a` modulo `p`, computed as the limit of `a^{q^k}`.
"""
function teichmuller_lift(R::GaloisRing, a::WElem)
    q = ZZ(R.p)^R.f
    t = a
    for _ in 1:(R.N + 1)
        t = w_pow(R, t, q)
    end
    return t
end

"""
    zpn(p, n) -> FiniteChainRing

The ring `Z/p^n` (`K = Q_p`, `E = z + p`).
"""
zpn(p::Integer, n::Integer) = finite_chain_ring(p, [0, 1], [[p], [1]], n; label="Z/$p^$n")

"""
    truncated_witt_vectors(p, f, n) -> FiniteChainRing

The ring `W_n(F_q) = W(F_q)/p^n`, `q = p^f` (`K` unramified, `E = z + p`).
"""
function truncated_witt_vectors(p::Integer, f::Integer, n::Integer)
    h = default_htilde(p, f)
    E = [vcat(BigInt(p), zeros(BigInt, f - 1)), vcat(BigInt(1), zeros(BigInt, f - 1))]
    return finite_chain_ring(p, h, E, n; label="W_$n(F_$p^$f)")
end

"""
    finite_chain_ring(Zm::Union{ZZModRing,zzModRing}) -> FiniteChainRing

The ring `Z/p^n` given as a Nemo residue ring `residue_ring(ZZ, p^n)` (the case `K = Q_p`,
`E = z + p` of [AKN24, §7.1, Input, p. 53]).
"""
function finite_chain_ring(Zm::Union{ZZModRing,zzModRing})
    m = ZZ(modulus(Zm))
    ok, n, p = is_prime_power_with_data(m)
    ok || throw(ArgumentError("the modulus $m is not a prime power"))
    return zpn(Int(p), n)
end

"""
    with_power(R::FiniteChainRing, n) -> FiniteChainRing

The ring `O_K/ϖ^n` for the same `K` (same `E`) and a different power `n`. Throws an
`ArgumentError` if `E` is not known to the precision `max(⌈n/e⌉, 2)` that `n` requires.
"""
function with_power(R::FiniteChainRing, n::Integer)
    return finite_chain_ring(R.p, R.htilde, R.E, n; Eprec=R.Eprec, label=R.label)
end
