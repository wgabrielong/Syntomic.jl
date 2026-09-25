# Low-level kernels for polynomials in z with coefficients in W(F_q)/p^N.
#
# A "W-polynomial" is a vector of f polynomials A = (A_0, …, A_{f-1}) in (Z/p^N)[z], standing for
# Σ_d A_d(z) x^d. These kernels are shared by the power-series type `WSeries` and by the
# envelope elements; they are the innermost loops of the whole computation.
#
# A product of W-polynomials is computed in two stages: the 2f-1 products P_k = Σ_{d+e=k} A_d B_e
# in (Z/p^N)[z] (schoolbook, Karatsuba in x or Kronecker packing, see `wpoly_mullow`), then the
# reduction of x^k, k ≥ f, modulo h̃ in the presentation W(F_q) = Z_p[x]/(h̃) of
# [AKN, §7.1, Step 2, p. 54].
#
# Ownership rule for the in-place functions (names ending in `!`): a polynomial is mutated only by
# the function that allocated it, or by a callee it hands it to, and only until it is stored in a
# shared object (a `WSeries`, `FElem`, `GElem` or a memo table). Scratch space (`WScratch`) is
# allocated by the function that uses it and never shared between tasks.

# ---------------------------------------------------------------------------------------------
# In-place FLINT primitives
# ---------------------------------------------------------------------------------------------

# The libflint functions called directly by the in-place primitives below.
const _FLINT_SYMBOLS = (
    :fmpz_mod_poly_mullow,
    :fmpz_mod_poly_scalar_addmul_fmpz,
    :fmpz_mod_poly_shift_left,
    :fmpz_mod_poly_add_series,
    :fmpz_mod_poly_set_trunc,
    :fmpz_mod_poly_get_fmpz_poly,
    :fmpz_mod_poly_set_fmpz_poly,
    :fmpz_poly_mullow,
    :fmpz_poly_scalar_mul_2exp,
    :fmpz_poly_scalar_fdiv_2exp,
    :fmpz_poly_truncate,
    :_fmpz_poly_normalise,
)

"""
    flint_inplace_available() -> Bool

Whether the in-place FLINT calls of the kernels can be made: Nemo keeps the `fmpz_mod` context of
`Z/p^N` in the field `ninv` of the base ring of a polynomial ring over `Z/p^N` (this is how Nemo
itself calls `fmpz_mod_poly_mullow`), and `libflint` exports the functions used. Otherwise the
kernels fall back to Nemo's allocating operations, with the same results.
"""
function flint_inplace_available()
    isdefined(Nemo, :fmpz_mod_ctx_struct) || return false
    isdefined(Nemo, :libflint) || return false
    hasfield(ZZModRing, :ninv) || return false
    fieldtype(ZZModRing, :ninv) === Nemo.fmpz_mod_ctx_struct || return false
    hasfield(ZZModPolyRing, :base_ring) || return false
    hasfield(ZZModPolyRingElem, :parent) || return false
    try
        lib = Base.Libc.Libdl.dlopen(Nemo.libflint)
        return all(s -> Base.Libc.Libdl.dlsym_e(lib, s) != C_NULL, _FLINT_SYMBOLS)
    catch
        return false
    end
end

"""
    FLINT_INPLACE

Whether the kernels use in-place FLINT calls (see [`flint_inplace_available`](@ref)); fixed when
the package is compiled.
"""
const FLINT_INPLACE = flint_inplace_available()

# The fmpz_mod context of the coefficient ring of a polynomial over Z/p^N.
_fmpz_mod_ctx(a::ZZModPolyRingElem) = a.parent.base_ring.ninv

"""
    poly_mullow!(z, x, y, n) -> ZZModPolyRingElem

`x·y mod z^n` for polynomials over `Z/p^N`, written into `z` (which may alias `x` or `y`) by
FLINT's `fmpz_mod_poly_mullow`. Returns `z`, or a new polynomial if in-place calls are unavailable
(see [`FLINT_INPLACE`](@ref)): callers always continue with the returned value.
"""
poly_mullow!(z::ZZModPolyRingElem, x::ZZModPolyRingElem, y::ZZModPolyRingElem, n::Integer) =
    _poly_mullow!(Val(FLINT_INPLACE), z, x, y, n)

function _poly_mullow!(
    ::Val{true},
    z::ZZModPolyRingElem,
    x::ZZModPolyRingElem,
    y::ZZModPolyRingElem,
    n::Integer,
)
    @ccall Nemo.libflint.fmpz_mod_poly_mullow(
        z::Ref{ZZModPolyRingElem},
        x::Ref{ZZModPolyRingElem},
        y::Ref{ZZModPolyRingElem},
        max(n, 0)::Int,
        _fmpz_mod_ctx(x)::Ref{Nemo.fmpz_mod_ctx_struct},
    )::Nothing
    return z
end

_poly_mullow!(::Val{false}, z, x, y, n::Integer) = mullow(x, y, Int(max(n, 0)))

"""
    poly_addmul!(z, x, c::ZZRingElem) -> ZZModPolyRingElem

`z + c·x` for polynomials over `Z/p^N` and an integer `c`, written into `z` (FLINT's
`fmpz_mod_poly_scalar_addmul_fmpz`, or an addition for `c = 1`, which is several times cheaper
than a scalar multiplication with the reduction of every coefficient). Returns `z`, or a new
polynomial without in-place calls.
"""
function poly_addmul!(z::ZZModPolyRingElem, x::ZZModPolyRingElem, c::ZZRingElem)
    isone(c) && return add!(z, z, x)
    return _poly_addmul!(Val(FLINT_INPLACE), z, x, c)
end

function _poly_addmul!(
    ::Val{true}, z::ZZModPolyRingElem, x::ZZModPolyRingElem, c::ZZRingElem
)
    @ccall Nemo.libflint.fmpz_mod_poly_scalar_addmul_fmpz(
        z::Ref{ZZModPolyRingElem},
        x::Ref{ZZModPolyRingElem},
        c::Ref{ZZRingElem},
        _fmpz_mod_ctx(x)::Ref{Nemo.fmpz_mod_ctx_struct},
    )::Nothing
    return z
end

_poly_addmul!(::Val{false}, z, x, c::ZZRingElem) = z + x * c

"""
    poly_shift_left!(z, s) -> ZZModPolyRingElem

`z·z^s` (the variable is also called `z`), in place. Returns `z`, or a new polynomial without
in-place calls.
"""
poly_shift_left!(z::ZZModPolyRingElem, s::Integer) =
    _poly_shift_left!(Val(FLINT_INPLACE), z, s)

function _poly_shift_left!(::Val{true}, z::ZZModPolyRingElem, s::Integer)
    s == 0 && return z
    @ccall Nemo.libflint.fmpz_mod_poly_shift_left(
        z::Ref{ZZModPolyRingElem},
        z::Ref{ZZModPolyRingElem},
        Int(s)::Int,
        _fmpz_mod_ctx(z)::Ref{Nemo.fmpz_mod_ctx_struct},
    )::Nothing
    return z
end

_poly_shift_left!(::Val{false}, z, s::Integer) = s == 0 ? z : shift_left(z, Int(s))

"""
    poly_add_trunc!(z, x, y, n) -> ZZModPolyRingElem

`(x + y) mod z^n`, written into `z` (which may alias `x` or `y`; FLINT's
`fmpz_mod_poly_add_series`). Returns `z`, or a new polynomial without in-place calls.
"""
poly_add_trunc!(
    z::ZZModPolyRingElem, x::ZZModPolyRingElem, y::ZZModPolyRingElem, n::Integer
) = _poly_add_trunc!(Val(FLINT_INPLACE), z, x, y, n)

function _poly_add_trunc!(
    ::Val{true},
    z::ZZModPolyRingElem,
    x::ZZModPolyRingElem,
    y::ZZModPolyRingElem,
    n::Integer,
)
    @ccall Nemo.libflint.fmpz_mod_poly_add_series(
        z::Ref{ZZModPolyRingElem},
        x::Ref{ZZModPolyRingElem},
        y::Ref{ZZModPolyRingElem},
        max(n, 0)::Int,
        _fmpz_mod_ctx(x)::Ref{Nemo.fmpz_mod_ctx_struct},
    )::Nothing
    return z
end

_poly_add_trunc!(::Val{false}, z, x, y, n::Integer) = truncate(x + y, Int(max(n, 0)))

"""
    poly_set_trunc!(z, x, n) -> ZZModPolyRingElem

`x mod z^n`, written into `z` (FLINT's `fmpz_mod_poly_set_trunc`). Returns `z`, or a new
polynomial without in-place calls.
"""
poly_set_trunc!(z::ZZModPolyRingElem, x::ZZModPolyRingElem, n::Integer) =
    _poly_set_trunc!(Val(FLINT_INPLACE), z, x, n)

function _poly_set_trunc!(
    ::Val{true}, z::ZZModPolyRingElem, x::ZZModPolyRingElem, n::Integer
)
    @ccall Nemo.libflint.fmpz_mod_poly_set_trunc(
        z::Ref{ZZModPolyRingElem},
        x::Ref{ZZModPolyRingElem},
        max(n, 0)::Int,
        _fmpz_mod_ctx(x)::Ref{Nemo.fmpz_mod_ctx_struct},
    )::Nothing
    return z
end

_poly_set_trunc!(::Val{false}, z, x, n::Integer) = truncate(x, Int(max(n, 0)))

"""
    poly_lift_trunc!(zz, a, n) -> ZZPolyRingElem

The lift of `a mod z^n` to `Z[z]` (coefficients in `[0, p^N)`), written into `zz`. Returns `zz`,
or a new polynomial without in-place calls.
"""
poly_lift_trunc!(zz::ZZPolyRingElem, a::ZZModPolyRingElem, n::Integer) =
    _poly_lift_trunc!(Val(FLINT_INPLACE), zz, a, n)

function _poly_lift_trunc!(
    ::Val{true}, zz::ZZPolyRingElem, a::ZZModPolyRingElem, n::Integer
)
    @ccall Nemo.libflint.fmpz_mod_poly_get_fmpz_poly(
        zz::Ref{ZZPolyRingElem},
        a::Ref{ZZModPolyRingElem},
        _fmpz_mod_ctx(a)::Ref{Nemo.fmpz_mod_ctx_struct},
    )::Nothing
    @ccall Nemo.libflint.fmpz_poly_truncate(
        zz::Ref{ZZPolyRingElem}, max(n, 0)::Int
    )::Nothing
    return zz
end

function _poly_lift_trunc!(::Val{false}, zz::ZZPolyRingElem, a, n::Integer)
    return truncate(lift(parent(zz), a), Int(max(n, 0)))
end

"""
    poly_reduce!(a, zz) -> ZZModPolyRingElem

The reduction of the integer polynomial `zz` modulo `p^N`, written into `a` (a polynomial over
`Z/p^N`). Returns `a`, or a new polynomial without in-place calls.
"""
poly_reduce!(a::ZZModPolyRingElem, zz::ZZPolyRingElem) =
    _poly_reduce!(Val(FLINT_INPLACE), a, zz)

function _poly_reduce!(::Val{true}, a::ZZModPolyRingElem, zz::ZZPolyRingElem)
    @ccall Nemo.libflint.fmpz_mod_poly_set_fmpz_poly(
        a::Ref{ZZModPolyRingElem},
        zz::Ref{ZZPolyRingElem},
        _fmpz_mod_ctx(a)::Ref{Nemo.fmpz_mod_ctx_struct},
    )::Nothing
    return a
end

function _poly_reduce!(::Val{false}, a::ZZModPolyRingElem, zz::ZZPolyRingElem)
    return map_coefficients(base_ring(parent(a)), zz; parent=parent(a))
end

# Integer polynomials (no context needed): x·y mod z^n, x·2^e and ⌊x/2^e⌋, written into z.
function _zz_mullow!(z::ZZPolyRingElem, x::ZZPolyRingElem, y::ZZPolyRingElem, n::Integer)
    return _zz_mullow!(Val(FLINT_INPLACE), z, x, y, n)
end

function _zz_mullow!(
    ::Val{true}, z::ZZPolyRingElem, x::ZZPolyRingElem, y::ZZPolyRingElem, n::Integer
)
    @ccall Nemo.libflint.fmpz_poly_mullow(
        z::Ref{ZZPolyRingElem}, x::Ref{ZZPolyRingElem}, y::Ref{ZZPolyRingElem}, Int(n)::Int
    )::Nothing
    return z
end

_zz_mullow!(::Val{false}, z, x, y, n::Integer) = mullow(x, y, Int(n))

function _zz_mul_2exp!(z::ZZPolyRingElem, x::ZZPolyRingElem, e::Integer)
    return _zz_mul_2exp!(Val(FLINT_INPLACE), z, x, e)
end

function _zz_mul_2exp!(::Val{true}, z::ZZPolyRingElem, x::ZZPolyRingElem, e::Integer)
    @ccall Nemo.libflint.fmpz_poly_scalar_mul_2exp(
        z::Ref{ZZPolyRingElem}, x::Ref{ZZPolyRingElem}, UInt(e)::UInt
    )::Nothing
    return z
end

_zz_mul_2exp!(::Val{false}, z, x, e::Integer) = x * ZZ(2)^e

function _zz_fdiv_2exp!(z::ZZPolyRingElem, x::ZZPolyRingElem, e::Integer)
    return _zz_fdiv_2exp!(Val(FLINT_INPLACE), z, x, e)
end

function _zz_fdiv_2exp!(::Val{true}, z::ZZPolyRingElem, x::ZZPolyRingElem, e::Integer)
    @ccall Nemo.libflint.fmpz_poly_scalar_fdiv_2exp(
        z::Ref{ZZPolyRingElem}, x::Ref{ZZPolyRingElem}, UInt(e)::UInt
    )::Nothing
    # the leading coefficients may have become zero
    @ccall Nemo.libflint._fmpz_poly_normalise(z::Ref{ZZPolyRingElem})::Nothing
    return z
end

function _zz_fdiv_2exp!(::Val{false}, z, x, e::Integer)
    return map_coefficients(c -> fdiv(c, ZZ(2)^e), x; parent=parent(x))
end

# ---------------------------------------------------------------------------------------------
# W-polynomials: basic operations
# ---------------------------------------------------------------------------------------------

"""
    wpoly_zero(R) -> Vector{ZZModPolyRingElem}

The zero W-polynomial (`f` new zero polynomials, which the caller owns).
"""
wpoly_zero(R::GaloisRing) = [zero(R.Znz) for _ in 1:(R.f)]

"""
    wpoly_add!(A, B) -> A

`A + B` for W-polynomials, written into the polynomials of `A` (which the caller owns). Only the
first `length(A)` entries of `B` are used (so `B` may be the scratch vector of a product in `x`).
"""
function wpoly_add!(A::Vector{ZZModPolyRingElem}, B::Vector{ZZModPolyRingElem})
    for d in eachindex(A)
        iszero(B[d]) && continue
        A[d] = add!(A[d], A[d], B[d])
    end
    return A
end

"""
    wpoly_double!(A, f) -> A

`2A` for the first `f` entries of `A`, in place.
"""
function wpoly_double!(A::Vector{ZZModPolyRingElem}, f::Integer)
    for d in 1:f
        iszero(A[d]) || (A[d] = add!(A[d], A[d], A[d]))
    end
    return A
end

"""
    WScratch(R)
    WScratch(R, P)

Scratch space for the in-place W-polynomial kernels: the `2f - 1` polynomials `prods` over `Z/p^N`
for the products in `x` (`P`, if given), and polynomials over `Z/p^N` and over `Z` for the
operands and partial products of Karatsuba and Kronecker packing, created on first use. A
`WScratch` is used by one task at a time: the function that needs it allocates it and never
shares it.
"""
struct WScratch
    prods::Vector{ZZModPolyRingElem}
    tmp::Vector{ZZModPolyRingElem}
    zz::Vector{ZZPolyRingElem}
    xbal::Vector{Matrix{ZZRingElem}}
end

function WScratch(R::GaloisRing, P::Vector{ZZModPolyRingElem})
    return WScratch(P, ZZModPolyRingElem[], ZZPolyRingElem[], Matrix{ZZRingElem}[])
end

WScratch(R::GaloisRing) = WScratch(R, [zero(R.Znz) for _ in 1:(2 * R.f - 1)])

# At least n scratch polynomials over Z/p^N (resp. over Z).
function _tmp!(ws::WScratch, R::GaloisRing, n::Integer)
    while length(ws.tmp) < n
        push!(ws.tmp, zero(R.Znz))
    end
    return ws.tmp
end

function _zz!(ws::WScratch, R::GaloisRing, n::Integer)
    while length(ws.zz) < n
        push!(ws.zz, zero(R.Zz))
    end
    return ws.zz
end

# The balanced representatives in (-p^N/2, p^N/2] of the coordinates xpow of x^k (computed once
# per scratch space).
function _xbal!(ws::WScratch, R::GaloisRing)
    if isempty(ws.xbal)
        half = R.pN >> 1
        push!(ws.xbal, map(c -> c > half ? c - R.pN : c, R.xpow))
    end
    return ws.xbal[1]
end

# The number of polynomial products of Karatsuba in x (0 if not implemented for this f).
_karatsuba_products(f::Integer) =
    if f == 2
        3
    elseif f == 3
        6
    elseif f == 4
        9
    else
        0
    end

# ---------------------------------------------------------------------------------------------
# Products in x: the 2f-1 polynomials P_k = Σ_{d+e=k} A_d B_e mod z^len, written into P
# ---------------------------------------------------------------------------------------------

"""
    wpoly_products_schoolbook!(R, P, A, B, len, ws) -> P

The `2f - 1` products `P_k = Σ_{d+e=k} A_d B_e mod z^len` (`0 ≤ k ≤ 2f-2`) by schoolbook
multiplication in `x` (one polynomial product for each pair of nonzero components), written into
`P` (owned by the caller). `ws` provides scratch space.
"""
function wpoly_products_schoolbook!(
    R::GaloisRing,
    P::Vector{ZZModPolyRingElem},
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer,
    ws::WScratch,
)
    f = R.f
    T = _tmp!(ws, R, 1)
    for k in 1:(2 * f - 1)
        P[k] = zero!(P[k])
    end
    for d in 1:f
        iszero(A[d]) && continue
        for e in 1:f
            iszero(B[e]) && continue
            k = d + e - 1
            if iszero(P[k])
                P[k] = poly_mullow!(P[k], A[d], B[e], len)
            else
                T[1] = poly_mullow!(T[1], A[d], B[e], len)
                P[k] = add!(P[k], P[k], T[1])
            end
        end
    end
    return P
end

# Karatsuba for a product of two linear polynomials in x: (a0 + a1 x)(b0 + b1 x) mod z^len into
# (c0, c1, c2), with 3 polynomial products; sa and sb are scratch polynomials (returned, as they
# may be replaced when in-place calls are unavailable).
function _karatsuba_linear!(c0, c1, c2, a0, a1, b0, b1, len, sa, sb)
    c0 = poly_mullow!(c0, a0, b0, len)
    c2 = poly_mullow!(c2, a1, b1, len)
    sa = poly_add_trunc!(sa, a0, a1, len)
    if a0 === b0 && a1 === b1
        c1 = poly_mullow!(c1, sa, sa, len)
    else
        sb = poly_add_trunc!(sb, b0, b1, len)
        c1 = poly_mullow!(c1, sa, sb, len)
    end
    c1 = sub!(c1, c1, c0)
    c1 = sub!(c1, c1, c2)
    return c0, c1, c2, sa, sb
end

"""
    wpoly_products_karatsuba!(R, P, A, B, len, ws) -> P

The `2f - 1` products `P_k = Σ_{d+e=k} A_d B_e mod z^len` by Karatsuba multiplication in `x`,
for `f = 2` (3 polynomial products), `f = 3` (6 products) and `f = 4` (9 products: two levels,
`A = A' + A'' x²` with linear `A'`, `A''`), written into `P`. `ws` provides scratch space.
"""
function wpoly_products_karatsuba!(
    R::GaloisRing,
    P::Vector{ZZModPolyRingElem},
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer,
    ws::WScratch,
)
    f = R.f
    if f == 2
        T = _tmp!(ws, R, 2)
        P[1], P[2], P[3], T[1], T[2] = _karatsuba_linear!(
            P[1], P[2], P[3], A[1], A[2], B[1], B[2], len, T[1], T[2]
        )
    elseif f == 3
        _karatsuba_quadratic!(P, A, B, len, _tmp!(ws, R, 3))
    elseif f == 4
        _karatsuba_cubic!(P, A, B, len, _tmp!(ws, R, 8))
    else
        throw(ArgumentError("wpoly_products_karatsuba!: not implemented for f = $f"))
    end
    return P
end

# f = 3: with m_d = a_d b_d and m_{de} = (a_d + a_e)(b_d + b_e),
#   P_0 = m_0, P_1 = m_01 - m_0 - m_1, P_2 = m_02 - m_0 - m_2 + m_1, P_3 = m_12 - m_1 - m_2, P_4 = m_2.
function _karatsuba_quadratic!(P, A, B, len, T)
    P[1] = poly_mullow!(P[1], A[1], B[1], len)          # m_0
    P[3] = poly_mullow!(P[3], A[2], B[2], len)          # m_1
    P[5] = poly_mullow!(P[5], A[3], B[3], len)          # m_2
    P[2] = _karatsuba_pair!(P[2], A, B, 1, 2, len, T)   # m_01
    P[2] = sub!(P[2], P[2], P[1])
    P[2] = sub!(P[2], P[2], P[3])
    P[4] = _karatsuba_pair!(P[4], A, B, 2, 3, len, T)   # m_12
    P[4] = sub!(P[4], P[4], P[3])
    P[4] = sub!(P[4], P[4], P[5])
    T[3] = _karatsuba_pair!(T[3], A, B, 1, 3, len, T)   # m_02
    T[3] = sub!(T[3], T[3], P[1])
    T[3] = sub!(T[3], T[3], P[5])
    P[3] = add!(P[3], P[3], T[3])
    return P
end

# (A_d + A_e)(B_d + B_e) mod z^len into c, using T[1] and T[2] as scratch.
function _karatsuba_pair!(c, A, B, d, e, len, T)
    T[1] = poly_add_trunc!(T[1], A[d], A[e], len)
    A === B && return poly_mullow!(c, T[1], T[1], len)
    T[2] = poly_add_trunc!(T[2], B[d], B[e], len)
    return poly_mullow!(c, T[1], T[2], len)
end

# f = 4: A = A' + A'' y with y = x², A' = a_0 + a_1 x, A'' = a_2 + a_3 x (likewise B); then
#   AB = A'B' + ((A' + A'')(B' + B'') - A'B' - A''B'') y + A''B'' y²,
# each of the three products of linear polynomials by Karatsuba (3 products).
function _karatsuba_cubic!(P, A, B, len, T)
    square = A === B
    P[1], P[2], P[3], T[1], T[2] = _karatsuba_linear!(
        P[1], P[2], P[3], A[1], A[2], B[1], B[2], len, T[1], T[2]
    )
    P[5], P[6], P[7], T[1], T[2] = _karatsuba_linear!(
        P[5], P[6], P[7], A[3], A[4], B[3], B[4], len, T[1], T[2]
    )
    # the operands A' + A'' and B' + B''
    T[3] = poly_add_trunc!(T[3], A[1], A[3], len)
    T[4] = poly_add_trunc!(T[4], A[2], A[4], len)
    if !square
        T[5] = poly_add_trunc!(T[5], B[1], B[3], len)
        T[6] = poly_add_trunc!(T[6], B[2], B[4], len)
    end
    sb0, sb1 = square ? (T[3], T[4]) : (T[5], T[6])
    # the middle product M = (M_0, M_1, M_2), with M_1 written directly into P[4]
    T[7], P[4], T[8], T[1], T[2] = _karatsuba_linear!(
        T[7], P[4], T[8], T[3], T[4], sb0, sb1, len, T[1], T[2]
    )
    T[7] = sub!(T[7], T[7], P[1])
    T[7] = sub!(T[7], T[7], P[5])
    P[4] = sub!(P[4], P[4], P[2])
    P[4] = sub!(P[4], P[4], P[6])
    T[8] = sub!(T[8], T[8], P[3])
    T[8] = sub!(T[8], T[8], P[7])
    P[3] = add!(P[3], P[3], T[7])
    P[5] = add!(P[5], P[5], T[8])
    return P
end

"""
    kronecker_bits(R, m) -> Int

The slot width `b = 2·bits(p^N) + bits(f·m) + 1` of Kronecker packing in `x` for operands with at
most `m` relevant `z`-coefficients: every coefficient of a product `P_k` is a sum of at most `f·m`
products of two integers in `[0, p^N)`, hence smaller than `2^{b-1}`.
"""
kronecker_bits(R::GaloisRing, m::Integer) = 2 * nbits(R.pN) + nbits(R.f * max(m, 1)) + 1

# Pack the components of A (lifted to [0, p^N), truncated mod z^len) into
# Σ_d A_d(z) 2^{b·d} ∈ Z[z] (Horner's rule in 2^b), written into zz; t is scratch.
function _kronecker_pack!(zz, t, A, len, b)
    zz = poly_lift_trunc!(zz, A[end], len)
    for d in (length(A) - 1):-1:1
        zz = _zz_mul_2exp!(zz, zz, b)
        iszero(A[d]) && continue
        t = poly_lift_trunc!(t, A[d], len)
        zz = add!(zz, zz, t)
    end
    return zz, t
end

"""
    wpoly_products_kronecker!(R, C, A, B, len, ws) -> C

The `2f - 1` products `P_k = Σ_{d+e=k} A_d B_e mod z^len` over `Z` (from the nonnegative lifts of
`A` and `B`, not reduced modulo `p^N`) by Kronecker packing in `x`: with the slot width `b` of
[`kronecker_bits`](@ref), `Σ_k P_k(z) 2^{b·k}` is the product of `Σ_d A_d(z) 2^{b·d}` and
`Σ_e B_e(z) 2^{b·e}` in `Z[z]` (one FLINT product with coefficients of about `2f·b` bits), and
the `P_k` are read off by bit extraction. Written into the integer polynomials `C[1:2f-1]`; `ws`
provides scratch space.
"""
function wpoly_products_kronecker!(
    R::GaloisRing,
    C::AbstractVector{ZZPolyRingElem},
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer,
    ws::WScratch,
)
    f = R.f
    m = min(len, maximum(length, A), maximum(length, B))
    b = kronecker_bits(R, m)
    Z = _zz!(ws, R, 2 * f + 2)
    pa, pb, t = Z[2 * f], Z[2 * f + 1], Z[2 * f + 2]
    pa, t = _kronecker_pack!(pa, t, A, len, b)
    if A === B
        C[1] = _zz_mullow!(C[1], pa, pa, len)
    else
        pb, t = _kronecker_pack!(pb, t, B, len, b)
        C[1] = _zz_mullow!(C[1], pa, pb, len)
    end
    # bit extraction: with Q_k = ⌊C/2^{bk}⌋, P_k = Q_k - 2^b Q_{k+1} (all coefficients are ≥ 0)
    for k in 1:(2 * f - 2)
        C[k + 1] = _zz_fdiv_2exp!(C[k + 1], C[k], b)
        t = _zz_mul_2exp!(t, C[k + 1], b)
        C[k] = sub!(C[k], C[k], t)
    end
    Z[2 * f], Z[2 * f + 1], Z[2 * f + 2] = pa, pb, t
    return C
end

# ---------------------------------------------------------------------------------------------
# Reduction modulo h̃
# ---------------------------------------------------------------------------------------------

"""
    wpoly_reduce_x!(R, P, ws) -> P

Reduce `Σ_{k<2f-1} P_k x^k` modulo `h̃` in place: afterwards `P[1:f]` holds the result
(`x^k = Σ_d xpow[k+1, d+1] x^d`, [AKN, §7.1, Step 2, p. 54]). The coordinates of `x^k` are used
as balanced representatives, so that `±1` (all coordinates for `p = 2` and a Conway polynomial)
becomes an addition or a subtraction, and other small values cost less than a scalar
multiplication by a representative in `[0, p^N)`. `ws` provides scratch space.
"""
function wpoly_reduce_x!(R::GaloisRing, P::Vector{ZZModPolyRingElem}, ws::WScratch)
    f = R.f
    f == 1 && return P
    xb = _xbal!(ws, R)
    for k in (f + 1):(2 * f - 1)
        iszero(P[k]) && continue
        for d in 1:f
            c = xb[k, d]
            if isone(c)
                P[d] = add!(P[d], P[d], P[k])
            elseif c == -1
                P[d] = sub!(P[d], P[d], P[k])
            elseif !iszero(c)
                P[d] = poly_addmul!(P[d], P[k], c)
            end
        end
    end
    return P
end

"""
    zz_reduce_x!(R, P, C, xb) -> P

Reduce `Σ_{k<2f-1} C_k x^k` for integer polynomials `C_k` modulo `h̃` and `p^N`: writes the result
into `P[1:f]` (polynomials over `Z/p^N`). The linear combinations `C_d + Σ_{k≥f} x^k_d C_k` are
formed over `Z`, with the balanced representatives `xb` of the coordinates of `x^k`
([AKN, §7.1, Step 2, p. 54]), and reduced modulo `p^N` once; `C[1:f]` is overwritten.
"""
function zz_reduce_x!(
    R::GaloisRing,
    P::Vector{ZZModPolyRingElem},
    C::AbstractVector{ZZPolyRingElem},
    xb::Matrix{ZZRingElem},
)
    f = R.f
    for d in 1:f
        acc = C[d]
        for k in (f + 1):(2 * f - 1)
            c = xb[k, d]
            (iszero(c) || iszero(C[k])) && continue
            acc = if isone(c)
                add!(acc, acc, C[k])
            elseif c == -1
                sub!(acc, acc, C[k])
            else
                addmul!(acc, C[k], c)
            end
        end
        P[d] = poly_reduce!(P[d], acc)
        C[d] = acc
    end
    return P
end

# ---------------------------------------------------------------------------------------------
# Choice of method
# ---------------------------------------------------------------------------------------------

"""
    KRONECKER_RULE

The benchmarked rule for Kronecker packing in [`wpoly_mullow`](@ref) (operands with all `f`
components nonzero): for residue degree `f` (`f > 5`: as for `f = 5`; `f = 2`: never), the entry
`(bmin, bmax, mmax, reuse)` means that Kronecker packing is used when
`bmin ≤ bits(p^N) ≤ bmax`, the operands have at most `mmax` relevant `z`-coefficients and, if
`reuse` is true, the scratch space is reused across products (the packed operands are large, so
allocating them afresh for every product costs more than Kronecker packing saves for `f ≤ 4`).
Otherwise Karatsuba in `x` (`f ≤ 4`) or schoolbook is used. Below 63 bits all coefficients are
single-word integers for FLINT, and the separate products of the other methods are much cheaper
than one product of packed multi-word integers.

Measured with `benchmark/kernels.jl` (one thread, on an 8-core machine under load from other
processes; the methods interleaved and the minimum of 10 batches taken; the scratch space reused):
time of Kronecker packing divided by the time of the best of Karatsuba and schoolbook, for full
operands of length `len`, `p = 2 / p = 3`:

| `f` | `bits(p^N)` | `len` 8 | `len` 32 | `len` 128 |
|:----|:------------|:--------|:---------|:----------|
| 2 | 32 | 2.58 / 2.45 | 2.17 / 2.18 | 1.10 / 1.07 |
| 2 | 64 | 1.09 / 0.86 | 1.18 / 1.03 | 2.36 / 1.78 |
| 2 | 128 | 1.07 / 0.92 | 1.29 / 1.16 | 1.61 / 1.36 |
| 2 | 256 | 0.76 / 0.72 | 0.87 / 0.82 | 1.01 / 0.96 |
| 2 | 512 | 0.63 / 0.62 | 1.01 / 0.93 | 1.04 / 1.00 |
| 3 | 32 | 1.86 / 1.78 | 1.62 / 1.52 | 0.95 / 0.94 |
| 3 | 64 | 0.83 / 0.80 | 1.05 / 0.99 | 1.57 / 1.34 |
| 3 | 128 | 1.02 / 0.85 | 0.81 / 0.77 | 1.39 / 1.15 |
| 3 | 256 | 0.72 / 0.65 | 0.82 / 0.71 | 0.89 / 0.78 |
| 3 | 512 | 0.77 / 0.73 | 1.32 / 1.17 | 0.92 / 0.87 |
| 4 | 32 | 1.39 / 1.53 | 1.61 / 1.50 | 1.25 / 1.16 |
| 4 | 64 | 0.81 / 0.74 | 0.75 / 0.74 | 1.19 / 0.95 |
| 4 | 128 | 0.61 / 0.55 | 0.73 / 0.68 | 1.19 / 0.94 |
| 4 | 256 | 0.71 / 0.64 | 0.77 / 0.72 | 0.80 / 0.72 |
| 4 | 512 | 0.76 / 0.71 | 1.13 / 1.03 | 0.87 / 0.81 |
| 5 | 32 | 1.04 / 0.93 | 0.99 / 1.04 | 0.70 / 0.67 |
| 5 | 64 | 0.42 / 0.39 | 0.45 / 0.41 | 0.77 / 0.62 |
| 5 | 128 | 0.37 / 0.35 | 0.45 / 0.40 | 0.77 / 0.64 |
| 5 | 256 | 0.52 / 0.45 | 0.58 / 0.52 | 0.44 / 0.41 |
| 5 | 512 | 0.46 / 0.40 | 0.64 / 0.57 | 0.46 / 0.44 |
| 6 | 32 | 0.93 / 0.94 | 0.92 / 0.88 | 0.60 / 0.60 |
| 6 | 64 | 0.37 / 0.29 | 0.37 / 0.36 | 0.62 / 0.51 |
| 6 | 128 | 0.36 / 0.32 | 0.41 / 0.37 | 0.65 / 0.53 |
| 6 | 256 | 0.45 / 0.39 | 0.49 / 0.46 | 0.38 / 0.34 |
| 6 | 512 | 0.63 / 0.54 | 0.54 / 0.49 | 0.37 / 0.35 |

With a new scratch space for every product (`p = 2`, `bits(p^N) = 74`, `f = 4`, length 32)
Kronecker packing took 71 μs against 50 μs for Karatsuba (36 μs against 39 μs when reused); at
`bits(p^N) = 49`, `f = 5`, length 16 it took 16 μs against 14 μs for schoolbook.
"""
const KRONECKER_RULE = Dict(
    3 => (96, 320, 96, true),
    4 => (63, 320, 96, true),
    5 => (63, typemax(Int), typemax(Int), false),
)

# Whether Kronecker packing is used for full operands with m relevant z-coefficients (`reuse`:
# whether the scratch space is reused across products).
function _use_kronecker(R::GaloisRing, m::Integer, reuse::Bool)
    FLINT_INPLACE || return false
    rule = get(KRONECKER_RULE, min(R.f, 5), nothing)
    rule === nothing && return false
    bmin, bmax, mmax, needs_reuse = rule
    (needs_reuse && !reuse) && return false
    return bmin <= nbits(R.pN) <= bmax && m <= mmax
end

"""
    wpoly_mullow_method(R, A, B, len; reuse=true) -> Symbol

The method used by [`wpoly_mullow`](@ref) for these operands: `:single` for `f = 1`,
`:schoolbook` when `x`-schoolbook needs at most `max(K_f, 2f - 1)` polynomial products (`K_f` the
number for Karatsuba; only the pairs of nonzero components are multiplied, so this covers
operands with coefficients in `Z_p`), `:kronecker` by the benchmarked rule
[`KRONECKER_RULE`](@ref) (`reuse`: whether the scratch space is reused across products), and
`:karatsuba` (`f ≤ 4`) or `:schoolbook` otherwise.
"""
function wpoly_mullow_method(
    R::GaloisRing,
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer;
    reuse::Bool=true,
)
    f = R.f
    f == 1 && return :single
    nA = count(!iszero, A)
    nB = A === B ? nA : count(!iszero, B)
    nprod = nA * nB
    kara = _karatsuba_products(f)
    nprod <= max(kara, 2 * f - 1) && return :schoolbook
    m = min(len, maximum(length, A), maximum(length, B))
    _use_kronecker(R, m, reuse) && return :kronecker
    return kara > 0 ? :karatsuba : :schoolbook
end

"""
    wpoly_mul!(R, P, A, B, len, ws; method=wpoly_mullow_method(R, A, B, len)) -> P

The product `A·B mod z^len` of W-polynomials with the given method, written into `P[1:f]` (`P`
has `2f - 1` entries owned by the caller; the others are overwritten). `ws` provides scratch
space.
"""
function wpoly_mul!(
    R::GaloisRing,
    P::Vector{ZZModPolyRingElem},
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer,
    ws::WScratch;
    method::Symbol=wpoly_mullow_method(R, A, B, len),
)
    if method === :single
        P[1] = poly_mullow!(P[1], A[1], B[1], len)
    elseif method === :schoolbook
        wpoly_products_schoolbook!(R, P, A, B, len, ws)
        wpoly_reduce_x!(R, P, ws)
    elseif method === :karatsuba
        wpoly_products_karatsuba!(R, P, A, B, len, ws)
        wpoly_reduce_x!(R, P, ws)
    elseif method === :kronecker
        C = view(_zz!(ws, R, 2 * R.f + 2), 1:(2 * R.f - 1))
        wpoly_products_kronecker!(R, C, A, B, len, ws)
        zz_reduce_x!(R, P, C, _xbal!(ws, R))
    else
        throw(ArgumentError("unknown multiplication method :$method"))
    end
    return P
end

# ---------------------------------------------------------------------------------------------
# Products of W-polynomials
# ---------------------------------------------------------------------------------------------

"""
    wpoly_mullow(R, A, B, len; ws=nothing, method) -> Vector{ZZModPolyRingElem}

The product of the W-polynomials `A` and `B` truncated to `z`-length `len` (i.e. modulo `z^len`),
as new polynomials. A scratch space `ws` (see `Syntomic.WScratch`) can be passed to be reused
across several products; the method defaults to `wpoly_mullow_method(R, A, B, len)`.

The `2f - 1` products `P_k = Σ_{d+e=k} A_d B_e` in `(Z/p^N)[z]` are computed by one of these
methods, then `x^k` (`k ≥ f`) is reduced modulo `h̃`:
- `:single` (`f = 1`): one FLINT `mullow`;
- `:schoolbook`: one `mullow` per pair of nonzero components (`f²` for full operands); chosen
  whenever this is no more than Karatsuba, e.g. for operands with coefficients in `Z_p`;
- `:karatsuba` (`f = 2, 3, 4`): Karatsuba in `x`, with 3, 6 and 9 `mullow`s;
- `:kronecker`: Kronecker packing in `x` into one product over `Z` with coefficients of about
  `4f·bits(p^N)` bits, followed by bit extraction; the reduction modulo `h̃` is done over `Z`.

For operands with all `f` components nonzero and `bits(p^N) ≥ 63`, Kronecker packing is used for
`f ≥ 5`, and for `f = 3, 4` in a benchmarked range of `bits(p^N)` and lengths when the scratch
space is reused (`Syntomic.KRONECKER_RULE`, whose docstring has the table of measurements);
otherwise Karatsuba (`f ≤ 4`) or schoolbook. For example, at `bits(p^N) = 128` and length 32,
Kronecker packing takes 0.73 (`f = 4`) and 0.45 (`f = 5`) of the time of the best other method
(`p = 2`), and for `f = 2` it is slower (1.29).
"""
function wpoly_mullow(
    R::GaloisRing,
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer;
    ws::Union{Nothing,WScratch}=nothing,
    method::Symbol=wpoly_mullow_method(R, A, B, len; reuse=ws !== nothing),
)
    f = R.f
    len <= 0 && return wpoly_zero(R)
    f == 1 && return [mullow(A[1], B[1], Int(len))]
    P = [zero(R.Znz) for _ in 1:(2 * f - 1)]
    wpoly_mul!(R, P, A, B, len, something(ws, WScratch(R, P)); method=method)
    return resize!(P, f)
end

"""
    wpoly_product!(R, ws, A, vA, B, vB, len) -> Union{Nothing,Vector{ZZModPolyRingElem}}

The product `(z^{vA} A)(z^{vB} B) mod z^len` in the scratch space `ws`, for W-polynomials `A`,
`B` given with their valuation shifts removed (only the `len - vA - vB` relevant coefficients are
multiplied): returns `ws.prods`, whose first `f` entries hold the product until the next use of
`ws`, or `nothing` if the product is zero for valuation reasons.
"""
function wpoly_product!(
    R::GaloisRing,
    ws::WScratch,
    A::Vector{ZZModPolyRingElem},
    vA::Integer,
    B::Vector{ZZModPolyRingElem},
    vB::Integer,
    len::Integer,
)
    s = vA + vB
    s >= len && return nothing
    P = ws.prods
    wpoly_mul!(R, P, A, B, len - s, ws)
    if s > 0
        for d in 1:(R.f)
            P[d] = poly_shift_left!(P[d], s)
        end
    end
    return P
end

"""
    wpoly_addmul!(R, out, A, vA, B, vB, len, ws) -> out

`out + (z^{vA} A)(z^{vB} B) mod z^len` (see [`wpoly_product!`](@ref)), written into the
polynomials of `out` (owned by the caller). `ws` provides scratch space.
"""
function wpoly_addmul!(
    R::GaloisRing,
    out::Vector{ZZModPolyRingElem},
    A::Vector{ZZModPolyRingElem},
    vA::Integer,
    B::Vector{ZZModPolyRingElem},
    vB::Integer,
    len::Integer,
    ws::WScratch,
)
    P = wpoly_product!(R, ws, A, vA, B, vB, len)
    P === nothing && return out
    return wpoly_add!(out, P)
end

"""
    wpoly_mullow_shifted(R, A, vA, B, vB, len) -> Vector{ZZModPolyRingElem}

The product `(z^{vA} A)(z^{vB} B)` modulo `z^len`, for W-polynomials `A`, `B` given with their
valuation shifts removed: only the `len - vA - vB` relevant coefficients are multiplied.
"""
function wpoly_mullow_shifted(
    R::GaloisRing,
    A::Vector{ZZModPolyRingElem},
    vA::Integer,
    B::Vector{ZZModPolyRingElem},
    vB::Integer,
    len::Integer,
)
    s = vA + vB
    s >= len && return wpoly_zero(R)
    P = wpoly_mullow(R, A, B, len - s)
    s == 0 && return P
    for d in eachindex(P)
        P[d] = poly_shift_left!(P[d], s)
    end
    return P
end

# ---------------------------------------------------------------------------------------------
# Scalars and σ
# ---------------------------------------------------------------------------------------------

"""
    wpoly_addmul_scalar!(R, out, A, a::WElem) -> out

`out + a·A` for a scalar `a ∈ W(F_q)/p^N`, written into the polynomials of `out` (owned by the
caller): `out_j += Σ_e M(a)_{j,e} A_e` with the multiplication matrix `M(a)` of
[`scalar_matrix`](@ref).
"""
function wpoly_addmul_scalar!(
    R::GaloisRing, out::Vector{ZZModPolyRingElem}, A::Vector{ZZModPolyRingElem}, a::WElem
)
    f = R.f
    if f == 1
        iszero(a[1]) || (out[1] = poly_addmul!(out[1], A[1], a[1]))
        return out
    end
    all(iszero, A) && return out
    M = scalar_matrix(R, a)
    for e in 1:f
        iszero(A[e]) && continue
        for j in 1:f
            c = M[j, e]
            iszero(c) || (out[j] = poly_addmul!(out[j], A[e], c))
        end
    end
    return out
end

"""
    scalar_matrix(R, a::WElem) -> Matrix{ZZRingElem}

The matrix of multiplication by `a` on `W(F_q)/p^N ≅ (Z/p^N)^f` (column `e+1` holds the coordinates
of `a·x^e`, as in `mult_matrix`), computed column by column: `a·x^e = x·(a·x^{e-1})` is a shift of
the coordinates and the reduction of `x^f` by `h̃` ([AKN, §7.1, Step 2, p. 54]), `f²` products of
integers in all.
"""
function scalar_matrix(R::GaloisRing, a::WElem)
    f = R.f
    M = Matrix{ZZRingElem}(undef, f, f)
    for d in 1:f
        M[d, 1] = mod(a[d], R.pN)
    end
    for e in 2:f
        top = M[f, e - 1]
        M[1, e] = mod(top * R.xpow[f + 1, 1], R.pN)
        for d in 2:f
            M[d, e] = mod(M[d - 1, e - 1] + top * R.xpow[f + 1, d], R.pN)
        end
    end
    return M
end

"""
    wpoly_scale(R, A, a::WElem) -> Vector{ZZModPolyRingElem}

The W-polynomial `a·A` for a scalar `a ∈ W(F_q)/p^N`, as new polynomials.
"""
wpoly_scale(R::GaloisRing, A::Vector{ZZModPolyRingElem}, a::WElem) =
    wpoly_addmul_scalar!(R, wpoly_zero(R), A, a)

"""
    wpoly_sigma(R, A) -> Vector{ZZModPolyRingElem}

Apply the Witt vector Frobenius `σ` to the coefficients of the W-polynomial `A` (no change of
the variable `z`).
"""
function wpoly_sigma(R::GaloisRing, A::Vector{ZZModPolyRingElem})
    f = R.f
    f == 1 && return copy(A)
    out = wpoly_zero(R)
    for e in 1:f
        iszero(A[e]) && continue
        for d in 1:f
            c = R.sigma[d, e]
            iszero(c) || (out[d] = poly_addmul!(out[d], A[e], c))
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Valuations and shifts
# ---------------------------------------------------------------------------------------------

"""
    ZERO_VALUATION

The z-valuation assigned to the zero W-polynomial (larger than any truncation length).
"""
const ZERO_VALUATION = typemax(Int) >> 2

"""
    wpoly_valuation(A) -> Int

The z-adic valuation of the W-polynomial `A` (the index of its first nonzero coefficient), or
[`ZERO_VALUATION`](@ref) if `A = 0`.
"""
function wpoly_valuation(A::Vector{ZZModPolyRingElem})
    v = ZERO_VALUATION
    for a in A
        iszero(a) && continue
        for k in 0:(min(length(a), v) - 1)
            if !iszero(coeff(a, k))
                v = k
                break
            end
        end
    end
    return v
end

"""
    wpoly_shift_right(A, v) -> Vector{ZZModPolyRingElem}

`A / z^v` for a W-polynomial divisible by `z^v`.
"""
wpoly_shift_right(A::Vector{ZZModPolyRingElem}, v::Integer) =
    (v == 0 || v >= ZERO_VALUATION) ? A : [shift_right(a, v) for a in A]
