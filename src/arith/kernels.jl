# Low-level kernels for polynomials in z with coefficients in W(F_q)/p^N.
#
# A "W-polynomial" is a vector of f polynomials A = (A_0, …, A_{f-1}) in (Z/p^N)[z], standing for
# Σ_d A_d(z) x^d. These kernels are shared by the power-series type `WSeries` and by the
# envelope elements; they are the innermost loops of the whole computation.
#
# A product of W-polynomials is computed in two stages: the 2f-1 products P_k = Σ_{d+e=k} A_d B_e
# in (Z/p^N)[z], or over Z for Kronecker packing (schoolbook, Karatsuba in x or Kronecker packing,
# see `wpoly_mullow`), then the reduction of x^k, k ≥ f, modulo h̃ in the presentation
# W(F_q) = Z_p[x]/(h̃) of [AKN, §7.1, Step 2, p. 54].
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
    poly_mullow!(z, x, y, n, ws=nothing) -> ZZModPolyRingElem

`x·y mod z^n` for polynomials over `Z/p^N`, written into `z` (which may alias `x` or `y`): with a
scratch space `ws` (a [`WScratch`](@ref) of the Galois ring of `x` and `y`) by the Kronecker
substitution kernel [`kron_mullow!`](@ref) in the reused buffers of `ws` when
[`kron_kernel_applies`](@ref) to the sizes, otherwise by FLINT's `fmpz_mod_poly_mullow` (the same
result). Returns `z`, or a new polynomial if in-place calls are unavailable (see
[`FLINT_INPLACE`](@ref)): callers always continue with the returned value.
"""
function poly_mullow!(
    z::ZZModPolyRingElem,
    x::ZZModPolyRingElem,
    y::ZZModPolyRingElem,
    n::Integer,
    ::Nothing=nothing,
)
    return _poly_mullow!(Val(FLINT_INPLACE), z, x, y, n)
end

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
# Kronecker substitution for products over Z/p^N with multi-limb p^N
# ---------------------------------------------------------------------------------------------

"""
    KronBuffers(pN, p, N)

The reused limb buffers of the Kronecker substitution kernel [`kron_mullow!`](@ref) for
polynomials over `Z/p^N` (`pN = p^N`): the packed operands `a` and `b` and their product `prod`
(grown as needed), a slot `tmp` of the product with the quotient `q` and the remainder `r` of its
division by `p^N`, the `K` limbs `m` of `p^N`, its bit length `nb`, the precision `N`, and whether
`p = 2`. A [`WScratch`](@ref) creates them on first use; like it, they are used by one task at a
time.
"""
struct KronBuffers
    a::Vector{UInt64}
    b::Vector{UInt64}
    prod::Vector{UInt64}
    tmp::Vector{UInt64}
    q::Vector{UInt64}
    r::Vector{UInt64}
    m::Vector{UInt64}
    K::Int
    nb::Int
    N::Int
    p2::Bool
end

function KronBuffers(pN::ZZRingElem, p::Integer, N::Integer)
    nb = nbits(pN)
    K = cld(nb, 64)
    m = zeros(UInt64, K)
    @ccall Nemo.libflint.fmpz_get_ui_array(
        m::Ptr{UInt64}, K::Int, pN::Ref{ZZRingElem}
    )::Nothing
    # a slot has at most 2K + 2 limbs for operands of fewer than 2^64 coefficients
    s = 2 * K + 2
    return KronBuffers(
        UInt64[],
        UInt64[],
        UInt64[],
        zeros(UInt64, s),
        zeros(UInt64, s),
        zeros(UInt64, s),
        m,
        K,
        nb,
        Int(N),
        p == 2,
    )
end

# The nl limbs of prod from the bit offset off on, written into tmp[1:nl] (prod must have at
# least off ÷ 64 + nl + 1 limbs).
@inline function _kron_read_slot!(
    tmp::Vector{UInt64}, prod::Vector{UInt64}, off::Int, nl::Int
)
    q = off >> 6
    s = off & 63
    if s == 0
        @inbounds for j in 1:nl
            tmp[j] = prod[q + j]
        end
    else
        @inbounds for j in 1:nl
            tmp[j] = (prod[q + j] >>> s) | (prod[q + j + 1] << (64 - s))
        end
    end
    return tmp
end

# Whether the n-limb integer a is smaller than the n-limb integer b.
@inline function _limbs_less(a::Vector{UInt64}, b::Vector{UInt64}, n::Int)
    @inbounds for j in n:-1:1
        a[j] == b[j] || return a[j] < b[j]
    end
    return false
end

"""
    KRON_LOW_RULE

The benchmarked rule for the short product of the Kronecker substitution kernel
[`kron_mullow!`](@ref). When only `nout < lx + ly - 1` coefficients of a product of operands of
`lx` and `ly` coefficients (after truncation) are needed, their slots lie in the low
`ml = ⌈nout·b/64⌉` limbs of the product of the packed operands (`b` the slot width). The kernel
then pads the packed operands with zeros and computes only these limbs (FLINT's short product,
by Mulders' algorithm in the buffer of the product, or by the basecase below 40 limbs; GMP's
`mpn_sqrlo` for a square) when `100·lx·ly ≥ percent·nout²`, i.e. the operands are about as long
as the truncation (as for two operands truncated at their common length, the common case), and
`ml ≤ maxmul` (`ml ≤ maxsqr` for a square); otherwise it computes the whole product. The result
is the same.

Measured with `benchmark/kernels.jl low` (one thread, on a machine under load from other
processes; the methods interleaved, the minimum of 3 rounds of 5 batches): time of the kernel
with the whole product divided by its time with the low half, for two operands of length `len`
truncated at `len` (`ml ≈ (2·limbs - 0.1)·len`), with `bits(p^N) ≈ 64·limbs - 8`; for `p = 3`
the ratios differ by 0.03 (median). The products gain 7–31% up to 800 limbs, and vary
irregularly above (0.75–1.25); the squares gain 2–68% up to 1280 limbs, and lose 21–38% from
about 1500 limbs on. In a separate run with shorter operands truncated at `len`, the low half of
a product was about as fast as the whole product for `lx = ly = 0.9·len` (0.9–1.1) and slower
for `lx = ly = 0.8·len` (0.7–1.0), hence the bound `percent = 90`.

Products (`p = 2`):

| limbs | 8 | 16 | 32 | 64 | 96 | 128 | 160 | 200 | 256 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 1.22 | 1.15 | 1.15 | 1.26 | 1.23 | 1.25 | 1.20 | 1.13 | 0.95 |
| 3 | 1.18 | 1.15 | 1.29 | 1.16 | 1.13 | 1.18 | 0.99 | 1.16 | 1.02 |
| 4 | 1.23 | 1.30 | 1.30 | 1.16 | 1.20 | 0.92 | 1.25 | 1.01 | 1.04 |
| 5 | 1.22 | 1.17 | 1.22 | 1.18 | 0.98 | 1.19 | 1.04 | 1.07 | 0.93 |
| 6 | 1.14 | 1.18 | 1.14 | 1.19 | 1.02 | 1.02 | 1.06 | 1.02 | 0.93 |
| 7 | 1.18 | 1.23 | 1.23 | 1.10 | 1.15 | 0.98 | 0.83 | 0.94 | 0.75 |
| 8 | 1.24 | 1.29 | 1.16 | 0.87 | 0.98 | 1.03 | 0.95 | 0.76 | 0.77 |

Squares (`p = 2`):

| limbs | 8 | 16 | 32 | 64 | 96 | 128 | 160 | 200 | 256 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 1.45 | 1.49 | 1.27 | 1.32 | 1.04 | 1.08 | 1.07 | 1.13 | 1.08 |
| 3 | 1.68 | 1.40 | 1.28 | 1.05 | 1.06 | 1.11 | 1.09 | 1.10 | 0.77 |
| 4 | 1.56 | 1.38 | 1.23 | 1.07 | 1.08 | 1.02 | 1.08 | 0.71 | 0.65 |
| 5 | 1.42 | 1.20 | 1.18 | 1.08 | 1.09 | 1.10 | 0.70 | 0.65 | 0.65 |
| 6 | 1.38 | 1.33 | 1.09 | 1.11 | 1.11 | 0.75 | 0.73 | 0.71 | 0.74 |
| 7 | 1.40 | 1.27 | 1.10 | 1.18 | 1.14 | 0.69 | 0.62 | 0.66 | 0.67 |
| 8 | 1.44 | 1.28 | 1.08 | 1.04 | 0.78 | 0.66 | 0.70 | 0.67 | 0.71 |
"""
const KRON_LOW_RULE = (percent=90, maxmul=800, maxsqr=1280)

# Whether the kernel computes only the low ml limbs of the product (the slots of the nout
# coefficients) rather than the whole product of the packed operands of lx and ly
# coefficients, by the benchmarked rule `KRON_LOW_RULE`.
function _kron_use_low(square::Bool, lx::Int, ly::Int, nout::Int, ml::Int)
    rule = KRON_LOW_RULE
    ml <= (square ? rule.maxsqr : rule.maxmul) || return false
    return 100 * lx * ly >= rule.percent * nout * nout
end

# The packed product for `_kron_mullow!` in the buffers kb: the slots of the nout = min(n,
# lx + ly - 1) coefficients of x·y (each of `bits` bits, read with `slot` limbs) in kb.prod,
# followed by zero limbs, as (bits, slot, nout); `nothing` if an operand is zero modulo z^n (see
# `_kron_mullow!` for `low`). The coefficients of a slot are exact: they are not reduced modulo p^N.
function _kron_product!(
    kb::KronBuffers, x::ZZModPolyRingElem, y::ZZModPolyRingElem, n::Int, low::Symbol=:auto
)
    lx = min(x.length, n)
    ly = min(y.length, n)
    (lx <= 0 || ly <= 0) && return nothing
    square = x === y
    # every coefficient of the product over Z is < min(lx, ly)·(p^N)² ≤ 2^(bits - 1)
    bits = 2 * kb.nb + (64 - leading_zeros(UInt64(min(lx, ly)))) + 1
    slot = cld(bits, 64)
    la = cld(lx * bits, 64)
    lb = square ? la : cld(ly * bits, 64)
    nout = min(n, lx + ly - 1)
    # the slots of the nout coefficients lie in the low ml limbs of the product; if these are
    # fewer than all (then lx, ly ≤ nout, so la, lb ≤ ml), the operands are zero-padded to ml
    # limbs and only the low ml limbs of their product are computed
    ml = cld(nout * bits, 64)
    uselow =
        nout < lx + ly - 1 &&
        (low === :always || (low === :auto && _kron_use_low(square, lx, ly, nout, ml)))
    # the limbs of the operands and of the product that are computed (for a short product that
    # is not a square, the low ml + 1 limbs of operands padded to ml + 1 limbs, with 2ml + 2
    # limbs of space for the product); the zeroed limbs above them are read with the last slots
    # (and masked off)
    na, nb = uselow ? (square ? (ml, ml) : (ml + 1, ml + 1)) : (la, lb)
    ptop = uselow ? (square ? ml : ml + 1) : la + lb
    nprod = uselow && !square ? max(ptop + slot + 2, 2 * na) : ptop + slot + 2
    a, b, prod, tmp, q = kb.a, kb.b, kb.prod, kb.tmp, kb.q
    length(a) < na && resize!(a, na)
    (!square && length(b) < nb) && resize!(b, nb)
    length(prod) < nprod && resize!(prod, nprod)
    if length(tmp) < slot
        resize!(tmp, slot)
        resize!(q, slot)
    end
    fill!(view(a, 1:na), zero(UInt64))
    square || fill!(view(b, 1:nb), zero(UInt64))
    GC.@preserve x y a b prod begin
        pa, pb, pp = pointer(a), pointer(b), pointer(prod)
        @ccall Nemo.libflint._fmpz_poly_bit_pack(
            pa::Ptr{UInt64}, x.coeffs::Ptr{Nothing}, lx::Int, UInt(bits)::UInt, 0::Cint
        )::Nothing
        if !square
            @ccall Nemo.libflint._fmpz_poly_bit_pack(
                pb::Ptr{UInt64}, y.coeffs::Ptr{Nothing}, ly::Int, UInt(bits)::UInt, 0::Cint
            )::Nothing
        end
        if uselow && square
            @ccall Base.GMP.libgmp.__gmpn_sqrlo(
                pp::Ptr{UInt64}, pa::Ptr{UInt64}, ml::Int
            )::Nothing
        elseif uselow && ml < 40
            # the low ml limbs by the basecase, and the next limb as the return value
            @inbounds prod[ml + 1] = @ccall Nemo.libflint._flint_mpn_mullow_n(
                pp::Ptr{UInt64}, pa::Ptr{UInt64}, pb::Ptr{UInt64}, ml::Int
            )::UInt64
        elseif uselow
            # the low ml + 1 limbs by Mulders' short product, which uses the 2ml + 2 limbs of
            # the product buffer as scratch space (the entry point `_flint_mpn_mullow_n` would
            # allocate this space for every product)
            @ccall Nemo.libflint._flint_mpn_mullow_n_mulders_recursive(
                pp::Ptr{UInt64}, pa::Ptr{UInt64}, pb::Ptr{UInt64}, (ml + 1)::Int
            )::Nothing
        elseif square
            @ccall Nemo.libflint.flint_mpn_sqr(
                pp::Ptr{UInt64}, pa::Ptr{UInt64}, la::Int
            )::UInt64
        elseif la >= lb
            # flint_mpn_mul needs the longer operand first
            @ccall Nemo.libflint.flint_mpn_mul(
                pp::Ptr{UInt64}, pa::Ptr{UInt64}, la::Int, pb::Ptr{UInt64}, lb::Int
            )::UInt64
        else
            @ccall Nemo.libflint.flint_mpn_mul(
                pp::Ptr{UInt64}, pb::Ptr{UInt64}, lb::Int, pa::Ptr{UInt64}, la::Int
            )::UInt64
        end
        @inbounds for j in (ptop + 1):nprod
            prod[j] = zero(UInt64)
        end
    end
    return bits, slot, nout
end

# The kernel of `kron_mullow!` in the buffers kb (see its docstring); x and y are polynomials
# over Z/p^N for the modulus of kb, with coefficients in [0, p^N). The low-half product is used
# when the rule `_kron_use_low` chooses it (`low = :auto`), always when possible (`:always`) or
# never (`:never`); the result is the same.
function _kron_mullow!(
    kb::KronBuffers,
    z::ZZModPolyRingElem,
    x::ZZModPolyRingElem,
    y::ZZModPolyRingElem,
    n::Int,
    low::Symbol=:auto,
)
    # both operands are packed before z is resized, so z may alias x or y
    shape = _kron_product!(kb, x, y, n, low)
    shape === nothing && return zero!(z)
    bits, slot, nout = shape
    K = kb.K
    ctx = _fmpz_mod_ctx(x)
    prod, tmp, q, r, m = kb.prod, kb.tmp, kb.q, kb.r, kb.m
    GC.@preserve z prod tmp q r m begin
        @ccall Nemo.libflint.fmpz_mod_poly_fit_length(
            z::Ref{ZZModPolyRingElem}, nout::Int, ctx::Ref{Nemo.fmpz_mod_ctx_struct}
        )::Nothing
        # the coefficients of z may have moved
        zc = z.coeffs
        ptmp, pq, pr, pm = pointer(tmp), pointer(q), pointer(r), pointer(m)
        if kb.p2
            # the residue modulo 2^N is given by the low N bits of the slot
            kn = cld(kb.N, 64)
            rem = kb.N & 63
            mask = rem == 0 ? typemax(UInt64) : (UInt64(1) << rem) - one(UInt64)
            for k in 0:(nout - 1)
                _kron_read_slot!(tmp, prod, k * bits, kn)
                @inbounds tmp[kn] &= mask
                @ccall Nemo.libflint.fmpz_set_ui_array(
                    (zc + 8 * k)::Ptr{Nothing}, ptmp::Ptr{UInt64}, kn::Int
                )::Nothing
            end
        else
            top = bits & 63
            topmask = top == 0 ? typemax(UInt64) : (UInt64(1) << top) - one(UInt64)
            for k in 0:(nout - 1)
                _kron_read_slot!(tmp, prod, k * bits, slot)
                # the bits above the slot belong to the next coefficient
                @inbounds tmp[slot] &= topmask
                nl = slot
                while nl > K && @inbounds(tmp[nl]) == 0
                    nl -= 1
                end
                if nl == K && _limbs_less(tmp, m, K)
                    @ccall Nemo.libflint.fmpz_set_ui_array(
                        (zc + 8 * k)::Ptr{Nothing}, ptmp::Ptr{UInt64}, K::Int
                    )::Nothing
                else
                    @ccall Base.GMP.libgmp.__gmpn_tdiv_qr(
                        pq::Ptr{UInt64},
                        pr::Ptr{UInt64},
                        0::Int,
                        ptmp::Ptr{UInt64},
                        nl::Int,
                        pm::Ptr{UInt64},
                        K::Int,
                    )::Nothing
                    @ccall Nemo.libflint.fmpz_set_ui_array(
                        (zc + 8 * k)::Ptr{Nothing}, pr::Ptr{UInt64}, K::Int
                    )::Nothing
                end
            end
        end
        @ccall Nemo.libflint._fmpz_mod_poly_set_length(
            z::Ref{ZZModPolyRingElem}, nout::Int
        )::Nothing
        @ccall Nemo.libflint._fmpz_mod_poly_normalise(z::Ref{ZZModPolyRingElem})::Nothing
    end
    return z
end

# The functions of libflint and of libgmp called by the Kronecker substitution kernel (and by
# its self-test).
const _KRON_FLINT_SYMBOLS = (
    :_fmpz_poly_bit_pack,
    :flint_mpn_mul,
    :flint_mpn_sqr,
    :fmpz_set_ui_array,
    :fmpz_get_ui_array,
    :fmpz_mod_poly_fit_length,
    :_fmpz_mod_poly_set_length,
    :_fmpz_mod_poly_normalise,
    :fmpz_mod_poly_mullow,
    :_flint_mpn_mullow_n,
    :_flint_mpn_mullow_n_mulders_recursive,
)
const _KRON_GMP_SYMBOLS = (:__gmpn_tdiv_qr, :__gmpn_sqrlo)

function _has_symbols(lib, symbols)
    handle = Base.Libc.Libdl.dlopen(lib)
    return all(s -> Base.Libc.Libdl.dlsym_e(handle, s) != C_NULL, symbols)
end

# Products of pseudo-random polynomials over Z/p^N by the kernel against FLINT's
# fmpz_mod_poly_mullow: p = 2, 3, 5 with 1 to 6 limbs (for p = 2 also N a multiple of 64),
# lengths 0 to 70, truncations shorter and longer than the product, squares, operands with all
# coefficients p^N - 1, reused outputs of other lengths and outputs aliasing an operand, each
# with the whole product and with its low half.
function _kron_self_test()
    state = Ref(0x2545f4914f6cdd1d)
    function next()
        # splitmix64
        state[] += 0x9e3779b97f4a7c15
        v = state[]
        v = (v ⊻ (v >> 30)) * 0xbf58476d1ce4e5b9
        v = (v ⊻ (v >> 27)) * 0x94d049bb133111eb
        return v ⊻ (v >> 31)
    end
    cases = (
        (2, 40),
        (2, 64),
        (2, 100),
        (2, 128),
        (2, 191),
        (2, 192),
        (2, 300),
        (2, 320),
        (2, 383),
        (3, 39),
        (3, 50),
        (3, 81),
        (3, 150),
        (3, 200),
        (3, 242),
        (5, 27),
        (5, 60),
        (5, 100),
        (5, 150),
        (5, 165),
    )
    for (p, N) in cases
        pN = ZZ(p)^N
        Zn, _ = residue_ring(ZZ, pN; cached=false)
        Znz, _ = polynomial_ring(Zn, :z; cached=false)
        kb = KronBuffers(pN, p, N)
        # residues of K + 1 random limbs, a quarter of them shifted down to a random size
        function rand_coeff()
            c = ZZ(0)
            for _ in 0:(kb.K)
                c = (c << 64) + ZZ(next())
            end
            c = mod(c, pN)
            return next() % 4 == 0 ? c >> Int(next() % kb.nb) : c
        end
        function rand_poly(len, largest)
            largest && return Znz([Zn(pN - 1) for _ in 1:len])
            return Znz([Zn(rand_coeff()) for _ in 1:len])
        end
        z = zero(Znz)
        for trial in 1:6
            lx, ly = Int(next() % 71), Int(next() % 71)
            x = rand_poly(lx, trial == 1)
            y = trial == 2 ? x : rand_poly(ly, trial == 1)
            for n in (lx + ly + 1, lx + ly - 2, max(lx, ly), (lx + ly) ÷ 3, 1, 0)
                n < 0 && continue
                expected = _poly_mullow!(Val(true), zero(Znz), x, y, n)
                # the whole product and its low half
                for low in (:never, :always)
                    _kron_mullow!(kb, z, x, y, n, low) == expected || return false
                end
                x2 = deepcopy(x)
                _kron_mullow!(kb, x2, x2, y, n, :always) == expected || return false
                y2 = deepcopy(y)
                _kron_mullow!(kb, y2, x, y2, n, :never) == expected || return false
            end
        end
    end
    return true
end

"""
    kron_kernel_available() -> Bool

Whether the Kronecker substitution kernel [`kron_mullow!`](@ref) can be used: the in-place FLINT
calls are available ([`FLINT_INPLACE`](@ref)), limbs have 64 bits, Nemo's polynomials over
`Z/p^N` have FLINT's layout (fields `coeffs`, `alloc`, `length`), libflint and libgmp export the
functions the kernel calls, and a self-test against FLINT's `fmpz_mod_poly_mullow` passes
(products for `p = 2, 3, 5` with 1 to 6 limbs, squares, truncations, aliasing and reused
outputs). Otherwise every product goes through FLINT, with the same results.
"""
function kron_kernel_available()
    FLINT_INPLACE || return false
    (Sys.WORD_SIZE == 64 && Base.GMP.BITS_PER_LIMB == 64) || return false
    fieldcount(ZZModPolyRingElem) >= 3 || return false
    fieldnames(ZZModPolyRingElem)[1:3] == (:coeffs, :alloc, :length) || return false
    fieldtypes(ZZModPolyRingElem)[1:3] == (Ptr{Nothing}, Int, Int) || return false
    try
        _has_symbols(Nemo.libflint, _KRON_FLINT_SYMBOLS) || return false
        _has_symbols(Base.GMP.libgmp, _KRON_GMP_SYMBOLS) || return false
        return _kron_self_test()
    catch
        return false
    end
end

"""
    KRONECKER_KERNEL

Whether products over `Z/p^N` in a scratch space use the Kronecker substitution kernel
[`kron_mullow!`](@ref) for the sizes of [`KRON_KERNEL_RULE`](@ref) (see
`Syntomic.kron_kernel_available`); fixed when the package is compiled.
"""
const KRONECKER_KERNEL = kron_kernel_available()

"""
    KRON_KERNEL_RULE

The benchmarked rule for the Kronecker substitution kernel [`kron_mullow!`](@ref): for `p^N` of
`K ≤ length(KRON_KERNEL_RULE)` limbs, a product in a scratch space uses the kernel when the
shorter operand (after truncation) has a length in the range `KRON_KERNEL_RULE[K]`, and FLINT's
`fmpz_mod_poly_mullow` otherwise:
- one limb: never (FLINT multiplies single-word coefficients without packing them);
- 2 to 8 limbs: from 7 coefficients on (FLINT multiplies shorter operands coefficient by
  coefficient);
- 2 and 3 limbs: up to 99 coefficients (from 100 on, FLINT uses its small-prime FFT);
- 4 to 6 limbs: up to 256 coefficients, the largest length measured (for 5 and 6 limbs and
  `p = 3` the ratio is only 0.98–1.10 above 64 coefficients, but mostly above 1);
- 7 and 8 limbs: up to 20 and 14 coefficients (from about 24 and 16 to 64 coefficients,
  FLINT's Schönhage–Strassen multiplication is about as fast or faster, and above 64 the kernel
  gains at most 10%);
- more than 8 limbs: never (not measured).

Measured with `benchmark/kernels.jl kernel` (one thread, on a machine under load from other
processes; the two methods interleaved, the minimum of 3 rounds of 5 batches, the scratch space
and the output reused): time of FLINT divided by the time of the kernel, for two operands of
length `len` truncated at `len`, with `bits(p^N) ≈ 64·limbs - 8`. From 7 coefficients on, the
kernel computes only the low half of these products (see [`KRON_LOW_RULE`](@ref)); with the
whole product, the ratios for 2 to 6 limbs and 7 to 96 coefficients are 0.2 lower (median;
at most 0.5), and 0.3 for squares (at most 1.8). Squares give similar ratios, and so do
operands of unequal lengths, by the length of the shorter one. The bounds for 7 and 8 limbs
are from a finer run (lengths 10 to 32, 5 rounds): at 20 and 14 coefficients the ratios are
1.08–1.47, at 24 and 16 coefficients the squares for `p = 3` 1.02.

`p = 2`:

| limbs | 4 | 6 | 7 | 8 | 16 | 32 | 64 | 96 | 100 | 128 | 256 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.41 | 1.20 | 1.11 | 1.17 | 0.93 | 0.80 | 0.88 | 0.91 | 0.95 | 0.98 | 0.72 |
| 2 | 1.33 | 1.30 | 2.50 | 2.65 | 1.89 | 1.66 | 1.69 | 1.53 | 0.88 | 0.69 | 0.48 |
| 3 | 0.99 | 0.99 | 2.32 | 2.21 | 1.79 | 1.66 | 1.47 | 1.31 | 0.94 | 0.77 | 0.67 |
| 4 | 0.84 | 0.87 | 1.98 | 1.99 | 1.72 | 1.57 | 1.33 | 1.37 | 1.35 | 1.19 | 1.06 |
| 5 | 0.72 | 0.73 | 1.97 | 1.78 | 1.46 | 1.47 | 1.35 | 1.04 | 1.07 | 1.07 | 1.12 |
| 6 | 0.67 | 0.68 | 1.59 | 1.62 | 1.53 | 1.38 | 1.31 | 1.09 | 1.08 | 1.08 | 1.09 |
| 7 | 0.61 | 0.59 | 1.62 | 1.93 | 1.50 | 1.08 | 0.68 | 1.07 | 1.10 | 1.07 | 1.05 |
| 8 | 0.59 | 0.58 | 1.58 | 1.66 | 1.23 | 0.87 | 0.74 | 1.08 | 1.06 | 1.06 | 1.05 |

`p = 3`:

| limbs | 4 | 6 | 7 | 8 | 16 | 32 | 64 | 96 | 100 | 128 | 256 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.98 | 0.85 | 0.82 | 0.85 | 0.73 | 0.67 | 0.70 | 0.79 | 0.80 | 0.87 | 0.67 |
| 2 | 0.81 | 0.99 | 1.91 | 1.82 | 1.65 | 1.42 | 1.56 | 1.41 | 0.84 | 0.68 | 0.46 |
| 3 | 0.86 | 0.89 | 1.68 | 1.70 | 1.48 | 1.40 | 1.30 | 1.20 | 0.90 | 0.74 | 0.71 |
| 4 | 0.78 | 0.79 | 1.59 | 1.59 | 1.52 | 1.45 | 1.24 | 1.29 | 1.27 | 1.21 | 1.08 |
| 5 | 0.72 | 0.70 | 1.47 | 1.44 | 1.41 | 1.28 | 1.20 | 1.08 | 0.98 | 1.09 | 1.06 |
| 6 | 0.69 | 0.67 | 1.33 | 1.35 | 1.33 | 1.23 | 1.21 | 1.10 | 1.00 | 1.00 | 1.05 |
| 7 | 0.69 | 0.63 | 1.33 | 1.59 | 1.29 | 0.99 | 0.68 | 1.04 | 0.99 | 1.09 | 1.09 |
| 8 | 0.62 | 0.62 | 1.36 | 1.39 | 1.16 | 0.87 | 0.76 | 1.03 | 1.01 | 1.04 | 1.01 |
"""
const KRON_KERNEL_RULE = (1:0, 7:99, 7:99, 7:256, 7:256, 7:256, 7:20, 7:14)

"""
    kron_kernel_applies(R, lx, ly) -> Bool

Whether a product over the coefficient ring `Z/p^N` of the Galois ring `R` of operands with `lx`
and `ly` coefficients (after truncation) uses the Kronecker substitution kernel
[`kron_mullow!`](@ref) (when a scratch space is given): the kernel is available
([`KRONECKER_KERNEL`](@ref)) and the sizes lie in the benchmarked range of
[`KRON_KERNEL_RULE`](@ref).
"""
function kron_kernel_applies(R::GaloisRing, lx::Integer, ly::Integer)
    KRONECKER_KERNEL || return false
    K = cld(nbits(R.pN), 64)
    return K <= length(KRON_KERNEL_RULE) && min(lx, ly) in KRON_KERNEL_RULE[K]
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
    WAcc(R, len)

A sum of W-polynomials over `W(F_q)/p^N` modulo `z^len`, built with [`acc_addmul!`](@ref) and
[`acc_add!`](@ref) and read with [`acc_finish!`](@ref). For `f = 1` and a modulus `p^N` of `K ≥ 2`
limbs (when the Kronecker substitution kernel is available, [`KRONECKER_KERNEL`](@ref)) the
coefficients are kept as `S` limbs each: for `p = 2` modulo `2^N` (`S = ⌈N/64⌉`), for odd `p` as
exact sums of products (`S = 2K + 2` limbs hold the sum of more than `2^63` products of two
residues), reduced modulo `p^N` once by `acc_finish!`. Otherwise (`S = 0`) the sum is kept as `f`
polynomials over `Z/p^N`. A `WAcc` is used by one task at a time.
"""
mutable struct WAcc
    R::GaloisRing
    len::Int
    S::Int
    data::Vector{UInt64}
    polys::Vector{ZZModPolyRingElem}
    used::Int
    tmp::Vector{UInt64}
    m::Vector{UInt64}
    q::Vector{UInt64}
    r::Vector{UInt64}
end

# The number of limbs of p^N.
_modulus_limbs(R::GaloisRing) = cld(nbits(R.pN), 64)

# The limbs per coefficient of an accumulator of R (0: polynomials).
function _acc_limbs(R::GaloisRing)
    (KRONECKER_KERNEL && R.f == 1) || return 0
    K = _modulus_limbs(R)
    K >= 2 || return 0
    return R.p == 2 ? cld(R.N, 64) : 2 * K + 2
end

function WAcc(R::GaloisRing, len::Integer)
    S = _acc_limbs(R)
    n = max(Int(len), 0)
    data = zeros(UInt64, S * n)
    polys = S == 0 ? wpoly_zero(R) : ZZModPolyRingElem[]
    K = _modulus_limbs(R)
    m = zeros(UInt64, K)
    if S > 0
        @ccall Nemo.libflint.fmpz_get_ui_array(
            m::Ptr{UInt64}, K::Int, R.pN::Ref{ZZRingElem}
        )::Nothing
    end
    return WAcc(
        R,
        n,
        S,
        data,
        polys,
        0,
        zeros(UInt64, 2 * K + 4),
        m,
        zeros(UInt64, S),
        zeros(UInt64, K),
    )
end

# Whether the K limbs of data from index i on are smaller than the K limbs of m.
@inline function _acc_less(data::Vector{UInt64}, i::Int, m::Vector{UInt64}, K::Int)
    @inbounds for j in K:-1:1
        data[i + j - 1] == m[j] || return data[i + j - 1] < m[j]
    end
    return false
end

"""
    WScratch(R)
    WScratch(R, P)

Scratch space for the in-place W-polynomial kernels: the `2f - 1` polynomials `prods` over `Z/p^N`
for the products in `x` (`P`, if given), polynomials over `Z/p^N` and over `Z` for the operands
and partial products of Karatsuba and Kronecker packing, the limb buffers
([`KronBuffers`](@ref)) of the Kronecker substitution kernel [`kron_mullow!`](@ref) and an
accumulator ([`WAcc`](@ref)), created on first use. A `WScratch` belongs to the Galois ring `R` it was created for (the kernels throw an
`ArgumentError` for another ring) and is used by one task at a time: the function that needs it
allocates it, or takes it from a [`ScratchPool`](@ref), and never shares it.
"""
struct WScratch
    R::GaloisRing
    prods::Vector{ZZModPolyRingElem}
    tmp::Vector{ZZModPolyRingElem}
    zz::Vector{ZZPolyRingElem}
    xbal::Vector{Tuple{Matrix{ZZRingElem},Bool}}
    kron::Vector{KronBuffers}
    acc::Vector{WAcc}
end

function WScratch(R::GaloisRing, P::Vector{ZZModPolyRingElem})
    return WScratch(
        R,
        P,
        ZZModPolyRingElem[],
        ZZPolyRingElem[],
        Tuple{Matrix{ZZRingElem},Bool}[],
        KronBuffers[],
        WAcc[],
    )
end

# The scratch space `ws` must belong to `R` (its cached data and scratch polynomials refer to it).
function _check_scratch(ws::WScratch, R::GaloisRing)
    ws.R === R || throw(ArgumentError("the scratch space belongs to another Galois ring"))
    return nothing
end

WScratch(R::GaloisRing) = WScratch(R, [zero(R.Znz) for _ in 1:(2 * R.f - 1)])

"""
    ScratchPool(R)

Scratch spaces ([`WScratch`](@ref)) of the Galois ring `R` for the tasks of parallel loops: a task
takes one with [`with_scratch`](@ref) and gives it back when it is done, so that their buffers (in
particular the limb buffers of the Kronecker substitution kernel) are reused by later loops
instead of being allocated for every piece of work. At most as many scratch spaces are created as
tasks use them at the same time.
"""
struct ScratchPool
    R::GaloisRing
    lock::ReentrantLock
    free::Vector{WScratch}
end

ScratchPool(R::GaloisRing) = ScratchPool(R, ReentrantLock(), WScratch[])

"""
    with_scratch(f, pool::ScratchPool)

`f(ws)` for a scratch space `ws` of the pool, which the calling task has to itself until `f`
returns (then it goes back to the pool).
"""
function with_scratch(f, pool::ScratchPool)
    ws = lock(pool.lock) do
        isempty(pool.free) ? nothing : pop!(pool.free)
    end
    ws === nothing && (ws = WScratch(pool.R))
    try
        return f(ws)
    finally
        lock(() -> push!(pool.free, ws), pool.lock)
    end
end

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

# The limb buffers of the Kronecker substitution kernel.
function _kron!(ws::WScratch, R::GaloisRing)
    isempty(ws.kron) && push!(ws.kron, KronBuffers(R.pN, R.p, R.N))
    return ws.kron[1]
end

"""
    scratch_acc!(ws, len) -> WAcc

The accumulator of the scratch space `ws`, set to zero modulo `z^len` (see [`WAcc`](@ref));
it is valid until the next call for `ws`.
"""
function scratch_acc!(ws::WScratch, len::Integer)
    isempty(ws.acc) && push!(ws.acc, WAcc(ws.R, len))
    return acc_reset!(ws.acc[1], len)
end

"""
    kron_mullow!(z, x, y, n, ws) -> ZZModPolyRingElem

`x·y mod z^n` for polynomials over `Z/p^N`, the coefficient ring of the Galois ring of the scratch
space `ws`, written into `z` (which may alias `x` or `y`) by Kronecker substitution in reused limb
buffers of `ws`. With `lx`, `ly` the lengths of the operands after truncation, their coefficients
(in `[0, p^N)`) are packed into slots of `b = 2·bits(p^N) + bits(min(lx, ly)) + 1` bits of two
integers; every coefficient of the product over `Z` is smaller than `min(lx, ly)·(p^N)²`, hence
than `2^{b-1}`, so one product of these integers (a squaring if `x === y`) holds the coefficients
of `x·y` in its slots. When fewer than `lx + ly - 1` coefficients are needed and the operands
are about as long as the truncation, only the low limbs of this product that hold their slots
are computed (a short product, by the rule `KRON_LOW_RULE`). Each coefficient is read off its
slot and reduced modulo `p^N` (its low `N` bits for `p = 2`, a division otherwise) directly into
`z`. The result is that of FLINT's `fmpz_mod_poly_mullow`, which allocates the packed integers
and the unpacked coefficients anew for every product.

Throws an `ArgumentError` if `x` or `y` is not a polynomial over `Z/p^N`; uses FLINT if the
kernel is unavailable (`KRONECKER_KERNEL`). Products in a scratch space use the kernel for the
benchmarked sizes of `KRON_KERNEL_RULE`.
"""
function kron_mullow!(
    z::ZZModPolyRingElem,
    x::ZZModPolyRingElem,
    y::ZZModPolyRingElem,
    n::Integer,
    ws::WScratch,
)
    KRONECKER_KERNEL || return poly_mullow!(z, x, y, n)
    R = ws.R
    (x.parent.base_ring === R.Zn && y.parent.base_ring === R.Zn) || throw(
        ArgumentError(
            "kron_mullow!: the operands are not polynomials over Z/$(R.p)^$(R.N)"
        ),
    )
    return _kron_mullow!(_kron!(ws, R), z, x, y, Int(n))
end

function poly_mullow!(
    z::ZZModPolyRingElem,
    x::ZZModPolyRingElem,
    y::ZZModPolyRingElem,
    n::Integer,
    ws::WScratch,
)
    if kron_kernel_applies(ws.R, min(length(x), n), min(length(y), n))
        return kron_mullow!(z, x, y, n, ws)
    end
    return poly_mullow!(z, x, y, n)
end

# The balanced representatives in (-p^N/2, p^N/2] of the coordinates xpow of x^k, and whether
# those of x^f, …, x^{2f-2} all lie in {-1, 0, 1}, so that the reduction modulo h̃ needs only
# additions and subtractions (e.g. for p = 2 and a Conway polynomial); computed once per scratch
# space.
function _xbal_data!(ws::WScratch, R::GaloisRing)
    if isempty(ws.xbal)
        half = R.pN >> 1
        xb = map(c -> c > half ? c - R.pN : c, R.xpow)
        high = view(xb, (R.f + 1):(2 * R.f - 1), :)
        push!(ws.xbal, (xb, all(c -> iszero(c) || isone(c) || c == -1, high)))
    end
    return ws.xbal[1]
end

_xbal!(ws::WScratch, R::GaloisRing) = _xbal_data!(ws, R)[1]
_unit_reduction!(ws::WScratch, R::GaloisRing) = _xbal_data!(ws, R)[2]

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
                P[k] = poly_mullow!(P[k], A[d], B[e], len, ws)
            else
                T[1] = poly_mullow!(T[1], A[d], B[e], len, ws)
                P[k] = add!(P[k], P[k], T[1])
            end
        end
    end
    return P
end

# Karatsuba for a product of two linear polynomials in x: (a0 + a1 x)(b0 + b1 x) mod z^len into
# (c0, c1, c2), with 3 polynomial products in the scratch space ws; sa and sb are scratch
# polynomials (returned, as they may be replaced when in-place calls are unavailable).
function _karatsuba_linear!(c0, c1, c2, a0, a1, b0, b1, len, sa, sb, ws)
    c0 = poly_mullow!(c0, a0, b0, len, ws)
    c2 = poly_mullow!(c2, a1, b1, len, ws)
    sa = poly_add_trunc!(sa, a0, a1, len)
    if a0 === b0 && a1 === b1
        c1 = poly_mullow!(c1, sa, sa, len, ws)
    else
        sb = poly_add_trunc!(sb, b0, b1, len)
        c1 = poly_mullow!(c1, sa, sb, len, ws)
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
            P[1], P[2], P[3], A[1], A[2], B[1], B[2], len, T[1], T[2], ws
        )
    elseif f == 3
        _karatsuba_quadratic!(P, A, B, len, _tmp!(ws, R, 3), ws)
    elseif f == 4
        _karatsuba_cubic!(P, A, B, len, _tmp!(ws, R, 8), ws)
    else
        throw(ArgumentError("wpoly_products_karatsuba!: not implemented for f = $f"))
    end
    return P
end

# f = 3: with m_d = a_d b_d and m_{de} = (a_d + a_e)(b_d + b_e),
#   P_0 = m_0, P_1 = m_01 - m_0 - m_1, P_2 = m_02 - m_0 - m_2 + m_1, P_3 = m_12 - m_1 - m_2, P_4 = m_2.
function _karatsuba_quadratic!(P, A, B, len, T, ws)
    P[1] = poly_mullow!(P[1], A[1], B[1], len, ws)          # m_0
    P[3] = poly_mullow!(P[3], A[2], B[2], len, ws)          # m_1
    P[5] = poly_mullow!(P[5], A[3], B[3], len, ws)          # m_2
    P[2] = _karatsuba_pair!(P[2], A, B, 1, 2, len, T, ws)   # m_01
    P[2] = sub!(P[2], P[2], P[1])
    P[2] = sub!(P[2], P[2], P[3])
    P[4] = _karatsuba_pair!(P[4], A, B, 2, 3, len, T, ws)   # m_12
    P[4] = sub!(P[4], P[4], P[3])
    P[4] = sub!(P[4], P[4], P[5])
    T[3] = _karatsuba_pair!(T[3], A, B, 1, 3, len, T, ws)   # m_02
    T[3] = sub!(T[3], T[3], P[1])
    T[3] = sub!(T[3], T[3], P[5])
    P[3] = add!(P[3], P[3], T[3])
    return P
end

# (A_d + A_e)(B_d + B_e) mod z^len into c, using T[1] and T[2] as scratch and the scratch space
# ws for the product.
function _karatsuba_pair!(c, A, B, d, e, len, T, ws)
    T[1] = poly_add_trunc!(T[1], A[d], A[e], len)
    A === B && return poly_mullow!(c, T[1], T[1], len, ws)
    T[2] = poly_add_trunc!(T[2], B[d], B[e], len)
    return poly_mullow!(c, T[1], T[2], len, ws)
end

# f = 4: A = A' + A'' y with y = x², A' = a_0 + a_1 x, A'' = a_2 + a_3 x (likewise B); then
#   AB = A'B' + ((A' + A'')(B' + B'') - A'B' - A''B'') y + A''B'' y²,
# each of the three products of linear polynomials by Karatsuba (3 products).
function _karatsuba_cubic!(P, A, B, len, T, ws)
    square = A === B
    P[1], P[2], P[3], T[1], T[2] = _karatsuba_linear!(
        P[1], P[2], P[3], A[1], A[2], B[1], B[2], len, T[1], T[2], ws
    )
    P[5], P[6], P[7], T[1], T[2] = _karatsuba_linear!(
        P[5], P[6], P[7], A[3], A[4], B[3], B[4], len, T[1], T[2], ws
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
        T[7], P[4], T[8], T[3], T[4], sb0, sb1, len, T[1], T[2], ws
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
# The largest length of the first `f` components of a W-polynomial (the only ones used).
_max_length(A::AbstractVector, f::Integer) = maximum(d -> length(A[d]), 1:f)

function _kronecker_pack!(zz, t, A, f, len, b)
    zz = poly_lift_trunc!(zz, A[f], len)
    for d in (f - 1):-1:1
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
    m = min(len, _max_length(A, f), _max_length(B, f))
    b = kronecker_bits(R, m)
    Z = _zz!(ws, R, 2 * f + 2)
    pa, pb, t = Z[2 * f], Z[2 * f + 1], Z[2 * f + 2]
    pa, t = _kronecker_pack!(pa, t, A, f, len, b)
    if A === B
        C[1] = _zz_mullow!(C[1], pa, pa, len)
    else
        pb, t = _kronecker_pack!(pb, t, B, f, len, b)
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
    _check_scratch(ws, R)
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

The benchmarked rule for Kronecker packing in [`wpoly_mullow`](@ref) (operands with more nonzero
components than schoolbook handles, see [`wpoly_mullow_method`](@ref)). For residue degree `f`
(`f > 5`: as for `f = 5`; `f = 2, 3`: never) the entry means that Kronecker packing is used when
`bmin ≤ bits(p^N) ≤ bmax`, the operands have at most `mmax` relevant `z`-coefficients and, if
`reuse` is set, the scratch space is reused across products (the packed operands are large, so
allocating them for every product costs more than Kronecker packing saves for `f ≤ 4`). The
bound `bmin` is replaced by `bmin_unit` when the reduction modulo `h̃` needs only additions and
subtractions (the balanced coordinates of `x^f, …, x^{2f-2}` lie in `{-1, 0, 1}`, e.g. for
`p = 2` and a Conway polynomial), which makes Karatsuba cheaper. Otherwise Karatsuba in `x`
(`f ≤ 4`) or schoolbook is used. Below 63 bits the coefficients are single-word integers for
FLINT, and the separate products of the other methods are much cheaper than one product of
packed multi-word integers.

The rule is set by the products of actual computations (operands of different lengths, most
shorter than the truncation length), where the gains are smaller than for the full operands of
the table below. Time spent in the products of `W(F_q)`-polynomials, Kronecker packing against
the best other method, for `syntomic_cohomology` over ramified quadratic extensions of `Q_q`
(one thread, the same machine):
- `f = 3`, `p = 3`, `bits(p^N) = 126`: 2.91–3.02 s against 2.81–2.84 s (slower; never used);
- `f = 4`, `p = 2`, `bits(p^N) = 123`: 12.6 s against 12.8 s (not significant; not used for unit
  reductions);
- `f = 4`, `p = 3`, `bits(p^N) = 88`: 0.97–1.04 s against 1.07–1.11 s;
- `f = 5`, `p = 2`, `bits(p^N) = 74`: 1.78 s against 3.24 s (schoolbook).

Measured with `benchmark/kernels.jl` (one thread, on an 8-core machine under load from other
processes; the methods interleaved, the minimum of 10 batches, the scratch space reused): time of
Kronecker packing divided by the time of the best of Karatsuba and schoolbook, for full operands
of length `len`, `p = 2 / p = 3` (`p = 5` is within a few percent of `p = 3`; `bits` is the
target size, `bits(p^N)` is at most 2 larger):

| `f` | `bits(p^N)` | `len` 8 | `len` 16 | `len` 32 | `len` 64 |
|:----|:------------|:--------|:---------|:---------|:---------|
| 2 | 48 | 2.82 / 2.61 | 2.74 / 2.40 | 2.61 / 2.28 | 2.53 / 2.34 |
| 2 | 64 | 1.05 / 0.92 | 1.15 / 0.97 | 1.21 / 1.03 | 1.24 / 1.10 |
| 2 | 96 | 1.24 / 1.05 | 1.38 / 1.17 | 1.51 / 1.32 | 1.53 / 1.30 |
| 2 | 128 | 1.05 / 1.00 | 1.20 / 1.06 | 1.30 / 1.20 | 1.35 / 1.18 |
| 2 | 192 | 1.35 / 1.17 | 1.25 / 1.08 | 1.08 / 0.98 | 0.87 / 0.85 |
| 2 | 256 | 0.77 / 0.73 | 1.09 / 0.97 | 0.87 / 0.82 | 0.77 / 0.73 |
| 2 | 384 | 0.81 / 0.76 | 1.09 / 1.00 | 0.81 / 0.80 | 0.68 / 0.68 |
| 3 | 48 | 2.02 / 2.00 | 2.06 / 1.99 | 2.19 / 2.17 | 2.13 / 2.08 |
| 3 | 64 | 0.85 / 0.78 | 0.94 / 0.83 | 1.07 / 0.98 | 1.11 / 1.04 |
| 3 | 96 | 1.15 / 1.04 | 1.14 / 1.03 | 1.07 / 0.98 | 0.93 / 0.84 |
| 3 | 128 | 0.97 / 0.85 | 0.90 / 0.86 | 0.83 / 0.78 | 0.74 / 0.67 |
| 3 | 192 | 0.73 / 0.69 | 1.00 / 0.92 | 0.86 / 0.80 | 0.69 / 0.66 |
| 3 | 256 | 0.71 / 0.65 | 0.95 / 0.90 | 0.77 / 0.78 | 0.67 / 0.60 |
| 3 | 384 | 0.78 / 0.73 | 1.04 / 0.96 | 0.79 / 0.76 | 0.64 / 0.62 |
| 4 | 48 | 1.89 / 1.78 | 1.87 / 1.80 | 2.11 / 1.98 | 2.05 / 2.01 |
| 4 | 64 | 0.85 / 0.71 | 0.93 / 0.71 | 0.86 / 0.69 | 0.80 / 0.66 |
| 4 | 96 | 1.07 / 0.87 | 1.07 / 0.92 | 1.06 / 0.87 | 0.90 / 0.76 |
| 4 | 128 | 0.63 / 0.52 | 0.84 / 0.69 | 0.75 / 0.65 | 0.64 / 0.57 |
| 4 | 192 | 0.71 / 0.59 | 0.96 / 0.78 | 0.79 / 0.69 | 0.63 / 0.57 |
| 4 | 256 | 0.70 / 0.60 | 0.95 / 0.85 | 0.76 / 0.68 | 0.66 / 0.58 |
| 4 | 384 | 0.94 / 0.81 | 1.39 / 1.18 | 1.09 / 0.93 | 0.85 / 0.76 |
| 5 | 48 | 1.28 / 1.25 | 1.24 / 1.20 | 1.16 / 1.14 | 0.91 / 0.90 |
| 5 | 64 | 0.41 / 0.39 | 0.45 / 0.40 | 0.45 / 0.41 | 0.40 / 0.38 |
| 5 | 96 | 0.39 / 0.38 | 0.55 / 0.50 | 0.51 / 0.48 | 0.43 / 0.41 |
| 5 | 128 | 0.36 / 0.34 | 0.50 / 0.45 | 0.43 / 0.40 | 0.35 / 0.32 |
| 5 | 192 | 0.41 / 0.38 | 0.54 / 0.48 | 0.45 / 0.40 | 0.35 / 0.33 |
| 5 | 256 | 0.51 / 0.42 | 0.76 / 0.65 | 0.57 / 0.51 | 0.50 / 0.43 |
| 5 | 384 | 0.53 / 0.46 | 0.67 / 0.61 | 0.50 / 0.46 | 0.40 / 0.39 |

With a new scratch space for every product (`p = 2`, `bits(p^N) = 74`, `f = 4`, length 32)
Kronecker packing took 71 μs against 50 μs for Karatsuba.
"""
const KRONECKER_RULE = Dict(
    4 => (bmin_unit=typemax(Int), bmin=63, bmax=320, mmax=96, reuse=true),
    5 => (bmin_unit=63, bmin=63, bmax=typemax(Int), mmax=typemax(Int), reuse=false),
)

# Whether Kronecker packing is used for operands with m relevant z-coefficients, in the scratch
# space ws (nothing: a new one for every product).
function _use_kronecker(R::GaloisRing, m::Integer, ws::Union{Nothing,WScratch})
    FLINT_INPLACE || return false
    rule = get(KRONECKER_RULE, min(R.f, 5), nothing)
    rule === nothing && return false
    (rule.reuse && ws === nothing) && return false
    m <= rule.mmax || return false
    bits = nbits(R.pN)
    bits <= rule.bmax || return false
    bits >= rule.bmin_unit && return true
    bits >= rule.bmin || return false
    return ws === nothing || !_unit_reduction!(ws, R)
end

"""
    wpoly_mullow_method(R, A, B, len; ws=nothing) -> Symbol

The method used by [`wpoly_mullow`](@ref) for these operands, when computed in the scratch space
`ws` (`nothing`: a new scratch space for this product): `:single` for `f = 1`, `:schoolbook` when
`x`-schoolbook needs at most `max(K_f, 2f - 1)` polynomial products (`K_f` the number for
Karatsuba; only the pairs of nonzero components are multiplied, so this covers operands with
coefficients in `Z_p`), `:kronecker` by the benchmarked rule [`KRONECKER_RULE`](@ref), and
`:karatsuba` (`f ≤ 4`) or `:schoolbook` otherwise.
"""
function wpoly_mullow_method(
    R::GaloisRing,
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer;
    ws::Union{Nothing,WScratch}=nothing,
)
    f = R.f
    f == 1 && return :single
    nA = count(d -> !iszero(A[d]), 1:f)
    nB = A === B ? nA : count(d -> !iszero(B[d]), 1:f)
    nprod = nA * nB
    kara = _karatsuba_products(f)
    nprod <= max(kara, 2 * f - 1) && return :schoolbook
    m = min(len, _max_length(A, f), _max_length(B, f))
    _use_kronecker(R, m, ws) && return :kronecker
    return kara > 0 ? :karatsuba : :schoolbook
end

"""
    wpoly_mul!(R, P, A, B, len, ws; method=wpoly_mullow_method(R, A, B, len; ws=ws)) -> P

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
    method::Symbol=wpoly_mullow_method(R, A, B, len; ws=ws),
)
    _check_scratch(ws, R)
    if method === :single
        P[1] = poly_mullow!(P[1], A[1], B[1], len, ws)
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
across several products; the method defaults to `wpoly_mullow_method(R, A, B, len; ws=ws)`.

The `2f - 1` products `P_k = Σ_{d+e=k} A_d B_e` in `(Z/p^N)[z]` are computed by one of these
methods, then `x^k` (`k ≥ f`) is reduced modulo `h̃`:
- `:single` (`f = 1`): one polynomial product;
- `:schoolbook`: one `mullow` per pair of nonzero components (`f²` for full operands); chosen
  whenever these are at most `max(K_f, 2f - 1)`, `K_f` the number for Karatsuba (e.g. for
  operands with coefficients in `Z_p`), and for `f ≥ 5` without Kronecker packing;
- `:karatsuba` (`f = 2, 3, 4`): Karatsuba in `x`, with 3, 6 and 9 `mullow`s;
- `:kronecker`: Kronecker packing in `x` into one product over `Z` with coefficients of about
  `4f·bits(p^N)` bits, followed by bit extraction; the reduction modulo `h̃` is done over `Z`.

A polynomial product in `(Z/p^N)[z]` is FLINT's `mullow`, or, in a scratch space and for the
sizes of `Syntomic.KRON_KERNEL_RULE` (`p^N` of several limbs), the Kronecker substitution kernel
`Syntomic.kron_mullow!` on reused limb buffers, with the same result.

For operands with all `f` components nonzero and `bits(p^N) ≥ 63`, Kronecker packing is used for
`f ≥ 5`, and for `f = 4` in a benchmarked range of `bits(p^N)` and lengths when the scratch space
is reused and the reduction modulo `h̃` is not by additions alone (`Syntomic.KRONECKER_RULE`,
whose docstring has the measurements); otherwise Karatsuba (`f ≤ 4`) or schoolbook. For
example, at `bits(p^N) = 128` and length 32, Kronecker packing takes 0.43 (`f = 5`) of the time
of schoolbook and 0.65 (`f = 4`, `p = 3`) of the time of Karatsuba, and for `f = 2` it is slower
(1.30, `p = 2`).
"""
function wpoly_mullow(
    R::GaloisRing,
    A::Vector{ZZModPolyRingElem},
    B::Vector{ZZModPolyRingElem},
    len::Integer;
    ws::Union{Nothing,WScratch}=nothing,
    method::Symbol=wpoly_mullow_method(R, A, B, len; ws=ws),
)
    f = R.f
    len <= 0 && return wpoly_zero(R)
    if f == 1
        ws === nothing || _check_scratch(ws, R)
        return [poly_mullow!(zero(R.Znz), A[1], B[1], len, ws)]
    end
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
    wpoly_lazy_zero(R) -> Vector{ZZPolyRingElem}

A zero accumulator for [`wpoly_lazy_addmul_scalar!`](@ref): `f` new integer polynomials, reduced
modulo `p^N` only at the end, by [`wpoly_lazy_reduce`](@ref).
"""
wpoly_lazy_zero(R::GaloisRing) = [zero(R.Zz) for _ in 1:(R.f)]

"""
    wpoly_lazy_addmul_scalar!(R, acc, A, a::WElem, ws) -> acc

`acc + a·A` as in [`wpoly_addmul_scalar!`](@ref), but over `Z`: the integer accumulator `acc`
(owned by the caller) is not reduced modulo `p^N`, which saves the reduction of every coefficient
for every term of a long sum. `ws` provides scratch space.
"""
function wpoly_lazy_addmul_scalar!(
    R::GaloisRing,
    acc::Vector{ZZPolyRingElem},
    A::Vector{ZZModPolyRingElem},
    a::WElem,
    ws::WScratch,
)
    _check_scratch(ws, R)
    f = R.f
    all(iszero, A) && return acc
    M = f == 1 ? nothing : scalar_matrix(R, a)
    Z = _zz!(ws, R, 1)
    for e in 1:f
        iszero(A[e]) && continue
        Z[1] = poly_lift_trunc!(Z[1], A[e], length(A[e]))
        for j in 1:f
            c = M === nothing ? a[1] : M[j, e]
            iszero(c) && continue
            acc[j] = isone(c) ? add!(acc[j], acc[j], Z[1]) : addmul!(acc[j], Z[1], c)
        end
    end
    return acc
end

"""
    wpoly_lazy_reduce(R, acc) -> Vector{ZZModPolyRingElem}

The W-polynomial `acc mod p^N` for an integer accumulator of
[`wpoly_lazy_addmul_scalar!`](@ref), as new polynomials.
"""
wpoly_lazy_reduce(R::GaloisRing, acc::Vector{ZZPolyRingElem}) =
    [poly_reduce!(zero(R.Znz), c) for c in acc]

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
        iszero(a) || (v = _poly_valuation(Val(FLINT_INPLACE), a, v))
    end
    return v
end

# The index of the first nonzero coefficient of a below `bound`, or `bound` if there is none. With
# in-place FLINT access the coefficients are read without creating integers: FLINT stores zero
# as the small integer 0.
function _poly_valuation(::Val{true}, a::ZZModPolyRingElem, bound::Int)
    GC.@preserve a begin
        c = Ptr{Int}(a.coeffs)
        for k in 1:min(length(a), bound)
            iszero(unsafe_load(c, k)) || return k - 1
        end
    end
    return bound
end

function _poly_valuation(::Val{false}, a::ZZModPolyRingElem, bound::Int)
    for k in 0:(min(length(a), bound) - 1)
        iszero(coeff(a, k)) || return k
    end
    return bound
end

"""
    wpoly_shift_right(A, v) -> Vector{ZZModPolyRingElem}

`A / z^v` for a W-polynomial divisible by `z^v`.
"""
wpoly_shift_right(A::Vector{ZZModPolyRingElem}, v::Integer) =
    (v == 0 || v >= ZERO_VALUATION) ? A : [shift_right(a, v) for a in A]
