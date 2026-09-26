# Accumulators for sums of products of polynomials in z over W(F_q)/p^N.
#
# The loops that combine normal forms of the g-envelope (the table of normal forms, the normal
# forms applied in a product, the transposed chain of Step 5) add up many products of
# W-polynomials. With the Kronecker substitution kernel the coefficients of a product are slots of
# a limb array; a `WAcc` adds the slots to limb accumulators instead of reducing every coefficient
# modulo p^N and adding polynomials: for p = 2 modulo 2^N (a mask), for odd p as exact sums that
# are reduced once, when the sum is read.

"""
    acc_reset!(acc, len) -> acc

Set the accumulator to zero modulo `z^len` (reusing its memory).
"""
function acc_reset!(acc::WAcc, len::Integer)
    n = max(Int(len), 0)
    if acc.S == 0
        for d in eachindex(acc.polys)
            acc.polys[d] = zero!(acc.polys[d])
        end
    else
        # only the coefficients below `used` can be nonzero
        fill!(view(acc.data, 1:(acc.S * acc.used)), zero(UInt64))
        m = length(acc.data)
        if m < acc.S * n
            resize!(acc.data, acc.S * n)
            fill!(view(acc.data, (m + 1):(acc.S * n)), zero(UInt64))
        end
    end
    acc.len = n
    acc.used = 0
    return acc
end

# Add the nl limbs tmp[1:nl] to the coefficient k (0-based) of the limb accumulator (p = 2: modulo
# 2^N, the top limb masked with `mask`; odd p: with the carry propagated through the S limbs).
@inline function _acc_add_limbs!(
    acc::WAcc, k::Int, tmp::Vector{UInt64}, nl::Int, mask::UInt64
)
    data = acc.data
    S = acc.S
    base = k * S
    carry = zero(UInt64)
    m = min(nl, S)
    @inbounds for j in 1:m
        s1, c1 = Base.add_with_overflow(data[base + j], tmp[j])
        s2, c2 = Base.add_with_overflow(s1, carry)
        data[base + j] = s2
        carry = UInt64(c1 | c2)
    end
    j = m + 1
    @inbounds while carry != 0 && j <= S
        s1, c1 = Base.add_with_overflow(data[base + j], carry)
        data[base + j] = s1
        carry = UInt64(c1)
        j += 1
    end
    @inbounds data[base + S] &= mask
    acc.used = max(acc.used, k + 1)
    return nothing
end

# The mask of the top limb of a coefficient (the low N bits modulo 2^N for p = 2).
function _acc_mask(acc::WAcc)
    acc.R.p == 2 || return typemax(UInt64)
    r = acc.R.N & 63
    return r == 0 ? typemax(UInt64) : (UInt64(1) << r) - one(UInt64)
end

"""
    acc_add!(acc, A, shift=0) -> acc

`acc + z^shift·A` for a W-polynomial `A` (only its coefficients below `z^{len - shift}` are
used).
"""
function acc_add!(acc::WAcc, A::Vector{ZZModPolyRingElem}, shift::Integer=0)
    R = acc.R
    n = acc.len - shift
    n <= 0 && return acc
    if acc.S == 0
        for d in 1:(R.f)
            iszero(A[d]) && continue
            t = shift == 0 ? A[d] : shift_left(A[d], Int(shift))
            acc.polys[d] = poly_add_trunc!(acc.polys[d], acc.polys[d], t, acc.len)
        end
        return acc
    end
    _acc_add_poly!(acc, A[1], Int(shift))
    return acc
end

# Add the coefficients of the polynomial a (residues in [0, p^N)) at z^shift, z^{shift+1}, ….
function _acc_add_poly!(acc::WAcc, a::ZZModPolyRingElem, shift::Int)
    K = _modulus_limbs(acc.R)
    tmp = acc.tmp
    mask = _acc_mask(acc)
    GC.@preserve a tmp begin
        c = a.coeffs
        for k in 0:(min(a.length, acc.len - shift) - 1)
            v = unsafe_load(Ptr{Int}(c), k + 1)
            iszero(v) && continue
            @ccall Nemo.libflint.fmpz_get_ui_array(
                pointer(tmp)::Ptr{UInt64}, K::Int, (c + 8 * k)::Ptr{Nothing}
            )::Nothing
            _acc_add_limbs!(acc, shift + k, tmp, K, mask)
        end
    end
    return nothing
end

"""
    acc_addmul!(acc, A, vA, B, vB, ws; factor=1) -> acc

`acc + factor·(z^{vA} A)(z^{vB} B) mod z^len` for W-polynomials `A`, `B` given with their
valuation shifts removed (as in [`wpoly_addmul!`](@ref)) and `factor` 1 or 2; `ws` is scratch space
of the calling task. With limb coefficients, a product of the Kronecker substitution kernel is
added slot by slot, without reducing its coefficients (see [`WAcc`](@ref)).
"""
function acc_addmul!(
    acc::WAcc,
    A::Vector{ZZModPolyRingElem},
    vA::Integer,
    B::Vector{ZZModPolyRingElem},
    vB::Integer,
    ws::WScratch;
    factor::Int=1,
)
    R = acc.R
    s = Int(vA + vB)
    n = acc.len - s
    n <= 0 && return acc
    if acc.S == 0
        P = wpoly_product!(R, ws, A, vA, B, vB, acc.len)
        P === nothing && return acc
        factor == 2 && wpoly_double!(P, R.f)
        wpoly_add!(acc.polys, P)
        return acc
    end
    x, y = A[1], B[1]
    lx, ly = min(length(x), n), min(length(y), n)
    (lx == 0 || ly == 0) && return acc
    if kron_kernel_applies(R, lx, ly)
        kb = _kron!(ws, R)
        bits, slot, nout = _kron_product!(kb, x, y, n)
        for _ in 1:factor
            _acc_add_slots!(acc, kb, bits, slot, nout, s)
        end
    else
        P = ws.prods
        P[1] = poly_mullow!(P[1], x, y, n, ws)
        for _ in 1:factor
            _acc_add_poly!(acc, P[1], s)
        end
    end
    return acc
end

# Add the nout coefficients of a product of the kernel (slots of `bits` bits, `slot` limbs, in
# kb.prod) at z^shift, z^{shift+1}, ….
function _acc_add_slots!(
    acc::WAcc, kb::KronBuffers, bits::Int, slot::Int, nout::Int, shift::Int
)
    tmp = kb.tmp
    prod = kb.prod
    if kb.p2
        # the low N bits of a slot are its residue modulo 2^N
        kn = cld(kb.N, 64)
        mask = _acc_mask(acc)
        for k in 0:(nout - 1)
            _kron_read_slot!(tmp, prod, k * bits, kn)
            @inbounds tmp[kn] &= mask
            _acc_add_limbs!(acc, shift + k, tmp, kn, mask)
        end
    else
        top = bits & 63
        topmask = top == 0 ? typemax(UInt64) : (UInt64(1) << top) - one(UInt64)
        for k in 0:(nout - 1)
            _kron_read_slot!(tmp, prod, k * bits, slot)
            @inbounds tmp[slot] &= topmask
            _acc_add_limbs!(acc, shift + k, tmp, slot, typemax(UInt64))
        end
    end
    return nothing
end

"""
    acc_finish!(acc) -> Vector{ZZModPolyRingElem}

The sum as a new W-polynomial (owned by the caller), each coefficient reduced modulo `p^N`; the
accumulator is set to zero.
"""
function acc_finish!(acc::WAcc)
    R = acc.R
    if acc.S == 0
        out = acc.polys
        acc.polys = wpoly_zero(R)
        return out
    end
    z = zero(R.Znz)
    n = acc.used
    if n > 0
        K = _modulus_limbs(R)
        S = acc.S
        m, q, r = acc.m, acc.q, acc.r
        @ccall Nemo.libflint.fmpz_mod_poly_fit_length(
            z::Ref{ZZModPolyRingElem},
            n::Int,
            _fmpz_mod_ctx(z)::Ref{Nemo.fmpz_mod_ctx_struct},
        )::Nothing
        data = acc.data
        GC.@preserve z data m q r begin
            zc = z.coeffs
            for k in 0:(n - 1)
                p = pointer(data, k * S + 1)
                if R.p == 2
                    # already reduced modulo 2^N
                    @ccall Nemo.libflint.fmpz_set_ui_array(
                        (zc + 8 * k)::Ptr{Nothing}, p::Ptr{UInt64}, S::Int
                    )::Nothing
                    continue
                end
                nl = S
                while nl > K && @inbounds(data[k * S + nl]) == 0
                    nl -= 1
                end
                if nl == K && _acc_less(data, k * S + 1, m, K)
                    @ccall Nemo.libflint.fmpz_set_ui_array(
                        (zc + 8 * k)::Ptr{Nothing}, p::Ptr{UInt64}, K::Int
                    )::Nothing
                else
                    @ccall Base.GMP.libgmp.__gmpn_tdiv_qr(
                        pointer(q)::Ptr{UInt64},
                        pointer(r)::Ptr{UInt64},
                        0::Int,
                        p::Ptr{UInt64},
                        nl::Int,
                        pointer(m)::Ptr{UInt64},
                        K::Int,
                    )::Nothing
                    @ccall Nemo.libflint.fmpz_set_ui_array(
                        (zc + 8 * k)::Ptr{Nothing}, pointer(r)::Ptr{UInt64}, K::Int
                    )::Nothing
                end
            end
        end
        @ccall Nemo.libflint._fmpz_mod_poly_set_length(
            z::Ref{ZZModPolyRingElem}, n::Int
        )::Nothing
        @ccall Nemo.libflint._fmpz_mod_poly_normalise(z::Ref{ZZModPolyRingElem})::Nothing
    end
    acc_reset!(acc, acc.len)
    return [z]
end
