# The Galois ring W(F_q)/p^N = (Z/p^N)[x]/(h̃(x)).
#
# [AKN, §7.1, Step 2, p. 54]: "Choose a basis 1, x, x², …, x^{f-1} for F_q over F_p
# and lift this to a basis for W(F_q) over Z_p … we present W(F_q) = Z_p[x]/h̃(x) where h̃ is an
# arbitrary Z_p-lift of the minimal polynomial h(x) used to present F_q."
#
# Elements of W(F_q)/p^N are stored as coordinate vectors (c_0, …, c_{f-1}) ∈ (Z/p^N)^f with
# respect to 1, x, …, x^{f-1}, each coordinate a representative in [0, p^N). The Witt vector
# Frobenius σ is the unique ring automorphism with σ(x) ≡ x^p (mod p); it is computed natively by
# Newton iteration on h̃, so it never depends on the precision of a q-adic field.

"""
    WElem

An element of `W(F_q)/p^N`: the vector of its coordinates `(c_0, …, c_{f-1})` in the basis
`1, x, …, x^{f-1}`, each a `ZZRingElem` in `[0, p^N)`.
"""
const WElem = Vector{ZZRingElem}

"""
    IntegerLike

Machine or FLINT integers.
"""
const IntegerLike = Union{Integer,ZZRingElem}

"""
    GaloisRing(p, htilde, N)

The Galois ring `W(F_q)/p^N = (Z/p^N)[x]/(h̃)`, `q = p^f`, `f = deg h̃` ([AKN, §7.1, Step 2]).

`htilde` is a monic integer polynomial, given by its coefficients `[h_0, …, h_{f-1}, 1]`, whose
reduction modulo `p` is irreducible. For `f = 1` use `htilde = [0, 1]`.

Fields: `p`, `f`, `N`, `pN = p^N`, the coefficient ring `Zn = Z/p^N`, the polynomial rings
`Znz = (Z/p^N)[z]` and `Zz = Z[z]` used for power series in `z`, `htilde`, the coordinates
`xpow[k+1, :]` of `x^k` for `0 ≤ k ≤ 2f-2`, and the `f×f` matrix `sigma` of the Frobenius
(column `d+1` holds the coordinates of `σ(x)^d`), so that `ι(σ(c)) = sigma * ι(c)`.
"""
struct GaloisRing
    p::Int
    f::Int
    N::Int
    pN::ZZRingElem
    Zn::ZZModRing
    Znz::ZZModPolyRing
    Zz::ZZPolyRing
    htilde::Vector{ZZRingElem}
    xpow::Matrix{ZZRingElem}
    sigma::Matrix{ZZRingElem}
end

function Base.show(io::IO, R::GaloisRing)
    return print(io, "GaloisRing W(F_$(R.p)^$(R.f))/$(R.p)^$(R.N)")
end

function GaloisRing(p::Integer, htilde::AbstractVector, N::Integer)
    Nemo.is_prime(ZZ(p)) || throw(ArgumentError("GaloisRing: p = $p is not prime"))
    N >= 1 || throw(ArgumentError("GaloisRing: the precision N must be positive, got $N"))
    h = ZZRingElem[ZZ(c) for c in htilde]
    f = length(h) - 1
    f >= 1 || throw(ArgumentError("GaloisRing: h̃ must have degree at least 1"))
    isone(h[end]) || throw(ArgumentError("GaloisRing: h̃ must be monic, got $h"))
    if f > 1
        Fp = Nemo.Native.GF(p)
        Fpx, _ = polynomial_ring(Fp, :x; cached=false)
        Nemo.is_irreducible(Fpx([Fp(c) for c in h])) || throw(ArgumentError("GaloisRing: h̃ = $h is not irreducible modulo $p"))
    end
    pN = ZZ(p)^N
    Zn, _ = residue_ring(ZZ, pN)
    Znz, _ = polynomial_ring(Zn, :z; cached=false)
    Zz, _ = polynomial_ring(ZZ, :z; cached=false)
    xpow = _powers_of_generator(h, f, pN)
    sigma = Matrix{ZZRingElem}(undef, f, f)
    R = GaloisRing(Int(p), f, Int(N), pN, Zn, Znz, Zz, h, xpow, sigma)
    _compute_frobenius_matrix!(R)
    return R
end

"""
    GaloisRing(p, N)

The ring `Z/p^N` (the case `f = 1`, `W(F_p) = Z_p`).
"""
GaloisRing(p::Integer, N::Integer) = GaloisRing(p, [0, 1], N)

"""
    change_precision(R::GaloisRing, N) -> GaloisRing

The same Galois ring (same `p` and `h̃`) at a different p-adic precision `N`.
"""
change_precision(R::GaloisRing, N::Integer) = GaloisRing(R.p, R.htilde, N)

# Coordinates of x^k (0 ≤ k ≤ 2f-2) modulo p^N, using x^f = -(h_0 + … + h_{f-1} x^{f-1}).
function _powers_of_generator(h::Vector{ZZRingElem}, f::Int, pN::ZZRingElem)
    K = max(2 * f - 1, f)
    xp = zeros(ZZRingElem, K, f)
    xp[1, 1] = one(ZZ)
    for k in 1:(K - 1)
        prev = xp[k, :]
        cur = zeros(ZZRingElem, f)
        # multiply prev by x: shift up, then reduce the x^f term
        for d in 1:(f - 1)
            cur[d + 1] = prev[d]
        end
        top = prev[f]
        for d in 1:f
            cur[d] = mod(cur[d] - top * h[d], pN)
        end
        xp[k + 1, :] = cur
    end
    return xp
end

# ---------------------------------------------------------------------------------------------
# Scalar arithmetic in W(F_q)/p^N
# ---------------------------------------------------------------------------------------------

"""
    w_zero(R) -> WElem
"""
w_zero(R::GaloisRing) = zeros(ZZRingElem, R.f)

"""
    w_one(R) -> WElem
"""
w_one(R::GaloisRing) = w_int(R, 1)

"""
    w_int(R, k) -> WElem

The image of the integer `k` in `W(F_q)/p^N`.
"""
function w_int(R::GaloisRing, k::IntegerLike)
    a = w_zero(R)
    a[1] = mod(ZZ(k), R.pN)
    return a
end

"""
    w_gen(R) -> WElem

The generator `x` of `W(F_q)/p^N = (Z/p^N)[x]/h̃` ([AKN, §7.1, Step 2]). For `f = 1` this is the
image of `-h_0`.
"""
function w_gen(R::GaloisRing)
    R.f == 1 && return w_int(R, -R.htilde[1])
    a = w_zero(R)
    a[2] = one(ZZ)
    return a
end

"""
    w_from_coordinates(R, v) -> WElem

The element with coordinates `v` (integers, reduced modulo `p^N`).
"""
function w_from_coordinates(R::GaloisRing, v::AbstractVector)
    length(v) == R.f ||
        throw(ArgumentError("expected $(R.f) coordinates, got $(length(v))"))
    return ZZRingElem[mod(ZZ(c), R.pN) for c in v]
end

function w_add(R::GaloisRing, a::WElem, b::WElem)
    return ZZRingElem[mod(a[d] + b[d], R.pN) for d in 1:(R.f)]
end
function w_sub(R::GaloisRing, a::WElem, b::WElem)
    return ZZRingElem[mod(a[d] - b[d], R.pN) for d in 1:(R.f)]
end
w_neg(R::GaloisRing, a::WElem) = ZZRingElem[mod(-a[d], R.pN) for d in 1:(R.f)]

"""
    w_mul_int(R, a, k) -> WElem

`k·a` for an integer `k`.
"""
w_mul_int(R::GaloisRing, a::WElem, k::IntegerLike) =
    ZZRingElem[mod(k * a[d], R.pN) for d in 1:(R.f)]

"""
    w_mul(R, a, b) -> WElem

The product in `W(F_q)/p^N` (schoolbook multiplication in `x`, then reduction modulo `h̃`).
"""
function w_mul(R::GaloisRing, a::WElem, b::WElem)
    f = R.f
    f == 1 && return ZZRingElem[mod(a[1] * b[1], R.pN)]
    prod = zeros(ZZRingElem, 2 * f - 1)
    for d in 1:f, e in 1:f
        prod[d + e - 1] += a[d] * b[e]
    end
    c = zeros(ZZRingElem, f)
    for k in 1:(2 * f - 1)
        iszero(prod[k]) && continue
        for d in 1:f
            c[d] += prod[k] * R.xpow[k, d]
        end
    end
    return ZZRingElem[mod(c[d], R.pN) for d in 1:f]
end

"""
    w_pow(R, a, e) -> WElem

`a^e` for an integer `e ≥ 0` (binary powering).
"""
function w_pow(R::GaloisRing, a::WElem, e::Integer)
    e >= 0 || throw(ArgumentError("w_pow: negative exponent"))
    result = w_one(R)
    base = a
    k = e
    while k > 0
        isodd(k) && (result = w_mul(R, result, base))
        k >>= 1
        k > 0 && (base = w_mul(R, base, base))
    end
    return result
end

"""
    w_iszero(R, a; prec=R.N) -> Bool

Whether `a ≡ 0 (mod p^prec)`.
"""
function w_iszero(R::GaloisRing, a::WElem; prec::Integer=R.N)
    m = ZZ(R.p)^prec
    return all(iszero(mod(c, m)) for c in a)
end

"""
    w_isequal(R, a, b; prec=R.N) -> Bool

Whether `a ≡ b (mod p^prec)`.
"""
w_isequal(R::GaloisRing, a::WElem, b::WElem; prec::Integer=R.N) =
    w_iszero(R, w_sub(R, a, b); prec=prec)

"""
    w_valuation(R, a; prec=R.N) -> Int

The p-adic valuation of `a`, capped at `prec` (so `w_valuation(R, 0) == prec`). Since `p` is a
uniformizer of `W(F_q)`, this is the minimum of the valuations of the coordinates.
"""
function w_valuation(R::GaloisRing, a::WElem; prec::Integer=R.N)
    v = Int(prec)
    for c in a
        cc = mod(c, ZZ(R.p)^prec)
        iszero(cc) && continue
        v = min(v, valuation(cc, R.p))
    end
    return v
end

"""
    w_isunit(R, a) -> Bool

Whether `a` is a unit of `W(F_q)/p^N`, i.e. `a ≢ 0 (mod p)`.
"""
w_isunit(R::GaloisRing, a::WElem) = w_valuation(R, a; prec=1) == 0

"""
    mult_matrix(R, a) -> Matrix{ZZRingElem}

The `f×f` matrix `M(a)` of multiplication by `a` on `W(F_q)/p^N ≅ (Z/p^N)^f`: column `d+1` holds
the coordinates of `a·x^d`. It satisfies `M(ab) = M(a)M(b)` and is used to expand
`W(F_q)`-linear maps into `Z_p`-matrices ([AKN, §7.1, Step 2], last paragraph).
"""
function mult_matrix(R::GaloisRing, a::WElem)
    f = R.f
    M = Matrix{ZZRingElem}(undef, f, f)
    xd = w_one(R)
    x = f == 1 ? w_one(R) : w_gen(R)
    for d in 1:f
        col = w_mul(R, a, xd)
        M[:, d] = col
        d < f && (xd = w_mul(R, xd, x))
    end
    return M
end

"""
    w_inv(R, a) -> WElem

The inverse of a unit `a` of `W(F_q)/p^N`. Throws a `DomainError` if `a` is not a unit.
"""
function w_inv(R::GaloisRing, a::WElem)
    w_isunit(R, a) || throw(DomainError(a, "w_inv: not a unit of W(F_q)/p^N"))
    R.f == 1 && return ZZRingElem[invmod(a[1], R.pN)]
    M = matrix(R.Zn, mult_matrix(R, a))
    Minv = inv(M)
    return ZZRingElem[lift(Minv[d, 1]) for d in 1:(R.f)]
end

"""
    w_divexact_p(R, a, k) -> WElem

`a / p^k` for `a` divisible by `p^k` (the result is determined modulo `p^{N-k}`; the returned
representative has its top `k` digits equal to zero). Throws [`PrecisionExhausted`](@ref) if
`a` is not divisible by `p^k`.
"""
function w_divexact_p(R::GaloisRing, a::WElem, k::Integer)
    k == 0 && return copy(a)
    pk = ZZ(R.p)^k
    out = similar(a)
    for d in 1:(R.f)
        q, r = divrem(a[d], pk)
        iszero(r) ||
            throw(PrecisionExhausted("w_divexact_p: element is not divisible by $(R.p)^$k"))
        out[d] = q
    end
    return out
end

"""
    w_frobenius(R, a) -> WElem

The Witt vector Frobenius `σ(a)`: the ring automorphism of `W(F_q)` lifting `t ↦ t^p` on `F_q`.
"""
function w_frobenius(R::GaloisRing, a::WElem)
    R.f == 1 && return copy(a)
    f = R.f
    out = zeros(ZZRingElem, f)
    for d in 1:f, e in 1:f
        out[d] += R.sigma[d, e] * a[e]
    end
    return ZZRingElem[mod(c, R.pN) for c in out]
end

# Evaluate the integer polynomial h̃ (or its derivative) at a ∈ W(F_q)/p^N by Horner's rule.
function _evaluate_integer_polynomial(R::GaloisRing, coeffs::Vector{ZZRingElem}, a::WElem)
    acc = w_int(R, coeffs[end])
    for k in (length(coeffs) - 1):-1:1
        acc = w_add(R, w_mul(R, acc, a), w_int(R, coeffs[k]))
    end
    return acc
end

# σ(x) is the unique root of h̃ in W(F_q)/p^N congruent to x^p modulo p (Hensel's lemma; h̃ is
# separable modulo p). Newton iteration r ← r - h̃(r)/h̃'(r) doubles the precision each step.
function _compute_frobenius_matrix!(R::GaloisRing)
    f = R.f
    if f == 1
        R.sigma[1, 1] = one(ZZ)
        return R
    end
    h = R.htilde
    dh = ZZRingElem[k * h[k + 1] for k in 1:f]
    r = w_pow(R, w_gen(R), R.p)
    for _ in 1:(2 * ceil(Int, log2(R.N + 1)) + 8)
        hr = _evaluate_integer_polynomial(R, h, r)
        w_iszero(R, hr) && break
        r = w_sub(R, r, w_mul(R, hr, w_inv(R, _evaluate_integer_polynomial(R, dh, r))))
    end
    w_iszero(R, _evaluate_integer_polynomial(R, h, r)) ||
        error("Newton iteration for σ(x) did not converge (this is a bug)")
    rd = w_one(R)
    for d in 1:f
        R.sigma[:, d] = rd
        d < f && (rd = w_mul(R, rd, r))
    end
    return R
end
