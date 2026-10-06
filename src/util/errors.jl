# Exception types used throughout the package.

"""
    PrecisionExhausted(msg)

Thrown when a computation runs out of p-adic precision: a division by `p^k` of an element
not known to enough digits, a failed divisibility check in a triangular solve, or an
elementary divisor that is indistinguishable from zero.

With `precision_policy = :adaptive` ([AKN24, Remark 7.3]) this exception is
caught and the computation is restarted at a higher precision.
"""
struct PrecisionExhausted <: Exception
    msg::String
end

function Base.showerror(io::IO, e::PrecisionExhausted)
    return print(io, "PrecisionExhausted: ", e.msg)
end

"""
    CertificateError(msg)

Thrown when the computed cohomology fails the final consistency check: the exponents of `H^1`
and `H^2` must satisfy `Σ h1 - Σ h2 = f·i·(n-1)`, since `|K_{2i-1}|/|K_{2i-2}| = q^{i(n-1)}`
([AKN24, Prop. 2.18, p. 9]). (A differential whose elementary divisors cannot be certified raises
[`PrecisionExhausted`](@ref) instead.)
"""
struct CertificateError <: Exception
    msg::String
end

Base.showerror(io::IO, e::CertificateError) = print(io, "CertificateError: ", e.msg)
