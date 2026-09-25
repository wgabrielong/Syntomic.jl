# Exception types used throughout the package.

"""
    PrecisionExhausted(msg)

Thrown when a computation runs out of p-adic precision: a division by `p^k` of an element
not known to enough digits, a failed divisibility check in a triangular solve, or an
elementary divisor that is indistinguishable from zero.

With `precision_policy = :adaptive` ([AKN, Remark 7.3]) this exception is
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

Thrown when the final certificate of a syntomic cohomology computation fails ([AKN, Lemma 7.1 and
Remark 7.3]): wrong rank of a differential, an elementary divisor
whose valuation is not below the certified precision, or a failed Euler-characteristic check.
"""
struct CertificateError <: Exception
    msg::String
end

Base.showerror(io::IO, e::CertificateError) = print(io, "CertificateError: ", e.msg)
