"""
    Syntomic

Syntomic cohomology `Z_p(i)(O_K/ϖ^n)` and p-adic algebraic K-theory of finite chain rings
`O_K/ϖ^n`, where `K/Q_p` is a finite extension with residue field `F_q`, `q = p^f`.

This is a Julia reimplementation, with extensions, of the algorithm of [AKN, §7.1] and of the
authors' SageMath implementation <https://github.com/antieau/kzpn> (GPL-3.0). It works over
`W(F_q)` rather than only `W(F_p)`, is split into small functions that each cite the step of the
algorithm they implement, and is parallel (threads within a computation, `Distributed` across
computations).

References:

- [AKN] B. Antieau, A. Krause, T. Nikolaus, *On the K-theory of Z/p^n*, arXiv:2405.04329.
  Citations such as "[AKN, §7.1, Step 3, p. 55]" refer to the steps of the algorithm in §7.1
  (pp. 53–57) and to page numbers of this paper.
"""
module Syntomic

using Nemo
using Hecke

using Distributed: Distributed
using LinearAlgebra: LinearAlgebra
using OhMyThreads: OhMyThreads
using PrecompileTools: PrecompileTools
using Serialization: Serialization

# Utilities
include("util/errors.jl")
include("util/threading.jl")

# Integer parameters and precision bookkeeping ([AKN, §7.1, Steps 1–2])
include("params.jl")
include("precision.jl")

# Arithmetic in W(F_q)/p^N and W(F_q)[[z]] ([AKN, §7.1, Step 2] presentation, Step 3 δ-structure)
include("arith/galois_ring.jl")
include("arith/kernels.jl")
include("arith/wseries.jl")

# Linear algebra over Z_p and W(F_q) (square completions, Smith normal form)
include("linalg/local_snf.jl")
include("linalg/wmatrix.jl")

export PrecisionExhausted, CertificateError
export SyntomicParams
export local_elementary_divisors

end # module Syntomic
