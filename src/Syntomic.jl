"""
    Syntomic

Syntomic cohomology `Z_p(i)(O_K/ϖ^n)` and p-adic algebraic K-theory of finite chain rings
`O_K/ϖ^n`, where `K/Q_p` is a finite extension with residue field `F_q`, `q = p^f`.

This is a Julia reimplementation, with extensions, of the algorithm in §7 of

> B. Antieau, A. Krause, T. Nikolaus, *On the K-theory of Z/p^n*, arXiv:2405.04329,

and of the authors' SageMath implementation <https://github.com/antieau/kzpn> (GPL-3.0).
It extends that code from `W(F_p)` to `W(F_q)`, is split into small functions that each cite
the Step of the algorithm (Algorithm.pdf, printed pp. 53–60) they implement, and is parallel
(threads within a computation, `Distributed` across computations).
"""
module Syntomic

using Nemo
using Hecke

using Distributed: Distributed
using LinearAlgebra: LinearAlgebra
using OhMyThreads: OhMyThreads
using PrecompileTools: PrecompileTools
using Serialization: Serialization

end # module Syntomic
