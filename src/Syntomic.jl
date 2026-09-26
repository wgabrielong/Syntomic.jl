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
using PrecompileTools: PrecompileTools

# Utilities
include("util/errors.jl")
include("util/threading.jl")
include("util/oncecache.jl")

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

# Input: the finite chain ring O_K/ϖ^n and its Eisenstein polynomial
include("input.jl")
include("eisenstein.jl")

# Step 3: prismatic envelopes
include("step3/lambdas.jl")
include("step3/envelope_f.jl")
include("step3/g_normal_form.jl")
include("step3/g_frobenius.jl")
include("step3/envelope_g.jl")

# Steps 4–10: the maps of the syntomic square
include("step4_orientation.jl")
include("step4_powers.jl")
include("step5_nabla_ok.jl")
include("step6_reduction.jl")
include("step7_nabla_r.jl")
include("step8_nabla_n.jl")
include("step9_can_phi.jl")
include("step10_can_phi_nabla.jl")

# Return: the syntomic complex, its cohomology, K-groups
include("plain.jl")
include("cohomology.jl")
include("pipeline.jl")
include("kgroups.jl")

# Sweeps over (n, i) with shared data; Distributed
include("sweep.jl")
include("memory.jl")

# Export of results as JSON or CSV
include("export.jl")

export PrecisionExhausted, CertificateError
export SyntomicParams
export FiniteChainRing, finite_chain_ring, zpn, truncated_witt_vectors
export residue_field, ramification_index, inertia_degree
export SyntomicComplex, SyntomicCohomology, syntomic_complex, syntomic_cohomology
export k_groups, k_theory_table, KTheoryTable
export save_results
export syntomic_sweep, add_syntomic_workers
export local_elementary_divisors

# Precompile the main code paths (f = 1 and f > 1, serial and threaded).
PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        syntomic_cohomology(zpn(2, 2), 3)
        syntomic_cohomology(truncated_witt_vectors(3, 2, 2), 2; descent=false)
        syntomic_cohomology(zpn(3, 2), 2; threaded=false, precision_policy=:apriori)
    end
end

end # module Syntomic
