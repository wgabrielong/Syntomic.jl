# [API](@id api)

## Rings

```@docs
FiniteChainRing
finite_chain_ring
zpn
truncated_witt_vectors
Syntomic.with_power
Syntomic.prime(::FiniteChainRing)
Syntomic.ramification_index(::FiniteChainRing)
Syntomic.inertia_degree(::FiniteChainRing)
Syntomic.residue_field(::FiniteChainRing)
SyntomicParams
```

## Syntomic cohomology and K-theory

```@docs
syntomic_cohomology
syntomic_complex
SyntomicCohomology
SyntomicComplex
k_groups
k_theory_table
KTheoryTable
Syntomic.KTheoryRow
Syntomic.group_string
Syntomic.group_summands
Syntomic.cyclic_primary_decomposition
```

## Export

```@docs
save_results
```

## Sweeps and Distributed

```@docs
syntomic_sweep
add_syntomic_workers
```

## Linear algebra

```@docs
local_elementary_divisors
```

## Errors

```@docs
PrecisionExhausted
CertificateError
```

## Internals

Data structures and utilities that are not specific to a step of the algorithm.

```@docs
Syntomic.computing_ring
Syntomic.WMatrix
Syntomic.PMatrix
Syntomic.wpoly_mullow
Syntomic.OnceCache
Syntomic.get_or_compute!
Syntomic.set_computed!
Syntomic.PARALLEL_FILL_WORK
Syntomic.tmap
Syntomic.tforeach
Syntomic.concurrently
Syntomic.unwrap_task_exception
Syntomic.rethrow_unwrapped
```
