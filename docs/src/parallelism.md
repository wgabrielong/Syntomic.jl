# [Parallelism](@id parallelism)

## Threads

Start Julia with several threads, e.g. `julia -t auto` (on Apple silicon, the number of
performance cores). Within one computation the following run in parallel when `threaded = true`
(the default):

- the envelope of `O_K/ϖ^n` (Steps 6, 8, 9) and the envelope of `O_K` (Steps 3–5) are built
  concurrently;
- the columns of the matrices of Steps 5, 6, 8 and 9;
- in Steps 7–10: the expansion of the W-matrices to `Z_p`-matrices (for `f > 1`), the triangular
  solves (in blocks of columns of the right-hand side, for matrices of size about 64 and more),
  and the exponent of `can` on the `∇`-terms concurrently with Step 7;
- the Smith normal forms of the two syntomic differentials.

All arithmetic is exact modular arithmetic, so the results are bit-for-bit identical for any
number of threads; `threaded = false` runs everything serially. FLINT itself stays
single-threaded.

If a parallel piece of work fails (for instance with [`PrecisionExhausted`](@ref)), no new pieces
are started and the running ones are finished before the exception is rethrown; the exception is
the one a serial computation would raise first, with the same type.

## Sweeps

[`syntomic_sweep`](@ref) computes `Z_p(i)(O_K/ϖ^n)` for all `n` and `i` in given ranges. The
envelope of `O_K`, the unit `w(v)` and the matrices `∇` for `O_K` do not depend on `n`
([AKN, Remark 4.67, p. 46]), so they are computed once for the largest F-precision `n·i` and each
weight `i`; only the remaining steps are repeated for every pair, in parallel. All pairs use the
same presentation of the rings (see [`Syntomic.descent_ring`](@ref)), and the Eisenstein
polynomial must be known to the precision that the largest `n` requires.

```julia
results = syntomic_sweep(zpn(2, 2), 2:5, 1:20)      # 80 pairs (n, i)
k_theory_table(zpn(3, 3), 1:10; integral=true)       # K_0, …, K_19 of Z/27
```

### Several rings and residue degrees

A sweep over a vector of rings returns one vector of results per ring. Rings whose computations
use the same ring ([`Syntomic.computing_ring`](@ref)) form a group that shares all the work up to
the W-matrices: with the default `descent = true`, this is the case for the truncated Witt
vectors `W_n(F_q)`, `q = p, p^2, p^3, …` (all computed with `Z/p^n`), and more generally for
rings over different residue fields whose Eisenstein polynomials lie in `Z_p[z]`. For a group, the
`O_K`-side data, the working precisions and the W-matrices of each pair are computed once; only
Steps 7–10, the expansion to `Z_p`-matrices and the Smith normal forms are computed for each
ring. The results are identical to those of separate sweeps.

```julia
Ws = [truncated_witt_vectors(3, f, 2) for f in 1:3]  # W_n(F_3), W_n(F_9), W_n(F_27)
syntomic_sweep(Ws, 2:3, 1:10)                        # one group: shared W-matrices
```

For example (8 threads, measured on a loaded machine), the grouped sweep for `W_n(F_{3^f})`,
`f = 1, 2, 3`, `n ∈ 2:3`, `i ∈ 1:12` takes 0.9 s against 1.7 s for three separate sweeps, and
the one for `W_n(F_{2^f})`, `f = 1, …, 4`, `n ∈ 2:4`, `i ∈ 1:10` takes 1.6–2.1 s against
3.7–4.0 s.

## Memory

Before any computation, a sweep estimates its peak memory ([`Syntomic.sweep_memory_estimate`](@ref))
and throws an `ArgumentError` if the estimate exceeds the free memory (`Sys.free_memory()`; on
macOS this counts only free pages, so it is conservative). Pass `check_memory = false` to skip
the check. With Distributed workers, each host must hold the pairs running on its workers.

The estimate counts the coefficients of the tables that are built, with the memory
`B = coefficient_bytes(p, N)` of one coefficient modulo `p^N` (8 bytes below `2^62`, otherwise a
GMP integer of `48 + 8⌈N log₂ p / 64⌉` bytes):

| part | model (bytes) |
|:-----|:--------------|
| exponents (Phase 0, policy `:exponent`) | the f-side estimate below at the first precision of `exponent_phase`, for the pairs running at the same time |
| envelope of `O_K` (Phases A, B) | `64 MiB + 3 · f · K_p(L) · L² · B`, where `K_p(L)` counts the tabulated normal forms of the g-envelope ([`Syntomic.g_table_entries`](@ref)) and `L = n_max · i_max` |
| `∇_{O_K}` for all weights | `|is| · f · (L-1)² · (B + 40)` |
| one pair `(n, i)`, `L = n·i` | `max(8 · f_c² · L³ · B, 2 · f_c · (c · B + 8 · e · L)) + 40 · (f(L-1))² · B`, with the numbers `e`, `c` of normal forms and coefficients of the f-envelope ([`Syntomic.f_table_entries`](@ref)) |

Here `f_c` is the residue degree of the ring used for the computation (1 with descent) and `f`
the one of the target ring. The term `8 · f_c² · L³ · B` is the memory allocated between two
garbage collections while the `L` columns of the matrices of a pair are built; the table of the
f-envelope dominates for `n = 2` and large `i`, where the number of monomials grows fastest. The
constants were calibrated against the peak resident memory of the parts of sweeps and of whole
sweeps (`benchmark/memory.jl`). The sweep needs the largest of Phase 0, Phases A and B, and
`∇_{O_K}` plus the pairs that run at the same time (the largest first, one per thread; this sum
is multiplied by 1.6 when several pairs run at the same time, since the heap grows further
between garbage collections when more memory is live).

Typical sizes, with 8 threads: the measured increase of the peak resident memory of the process
(after loading the package, about 0.6 GiB) and the estimate. In all calibration runs the estimate
was above the measured peak, by a factor between 1.1 and 3.5 for the parts and whole sweeps.

| sweep | measured peak | estimate |
|:------|--------------:|---------:|
| `Z/2^n`, `n ∈ 2:5`, `i ∈ 1:12` | 0.63 GiB | 1.12 GiB |
| `Z/3^n`, `n ∈ 2:4`, `i ∈ 1:15` | 0.56 GiB | 1.32 GiB |
| `W_n(F_8)`, `n ∈ 2:3`, `i ∈ 1:10` | 0.16 GiB | 0.45 GiB |
| `W_n(F_{3^f})`, `f ∈ 1:3`, `n ∈ 2:3`, `i ∈ 1:12` | 0.62 GiB | 0.92 GiB |
| `Z/2^n`, `n ∈ 2:3`, `i ∈ 1:30` | 3.67 GiB | 5.83 GiB |
| `Z/4`, `i ∈ 1:40` | 3.33 GiB | 4.02 GiB |
| `Z/4`, `i ∈ 1:50` | 9.05 GiB | 10.0 GiB |

The envelope of `O_K` alone takes 0.76 GiB at `p = 2`, `L = 100`, `N = 130`; for `n = 2` and
large `i` the tables of the envelopes of `O_K/ϖ^2` of the pairs running at the same time
dominate.

```@docs
Syntomic.sweep_memory_estimate
Syntomic.check_sweep_memory
Syntomic.available_memory
Syntomic.estimated_working_precision
Syntomic.shared_memory_estimate
Syntomic.nablaOK_memory_estimate
Syntomic.pair_memory_estimate
Syntomic.fside_memory_estimate
Syntomic.g_table_entries
Syntomic.f_table_entries
Syntomic.coefficient_bytes
```

## Distributed

Start worker processes with [`add_syntomic_workers`](@ref) (each loads only Nemo, Hecke and
Syntomic.jl) and pass `distributed = true`:

```julia
using Distributed, Syntomic
add_syntomic_workers(4; threads_per_worker=2)        # workers × threads ≤ Sys.CPU_THREADS
syntomic_sweep(zpn(2, 3), 2:4, 1:16; distributed=true)
syntomic_sweep([zpn(2, 2), zpn(3, 2), zpn(5, 2)], 2:3, 1:12; distributed=true)   # one group per worker
```

[`add_syntomic_workers`](@ref) refuses to start more threads in total than there are CPU threads
(`Sys.CPU_THREADS`) unless `oversubscribe = true`. A sweep over several groups of rings sends one
group to each worker if there are at least as many groups as workers; otherwise the pairs of each
group are distributed. Only plain data (the rings, the working precisions and the matrices `∇`
for `O_K`) is sent to the workers.

```@docs
Syntomic.SweepJob
Syntomic.run_sweep_job
Syntomic.shared_nabla_OK
Syntomic.sweep_precisions
Syntomic.sweep_groups
Syntomic.PlainWMatrix
```
