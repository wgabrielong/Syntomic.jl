# [Parallelism](@id parallelism)

## Threads

Start Julia with several threads, e.g. `julia -t auto` (on Apple silicon, the number of
performance cores). Within one computation the following run in parallel when `threaded = true`
(the default):

- the envelope of `O_K/ϖ^n` (Steps 6, 8, 9) and the envelope of `O_K` (Steps 3–5) are built
  concurrently; at the `:exponent` precision, the exact exponents of `red` and `can` are computed
  while the envelope of `O_K` is built at an estimate of that precision (see the Precision page);
- in the envelope of `O_K`: the normal forms of the monomials that products can meet, computed
  level by level as soon as the relations they need are known (a normal form only needs those of
  earlier levels, [`Syntomic.g_fill_normal_forms!`](@ref)), the normal forms of a level and the
  components of the expensive ones in parallel; the rows of every product and the normal forms
  applied in it; `φ`; and the transposed chain of Step 5 concurrently with `w(v^i)`;
- the columns of the matrices of Steps 5, 6, 8 and 9;
- in Steps 7–10: the expansion of the W-matrices to `Z_p`-matrices (for `f > 1`), the triangular
  solves (in blocks of columns of the right-hand side, for matrices of size about 100 and more,
  i.e. `n²·c ≥ Syntomic.PARALLEL_SOLVE_WORK`), and from that size on the exponent of `can` on the
  `∇`-terms concurrently with Step 7;
- the Smith normal forms of the two syntomic differentials.

All arithmetic is exact modular arithmetic, so the results are bit-for-bit identical for any
number of threads; `threaded = false` runs everything serially. FLINT itself stays
single-threaded.

Measured with `benchmark/report.jl` (Apple M2 Max, each case in a fresh process, minimum of 3
runs; seconds, and the speedup over one thread):

| case | 1 thread | 2 threads | 4 threads | 8 threads |
|:-----|---------:|----------:|----------:|----------:|
| `syntomic_cohomology(zpn(2, 2), 30)` | 0.171 | 0.150 (1.1×) | 0.114 (1.5×) | 0.082 (2.1×) |
| `syntomic_cohomology(zpn(2, 2), 40)` | 0.795 | 0.559 (1.4×) | 0.387 (2.1×) | 0.314 (2.5×) |
| `syntomic_cohomology(zpn(3, 4), 8)` | 0.010 | 0.010 (1.0×) | 0.010 (1.0×) | 0.009 (1.1×) |
| `z² + 2az + 2a` over `Q_4`, `n = 4`, `i = 12` | 0.241 | 0.173 (1.4×) | 0.164 (1.5×) | 0.153 (1.6×) |
| `syntomic_sweep(zpn(2, 2), 2:5, 1:12)` | 0.806 | 0.517 (1.6×) | 0.312 (2.6×) | 0.287 (2.8×) |
| `syntomic_sweep(zpn(3, 2), 2:3, 1:24)` | 2.115 | 1.358 (1.6×) | 0.827 (2.6×) | 0.524 (4.0×) |

The speedup grows with the size of the computation: at the `:exponent` precision,
`syntomic_cohomology(zpn(2, 2), 40)` takes 1.61 s without threads and 0.53 s with 8, and
`syntomic_cohomology(zpn(2, 2), 60)` 17.6 s and 3.3 s (5.3×).

What remains sequential in a single computation is the order of the levels of normal forms and
of the relations for `g_0^p, g_1^p, …` that they need, the chain of products that computes
`w(v^i)`, and the steps of the transposed chain of Step 5; sweeps also run their pairs in
parallel. Without threads, the normal forms are computed when first needed instead (a few percent
fewer of them are needed).

If a parallel piece of work fails (for instance with [`PrecisionExhausted`](@ref)), or the call is
interrupted, no new pieces are started and the running ones are finished before the exception is
rethrown. The exception is the one a serial computation would raise first. The exceptions of the
package's interface keep their type (`PrecisionExhausted`, `CertificateError`, `ArgumentError`,
`DimensionMismatch` and `InterruptException`); any other error, such as an internal error, is
rethrown wrapped in a `CapturedException` or `TaskFailedException` that keeps the backtrace of
the task in which it occurred.

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
and throws an `ArgumentError` if the estimate exceeds the available memory
([`Syntomic.available_memory`](@ref): on Linux the memory that is free or reclaimable, on macOS the
share of memory that the system reports as available). A sweep over several rings checks the
estimates of all its groups before computing any of them. Pass `check_memory = false` to skip the
check. With Distributed workers, each host must hold the pairs (or groups) running on its
workers.

The estimate counts the coefficients of the tables that are built, with the memory
`B = coefficient_bytes(p, N)` of one coefficient modulo `p^N` (8 bytes below `2^62`, otherwise a
GMP integer of `48 + 8⌈N log₂ p / 64⌉` bytes):

| part | model (bytes) |
|:-----|:--------------|
| exponents (Phase 0, policy `:exponent`) | the f-side estimate below at the first precision of `exponent_phase`, for the pairs running at the same time |
| envelope of `O_K` (Phases A, B) | `64 MiB + f · K_p(L) · L² · B`, where `K_p(L)` counts the tabulated normal forms of the g-envelope ([`Syntomic.g_table_entries`](@ref)) and `L = n_max · i_max` |
| `∇_{O_K}` for all weights | `|is| · f · (L-1)² · (B + 40)` |
| one pair `(n, i)`, `L = n·i` | `max(2 · f_c · L³ · B, 0.6 · f_c · (c · B + 8 · e · L)) + 0.3 · ((fL)³ + 40 · (f(L-1))²) · B`, with the numbers `e`, `c` of normal forms and coefficients of the f-envelope ([`Syntomic.f_table_entries`](@ref)) |

Here `f_c` is the residue degree of the ring used for the computation (1 with descent) and `f`
the one of the target ring. The term `2 · f_c · L³ · B` is the memory allocated between two
garbage collections while the `L` columns of the matrices of a pair are built; the table of the
f-envelope dominates for `n = 2` and large `i`, where the number of monomials grows fastest; the
last term covers the `Z_p`-matrices of the square and of the Smith normal forms. The
constants were calibrated against the peak resident memory of the parts of sweeps and of whole
sweeps (`benchmark/memory.jl`). The sweep needs the larger of Phase 0 together with Phases A and B
(which run at the same time), and `∇_{O_K}` plus the pairs that run at the same time (the largest first, one per thread; this sum
is multiplied by 2 when several pairs run at the same time, since the heap grows further
between garbage collections when more memory is live).

Typical sizes, with 8 threads: the measured increase of the peak resident memory of the process
(after loading the package, about 0.6 GiB) and the estimate. In all calibration runs the estimate
was above the measured peak: by a factor between 1.6 and 3.6 for whole sweeps, and between 1.5
and 4 for single pairs.

| sweep | measured peak | estimate |
|:------|--------------:|---------:|
| `Z/2^n`, `n ∈ 2:5`, `i ∈ 1:12` | 0.17 GiB | 0.37 GiB |
| `Z/3^n`, `n ∈ 2:4`, `i ∈ 1:15` | 0.13 GiB | 0.43 GiB |
| `W_n(F_8)`, `n ∈ 2:3`, `i ∈ 1:10` | 0.05 GiB | 0.16 GiB |
| `W_n(F_{3^f})`, `f ∈ 1:3`, `n ∈ 2:3`, `i ∈ 1:12` | 0.11 GiB | 0.33 GiB |
| `Z/2^n`, `n ∈ 2:3`, `i ∈ 1:30` | 0.73 GiB | 1.90 GiB |
| `Z/4`, `i ∈ 1:40` | 0.79 GiB | 1.32 GiB |
| `Z/4`, `i ∈ 1:50` | 1.84 GiB | 3.74 GiB |

The envelope of `O_K` alone takes 0.25 GiB at `p = 2`, `L = 100`, `N = 130`; for `n = 2` and
large `i` the tables of the envelopes of `O_K/ϖ^2` of the pairs running at the same time
dominate.

```@docs
Syntomic.sweep_memory_estimate
Syntomic.check_sweep_memory
Syntomic.check_worker_memory
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
syntomic_sweep([zpn(2, 2), zpn(3, 2), zpn(5, 2), zpn(7, 2)], 2:3, 1:12; distributed=true)   # 4 groups, one per worker
```

[`add_syntomic_workers`](@ref) refuses to start workers whose threads, together with those of the
workers already running on this host, exceed the CPU threads (`Sys.CPU_THREADS`), unless
`oversubscribe = true` (the threads of the calling process are not counted). A sweep over several groups of rings sends one
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
