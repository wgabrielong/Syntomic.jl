# [Parallelism](@id parallelism)

## Threads

Start Julia with several threads, e.g. `julia -t auto` (on Apple silicon, the number of
performance cores). Within one computation the following run in parallel when `threaded = true`
(the default):

- the envelope of `O_K/ϖ^n` (Steps 6, 8, 9) and the envelope of `O_K` (Steps 3–5) are built
  concurrently;
- the columns of the matrices of Steps 5, 6, 8 and 9;
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

## Distributed

Start worker processes with [`add_syntomic_workers`](@ref) (each loads only Nemo, Hecke and
Syntomic.jl) and pass `distributed = true`:

```julia
using Distributed, Syntomic
add_syntomic_workers(4; threads_per_worker=2)        # keep workers × threads ≤ CPU threads
syntomic_sweep(zpn(2, 3), 2:4, 1:16; distributed=true)
syntomic_sweep([zpn(2, 2), zpn(3, 2), zpn(5, 2)], 2:3, 1:12; distributed=true)   # one ring per worker
```

Only plain data (the rings, the working precisions and the matrices `∇` for `O_K`) is sent to
the workers.

```@docs
Syntomic.SweepJob
Syntomic.run_sweep_job
Syntomic.shared_nabla_OK
Syntomic.sweep_precisions
Syntomic.PlainWMatrix
```
