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
