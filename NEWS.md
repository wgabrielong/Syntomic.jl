# Syntomic.jl release notes

## v0.1.0 (unreleased)

First version: a reimplementation of the algorithm of [AKN, §7.1] (Antieau–Krause–Nikolaus,
*On the K-theory of Z/p^n*, arXiv:2405.04329) for the syntomic cohomology `Z_p(i)(O_K/ϖ^n)`.

- Rings `O_K/ϖ^n` over `W(F_q)` for any residue field `F_q`, given by an Eisenstein polynomial
  over a q-adic or p-adic field, an Eisenstein extension, integer coefficient lists in a Conway or
  Teichmüller basis, or as `zpn(p, n)` and `truncated_witt_vectors(p, f, n)`.
- `syntomic_cohomology`, `syntomic_complex`, `k_groups`, `k_theory_table`.
- Certified results ([AKN, Lemma 7.1, Remark 7.3]) with three precision policies
  (`:adaptive` (default), `:exponent`, `:apriori`).
- The precision to which the Eisenstein polynomial is known is recorded; changing the power `n`
  (`with_power`, sweeps) checks that it suffices.
- Parallel: threads within a computation, `syntomic_sweep` over `(n, i)` sharing the `O_K`-side
  data, Distributed workers (`add_syntomic_workers`). A failure stops the remaining work and is
  rethrown with the same exception type as in a serial computation.
- Tested against [AKN, Appendix A and Fig. 1], the original SageMath code, a version of it for
  `W(F_q)`, a brute-force oracle for `K_1` and the Hesselholt–Madsen formula.
- `k_theory_table` returns a `KTheoryTable` (a vector of the same rows) that displays the primary
  decomposition of each K-group, e.g. `K_3(Z/2^2) ≅ Z/2^3 ⊕ Z/3`, with orders written as powers
  and multiplicities as `(Z/2)^⊕2`.
- `save_results` writes cohomology results and K-theory tables as JSON or CSV (no new
  dependencies; big integers are strings in JSON).
- `residue_field(R)` returns the residue field `F_q` (generator the image of `x`) and the reduction
  map; `syn0, syn1 = syntomic_complex(R, i)` destructures the syntomic complex.
- Performance: in-place FLINT arithmetic in `W(F_q)[z]`, Karatsuba multiplication in `x` for
  `f = 2, 3, 4` and Kronecker packing for `f ≥ 5` and, in a benchmarked range of precisions and
  lengths, `f = 4` (the rule `Syntomic.KRONECKER_RULE`); Smith normal forms
  on machine words when possible; `w(v^i)` as the fixed point of `x ↦ w(u)^i φ(x)`; `∇_{O_K}` from
  a transposed chain that does not depend on `i`; parallel triangular solves; the exponents of
  `red` and `can` computed at increasing precisions. Results are unchanged, bit for bit.
- The exponents of `red` and `can` (which only choose the working precision) are estimated before
  any computation (`Syntomic.predicted_exponents`). Under `:adaptive` every attempt but the last
  uses the estimates, so the exact exponents are computed only for the last attempt; the first
  guess is `⌈(p-1)T/(2p)⌉ + 1` (`⌈T/4⌉ + 1` for `p = 2`), which avoids the second attempt that
  most computations for odd `p` needed. At the `:exponent` precision the envelope of `O_K` is built
  at the estimated precision while the exact exponents are computed, in sweeps too.
- The envelope of `O_K` with threads: the normal forms of its monomials that products can need
  are computed level by level as soon as the relations they need are known, the normal forms of
  a level and the components of the expensive ones in parallel (`Syntomic.g_fill_normal_forms!`;
  before, the first task to need a shared normal form computed all the ones it depends on while
  the others waited); the transposed chain of Step 5 runs concurrently with `w(v^i)`.
- Products over `Z/p^N` with `p^N` of 2 to 8 limbs use a Kronecker substitution kernel on limb
  buffers kept in the scratch space (`Syntomic.kron_mullow!`, 1.2–2.6 times faster than FLINT at
  2–4 limbs), computing only the low half of the product when only that is needed
  (`Syntomic.KRON_KERNEL_RULE`, `Syntomic.KRON_LOW_RULE`); the sums of products in the envelope
  of `O_K` add the coefficients of these products as limbs and reduce each sum once
  (`Syntomic.WAcc`); scratch spaces are reused by the parallel loops (`Syntomic.ScratchPool`).
  Results are unchanged, bit for bit. `zpn(2, 2)`, `i = 40` at the `:exponent` precision: 1.13 s
  → 0.53 s with 8 threads, 2.49 s → 1.61 s without threads; `i = 60`: 7.2 s → 3.3 s and 21.7 s
  → 17.6 s; `zpn(3, 2)`, `i = 30`: 0.21 s → 0.11 s and 0.44 s → 0.28 s; 120 rings and weights at
  the default precision policy: 15.1 s → 9.0 s with 8 threads.
- The envelope of `O_K/ϖ^n` computes normal forms level by level: with `B_s = ∏ f_u^{e_u}` for
  `s = Σ e_u p^u`, the relation `z_0^n B_s = C_{k(s)} B_{s+1}` (`C_k = c_0 ⋯ c_{k-1}`, `k(s)` the
  number of trailing digits `p - 1` of `s`) reduces a power series times `B_s` with one truncated
  product of power series per carry (`Syntomic.f_series_times_monomial`), instead of a memo table
  of the normal forms of all monomials `z_0^a ∏ f_u^{e_u}` (69,000 of them at `L = 80`, 480,000 at
  `L = 120`). Results are unchanged, bit for bit (the normal form does not depend on the order in
  which the relations are applied). For `zpn(2, 2)` without threads, Steps 6, 8 and 9 take 0.023 s
  instead of 0.16 s at `i = 40` and 0.16 s instead of 1.57 s at `i = 60`, the exponents of `red`
  and `can` 0.003 s instead of 0.09 s and 0.03 s instead of 1.30 s. With threads, the envelope of
  `O_K` built at the same time no longer waits for the garbage collections that the table caused:
  at the `:exponent` precision, `i = 40` takes 0.37 s instead of 0.61 s with 8 threads (1.32 s
  instead of 1.57 s without threads) and `i = 60` 2.56 s instead of 3.29 s (14.9 s instead of
  17.9 s); at the default precision policy, 0.24 s instead of 0.31 s and 1.31 s instead of 1.57 s.
  Sweeps are 4–18% faster. The peak memory of sweeps with `n = 2` halves (0.85 GiB instead of
  1.84 GiB for `Z/4`, `i ∈ 1:50`), and the memory model of sweeps no longer counts that table.
- Sweeps compute `w(v^i)` and `∇_{O_K}(i)` only modulo `F^{≥n_max·i}`, and a sweep over several
  rings computes the rings that descend to the same ring over `Z_p` (e.g. `W_n(F_q)` for several
  `q = p^f`) together.
- `syntomic_sweep` estimates its peak memory before computing and refuses sweeps that do not fit
  (`check_memory=false` skips the check); `add_syntomic_workers` refuses to start workers whose
  threads, with those of the workers already running on the host, exceed the CPU threads unless
  `oversubscribe=true`.
- Documentation pages "Input conventions" and "Reproducing the fixtures"; the Sage fixtures are
  generated from a clean checkout of antieau/kzpn at commit `ed381b4bf8df`, recorded in their
  provenance with the SHA-256 of the upstream files.
