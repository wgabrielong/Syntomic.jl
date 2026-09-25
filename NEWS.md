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
  decomposition of each K-group, e.g. `K_3(Z/2^2) ≅ Z/8 ⊕ Z/3`.
- `save_results` writes cohomology results and K-theory tables as JSON or CSV (no new
  dependencies; big integers are strings in JSON).
- `residue_field(R)` returns the residue field `F_q` (generator the image of `x`) and the reduction
  map; `syn0, syn1 = syntomic_complex(R, i)` destructures the syntomic complex.
- Performance: in-place FLINT arithmetic in `W(F_q)[z]`, Karatsuba multiplication in `x` for
  `f ≤ 4` and Kronecker packing for larger `f` (chosen by a benchmarked rule); Smith normal forms
  on machine words when possible; `w(v^i)` as the fixed point of `x ↦ w(u)^i φ(x)`; `∇_{O_K}` from
  a transposed chain that does not depend on `i`; parallel triangular solves; the exponents of
  `red` and `can` computed at increasing precisions. Results are unchanged, bit for bit.
- Sweeps compute `w(v^i)` and `∇_{O_K}(i)` only modulo `F^{≥n_max·i}`, and a sweep over several
  rings computes the rings that descend to the same ring over `Z_p` (e.g. `W_n(F_q)` for several
  `q = p^f`) together.
- `syntomic_sweep` estimates its peak memory before computing and refuses sweeps that do not fit
  (`check_memory=false` skips the check); `add_syntomic_workers` refuses to oversubscribe the CPU
  threads unless `oversubscribe=true`.
- Documentation pages "Input conventions" and "Reproducing the fixtures"; the Sage fixtures are
  generated from a clean checkout of antieau/kzpn at commit `ed381b4bf8df`, recorded in their
  provenance with the SHA-256 of the upstream files.
