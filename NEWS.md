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
