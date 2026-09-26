# [Precision](@id precision)

All computations take place in `W(F_q)/p^N` for a working precision `N`. Every object carries the
number of p-adic digits to which it is known, and the final result is **certified**: by
[AKN, Lemma 7.1, p. 57] and [AKN, Remark 7.3, p. 58], if both syntomic differentials are known
modulo `p^M`, have full rank `f(in-1)` modulo `p^M`, and all their elementary divisors have
valuation `< M`, then the computed cohomology is correct. Otherwise a
[`PrecisionExhausted`](@ref) error is raised — the package never returns an uncertified answer.

## Precision policies

The keyword `precision_policy` of [`syntomic_cohomology`](@ref) chooses `N`. Let
`T = Σ_{j=1}^{in-1} ε(i, j) + v_p{j, n}` be the target precision of [AKN, Lemma 7.2, p. 58].

| policy | working precision | notes |
|:-------|:------------------|:------|
| `:adaptive` (default) | as `:exponent`, with a guess `T' ≤ T` in place of `T` and estimated exponents | [AKN, Remark 7.3, p. 58]: start with `T' = ⌈(p-1)T/(2p)⌉ + 1` and double on failure; the attempts use the estimates of `e_red`, `e_can` below, the last attempt uses `:exponent` |
| `:exponent` | `2T + 1 + (M_g + 1) + e_red + e_can` | `e_red`, `e_can` are the exact p-exponents of `red` and of `can` on the `∇`-terms, computed from the f-envelope alone while the envelope of `O_K` is built; default of [`syntomic_complex`](@ref) and of sweeps |
| `:apriori` | `2T + 1 + (M_g + 1) + Σ_{s=p}^{i-1} n v_p(s!) + n C(i,2)` | [AKN, §7.1, Step 1, p. 54]; no preliminary computation |

The terms are the precision losses of [AKN, Lemma 7.5(c), 7.6, 7.8 and 7.9, pp. 59–60]. The
losses in Steps 7, 8 and 10 are bounded by the p-exponents of the cokernels of the matrices that
are inverted; the policy `:apriori` bounds these exponents by determinant valuations, which can
be several times larger. For example, for `K_{47}(Z/4)` (`p = 2`, `n = 2`, `i = 24`) the
policies `:apriori` and `:exponent` use `N = 1101` and `150`, and the first attempt of `:adaptive`
uses `N = 83`.

### Estimated exponents and overlapping

The exponents `e_red` and `e_can` of the matrices of Steps 6 and 8 only determine the working
precision: the solves of Steps 7, 8 and 10 use the exponents of the matrices they actually invert,
and the final certificate does not use them. So they need not be known before the computation
starts. [`Syntomic.predicted_exponents`](@ref) estimates them by the heuristic formulas

    e_can ≈ ⌈n(i-1)/e⌉ - v_p((i-2)!),     e_red ≈ ⌈n(i-2)/(p·e)⌉,

raised to the lower bounds `e_can ≥ i - 1` and `e_red ≥ v_p((i-1)!)` that follow from the
diagonals of the (triangular) matrices ([`Syntomic.exponent_lower_bounds`](@ref)). A wrong
estimate costs time, never correctness. On 505 rings and weights (`Z/p^n` for `p ≤ 19`,
`n ≤ 8`, including `n = 1` and `Z/4` up to `i = 60`, and ramified rings with `n ≤ e` and `n > e`)
the precision computed from the estimates plus a margin of 3 digits was never below the exact
`:exponent` precision, and at most 11 digits above it (3.7 on average).

- Under `:adaptive`, every attempt but the last uses these estimates, so the preliminary pass that
  computes the exact exponents runs only if the last attempt, at the `:exponent` precision, is
  reached. The first guess `T' = ⌈(p-1)T/(2p)⌉ + 1` (which is `⌈T/4⌉ + 1` for `p = 2`) accounts
  for the exponent of the cokernel of `N^{≥i}∇`, which is close to `T` for large `p` (it must
  exceed the exponents of `H^1`, and of `H^2` plus that exponent); on the 505 rings and weights
  above the first attempt was certified in 504 cases. An attempt whose precision is at most
  4 digits above the largest `N` with `p^N < 2^62` is run at that `N` instead, where the
  arithmetic uses single-word integers.
- At the `:exponent` precision, the envelope of `O_K` — the larger part of the work — is built at
  the precision computed from the estimates while the exact exponents are computed
  (concurrently, with threads), and used if that precision is large enough (it is recomputed
  otherwise). Sweeps build their shared envelope of `O_K` in the same way. Without threads the
  same precisions are used, so the results do not depend on the number of threads.

```@docs
Syntomic.apriori_working_precision
Syntomic.exponent_working_precision
Syntomic.adaptive_working_precision
Syntomic.initial_adaptive_guess
Syntomic.exponent_lower_bounds
Syntomic.predicted_exponents
Syntomic.one_word_precision
Syntomic.predicted_exponent_working_precision
Syntomic.predicted_adaptive_working_precision
Syntomic.loss_delta
Syntomic.loss_quotient
Syntomic.loss_nygaard
Syntomic.epsilon
Syntomic.braces
```
