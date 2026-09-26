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
starts. [`Syntomic.predicted_exponents`](@ref) estimates them as

    e_can ≈ ⌈n(i-1)/e⌉ - v_p((i-2)!),     e_red ≈ ⌈n(i-2)/(p·e)⌉

(on 450 rings and weights the first was exact or one too small, the second at most 5 too large).

- Under `:adaptive`, every attempt but the last uses these estimates, so the preliminary pass that
  computes the exact exponents runs only if the last attempt, at the `:exponent` precision, is
  reached. The first guess `T' = ⌈(p-1)T/(2p)⌉ + 1` (which is `⌈T/4⌉ + 1` for `p = 2`) accounts
  for the exponent of the cokernel of `N^{≥i}∇`, which is close to `T` for large `p`; on 120
  rings and weights (`p ≤ 13`) the first attempt was always certified.
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
Syntomic.predicted_exponents
Syntomic.predicted_exponent_working_precision
Syntomic.predicted_adaptive_working_precision
Syntomic.loss_delta
Syntomic.loss_quotient
Syntomic.loss_nygaard
Syntomic.epsilon
Syntomic.braces
```
