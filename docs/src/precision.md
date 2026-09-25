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
| `:adaptive` (default) | as `:exponent`, with a guess `T' ≤ T` in place of `T` | [AKN, Remark 7.3, p. 58]: start with `T' = ⌈T/4⌉` and double on failure; the last attempt uses `:exponent` |
| `:exponent` | `2T + 1 + (M_g + 1) + e_red + e_can` | `e_red`, `e_can` are the exact p-exponents of `red` and of `can` on the `∇`-terms, computed first from the f-envelope alone; default of [`syntomic_complex`](@ref) and of sweeps |
| `:apriori` | `2T + 1 + (M_g + 1) + Σ_{s=p}^{i-1} n v_p(s!) + n C(i,2)` | [AKN, §7.1, Step 1, p. 54]; no preliminary computation |

The terms are the precision losses of [AKN, Lemma 7.5(c), 7.6, 7.8 and 7.9, pp. 59–60]. The
losses in Steps 7, 8 and 10 are bounded by the p-exponents of the cokernels of the matrices that
are inverted; the policy `:apriori` bounds these exponents by determinant valuations, which can
be several times larger. For example, for `K_{47}(Z/4)` (`p = 2`, `n = 2`, `i = 24`) the
policies use `N = 1101`, `150` and `80`.

```@docs
Syntomic.apriori_working_precision
Syntomic.exponent_working_precision
Syntomic.adaptive_working_precision
Syntomic.loss_delta
Syntomic.loss_quotient
Syntomic.loss_nygaard
Syntomic.epsilon
Syntomic.braces
```
