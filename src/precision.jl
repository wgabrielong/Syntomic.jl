# p-adic precision: target precision and precision losses.
#
# [AKN, §7.1, Step 1 ("find the working p-adic precision"), p. 54], and [AKN, §7.2]
# (pp. 57–60). All functions here are pure functions of the integers (p, n, i); they do not
# depend on the residue degree f (the relevant cokernels are W(F_q)-modules, so exponents are
# bounded by the same expressions with p in place of q; [AKN, p. 57]).

"""
    padic_valuation(p, x) -> Int

The p-adic valuation of a nonzero integer `x`.
"""
function padic_valuation(p::Integer, x::Integer)
    p >= 2 || throw(ArgumentError("padic_valuation: p must be at least 2, got $p"))
    x == 0 && throw(ArgumentError("padic_valuation: the valuation of 0 is infinite"))
    v = 0
    y = x
    while y % p == 0
        v += 1
        y = div(y, p)
    end
    return v
end

"""
    braces(j, n) -> Int

`{j, n}` in the notation of [AKN, Prop. 2.18]: `n` if `n | j`, and `j` otherwise.
"""
braces(j::Integer, n::Integer) = j % n == 0 ? n : j

"""
    epsilon(i, j, n) -> Int

`ε(i, j)` of [AKN, Prop. 2.18, p. 10]: `1` if `n ∤ j` and `⌈j/n⌉ ≤ i`, and `0` otherwise.
In particular `Σ_{j=1}^{in-1} ε(i, j) = i(n-1)`.
"""
epsilon(i::Integer, j::Integer, n::Integer) = (j % n != 0 && cld(j, n) <= i) ? 1 : 0

"""
    target_precision(p, n, i) -> Int

The target p-adic precision `T = Σ_{j=1}^{in-1} ε(i,j) + v_p{j,n}` of [AKN, Lemma 7.2]
([AKN, §7.1, Step 1]): an upper bound for the p-exponents of the cokernels of `N^{≥i}∇`, hence of
`H¹` and `H²` of `Z_p(i)(O_K/ϖ^n)`.
"""
function target_precision(p::Integer, n::Integer, i::Integer)
    return sum(
        (epsilon(i, j, n) + padic_valuation(p, braces(j, n)) for j in 1:(n * i - 1)); init=0
    )
end

"""
    h2_exponent_bound(p, n, i) -> Int

`Σ_{j=1}^{in-1} v_p{j,n}`: a bound for the p-exponent of the cokernel of `∇`, hence of
`H²(Z_p(i)(O_K/ϖ^n))` ([AKN, §7.2.1, p. 57]).
"""
function h2_exponent_bound(p::Integer, n::Integer, i::Integer)
    return sum((padic_valuation(p, braces(j, n)) for j in 1:(n * i - 1)); init=0)
end

"""
    loss_delta(p, n, i) -> Int

Precision lost to the divisions by `p` in `δ(x) = (φ(x) - x^p)/p` during [AKN, §7.1, Step 3]:
`M_g + 1 = ⌊log_p(in-1)⌋ + 1` ([AKN, Lemma 7.5(c), p. 59]).
"""
loss_delta(p::Integer, n::Integer, i::Integer) = num_g_variables(p, n * i)

"""
    loss_quotient(p, n, i) -> Int

`Σ_{s=p}^{i-1} n·v_p(s!)`: the p-adic valuation of `det(red)`, bounding the precision lost when
inverting `red` in [AKN, §7.1, Step 7] ([AKN, Lemma 7.6, p. 59]). It is zero for `i ≤ p`
([AKN, Remark 7.7]).
"""
function loss_quotient(p::Integer, n::Integer, i::Integer)
    return sum((n * padic_valuation(p, factorial(big(s))) for s in p:(i - 1)); init=0)
end

"""
    loss_nygaard(n, i) -> Int

`n·C(i, 2)`: the p-adic valuation of `det(can)` on the `∇`-terms, bounding the precision lost
when inverting it in [AKN, §7.1, Step 8] ([AKN, Lemma 7.8, p. 60]).
"""
loss_nygaard(n::Integer, i::Integer) = n * binomial(i, 2)

"""
    apriori_working_precision(p, n, i) -> Int

The working precision of policy `:apriori`:

    2T + 1 + (M_g + 1) + Σ_{s=p}^{i-1} n·v_p(s!) + n·C(i, 2),

where `T` is the target precision ([AKN, Lemma 7.2]). The terms are the precision losses of
[AKN, Lemma 7.5(c)] (δ), [AKN, Lemma 7.6] (passage from `O_K` to `O_K/ϖ^n`), [AKN, Lemma 7.8] (Nygaard lift of `∇`)
and [AKN, Lemma 7.9] (`can - φ^∇`), cf. [AKN, §7.1, Step 1, p. 54]. The final `+1` makes
the syntomic differentials known to precision at least `T + 1`, so that every exponent of the
cohomology is strictly below the precision, as required by [AKN, Lemma 7.1] and [AKN, Remark 7.3].
"""
function apriori_working_precision(p::Integer, n::Integer, i::Integer)
    T = target_precision(p, n, i)
    return 2 * T + 1 + loss_delta(p, n, i) + loss_quotient(p, n, i) + loss_nygaard(n, i)
end

"""
    exponent_working_precision(p, n, i, e_red, e_can1) -> Int

The working precision of policy `:exponent`: `2T + 1 + (M_g + 1) + e_red + e_can1`, where
`e_red` and `e_can1` are the exact p-exponents (largest elementary-divisor valuations) of `red`
and of `can` on the `∇`-terms, computed in a cheap preliminary pass. The proofs of
[AKN, Lemmas 7.6 and 7.8] bound the loss by these exponents; the determinant valuations used by
[`apriori_working_precision`](@ref) are often several times larger.
"""
function exponent_working_precision(
    p::Integer, n::Integer, i::Integer, e_red::Integer, e_can1::Integer
)
    T = target_precision(p, n, i)
    return 2 * T + 1 + loss_delta(p, n, i) + e_red + e_can1
end

"""
    adaptive_working_precision(p, n, i, Tguess, e_red, e_can1) -> Int

The working precision of policy `:adaptive` ([AKN, Remark 7.3]) for a guess `Tguess ≤ T` of the target
precision: [`exponent_working_precision`](@ref) with `Tguess` in place of `T`.
"""
function adaptive_working_precision(
    p::Integer, n::Integer, i::Integer, Tguess::Integer, e_red::Integer, e_can1::Integer
)
    return 2 * Tguess + 1 + loss_delta(p, n, i) + e_red + e_can1
end

"""
    initial_adaptive_guess(p, n, i) -> Int

The first guess `⌈T/4⌉` (at least 1) for the target precision under `:adaptive`.
"""
initial_adaptive_guess(p::Integer, n::Integer, i::Integer) =
    max(1, cld(target_precision(p, n, i), 4))

# ---------------------------------------------------------------------------------------------
# Determinant valuations of the W(F_q)-matrices (for invariant checks and Phase 0).
# Each is the p-adic valuation of the W-level determinant; over Z_p multiply by f.
# ---------------------------------------------------------------------------------------------

"""
    det_valuation_can0(n, i) -> Int

`Σ_{j=1}^{in-1} (i - ⌊j/n⌋)`: the valuation of `det(can)` on `F^{[1,in-1]}N^{≥i}` ([AKN, Lemma 4.63]);
its diagonal entries are exactly `p^{i-⌊j/n⌋}`.
"""
det_valuation_can0(n::Integer, i::Integer) =
    sum((i - div(j, n) for j in 1:(n * i - 1)); init=0)

"""
    det_valuation_can1(n, i) -> Int

`Σ_{m=0}^{in-2} (i - 1 - ⌊m/n⌋) = n·C(i,2)`: the valuation of `det(can)` on the `∇`-terms
([AKN, Lemma 4.63], [AKN, Lemma 7.8]); its diagonal entries are exactly `p^{i-1-⌊m/n⌋}`.
"""
det_valuation_can1(n::Integer, i::Integer) =
    sum((i - 1 - div(m, n) for m in 0:(n * i - 2)); init=0)

"""
    det_valuation_red(p, n, i) -> Int

`Σ_{j=1}^{in-1} v_p(⌊j/n⌋!)`: the valuation of `det(red)` ([AKN, Lemma 4.64], [AKN, Lemma 7.6]).
"""
function det_valuation_red(p::Integer, n::Integer, i::Integer)
    return sum(
        (padic_valuation(p, factorial(big(div(j, n)))) for j in 1:(n * i - 1)); init=0
    )
end

"""
    det_valuation_nabla_OK(p, L) -> Int

`Σ_{j=1}^{L-1} v_p(j)`: the valuation of `det(∇)` for `O_K`, whose diagonal entries are
units times `j` ([AKN, Lemma 4.61]).
"""
function det_valuation_nabla_OK(p::Integer, L::Integer)
    return sum((padic_valuation(p, j) for j in 1:(L - 1)); init=0)
end

"""
    det_valuation_nabla_R(p, n, i) -> Int

`Σ_{j=1}^{in-1} v_p{j,n}` ([AKN, Lemma 4.62(1)]).
"""
det_valuation_nabla_R(p::Integer, n::Integer, i::Integer) = h2_exponent_bound(p, n, i)

"""
    det_valuation_nabla_N(p, n, i) -> Int

`Σ_{j=1}^{in-1} ε(i,j) + v_p{j,n} = T` ([AKN, Lemma 4.62(2)]).
"""
det_valuation_nabla_N(p::Integer, n::Integer, i::Integer) = target_precision(p, n, i)
