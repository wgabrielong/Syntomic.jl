# [Input conventions](@id input)

This page describes how a ring `O_K/ϖ^n` is given to Syntomic.jl, what is stored, and how the
results refer to it.

## The ring

The input of the algorithm of [AKN, §7.1, Input, p. 53] is a finite extension `K/Q_p` with
residue field `F_q`, `q = p^f`, given by an Eisenstein polynomial `E(z) ∈ W(F_q)[z]` of degree
`e` (the ramification index), together with the power `n ≥ 1` of the uniformizer and the weight
`i ≥ 1`. The ring is

```
R = O_K/ϖ^n = W(F_q)[z]/(E(z), z^n),
```

so `zpn(p, n)` (`E = z + p`) is `Z/p^n`, but for `e > 1` the power `n` is not the p-adic length:
for `n ≤ e`, `O_K/ϖ^n = F_q[z]/z^n`. A [`FiniteChainRing`](@ref) stores `R` as exact integer
data (`p`, `f`, `n`, `e`, `h̃`, the coordinates of `E`, the precision `Eprec` and a label), so
that it can be sent to other processes and rebuilt at any p-adic precision.

## The presentation `h̃` of `W(F_q)`

The Witt vectors are presented as `W(F_q) = Z_p[x]/(h̃)` for a monic integer polynomial `h̃` of
degree `f` that is irreducible modulo `p` ([AKN, §7.1, Step 2, p. 54]); elements of `W(F_q)` are
given by their coordinates in the basis `1, x, …, x^{f-1}`.

- For `f = 1`, `h̃ = x` (so `x = 0` and `W(F_p) = Z_p`).
- For `f > 1`, the default is the Conway polynomial used by Nemo's `qadic_field(p, f)`. Input over a
  q-adic field uses the defining polynomial of that field, and the constructors taking integer
  data accept any `h̃` (keyword `htilde`).

The residue field is `F_q = F_p[x]/(h̃ mod p)`; [`residue_field`](@ref Syntomic.residue_field(::FiniteChainRing))`(R)`
returns it as a Nemo finite field whose generator is the image of `x`, with the reduction map.

## Input forms

| input | `h̃` | example |
|:------|:----|:--------|
| integer data `finite_chain_ring(p, htilde, E, n)` | given | `finite_chain_ring(2, [0, 1], [[2], [2], [1]], 3)` (`E = z^2 + 2z + 2`) |
| a polynomial over `qadic_field(p, f)` | its defining polynomial | `finite_chain_ring(z^2 + 2a*z + 2a, 5)` |
| a polynomial over `padic_field(p)` | `x` (`f = 1`) | `finite_chain_ring(y + 3, 4)` |
| an Eisenstein extension (`eisenstein_extension`) | as for its base field | `finite_chain_ring(L, 5)` |
| an integer polynomial, `E ∈ Z_p[z] ⊆ W(F_q)[z]` | Conway, or `htilde` | `finite_chain_ring(t^2 + 3, 3; p=3, f=2)` |
| coefficient lists in a basis of `W(F_q)` | Conway, or `htilde` | `finite_chain_ring([[3], [0, 1]]; p=3, f=2, n=3, basis=:teichmuller)` |
| `zpn(p, n)`, `finite_chain_ring(residue_ring(ZZ, p^n)[1])` | `x` | `Z/p^n` |
| `truncated_witt_vectors(p, f, n)` | Conway | `W_n(F_q)` |

See [`finite_chain_ring`](@ref) for the keyword arguments. All forms describing the same `E` in the
same presentation give equal rings (`==`).

## Conway and Teichmüller bases

For coefficient lists (`finite_chain_ring(coeffs; p, f, n, basis)`), the coefficient of `z^k` is
`coeffs[k+1]`, an integer or a list `[a_0, a_1, …]` standing for `Σ_j a_j y^j`, where

- `basis = :conway` (default): `y = x`, the generator of `Z_p[x]/(h̃)`. The input is exact.
- `basis = :teichmuller`: `y = ζ`, the Teichmüller lift of the image of `x` in `F_q` (the unique
  root of unity of order prime to `p`, or `0`, congruent to `x` modulo `p`). If `h̃` is the Conway
  polynomial, `ζ` is a primitive `(q-1)`-st root of unity. `ζ` is computed modulo `p^precision`
  (keyword `precision`, by default the precision `n` requires, see below), so `E` is only known to
  that precision.

## The precision contract

Since `p` is a unit times `ϖ^e` in `O_K`, `p^⌈n/e⌉ ∈ (ϖ^n)`, and `R` depends only on `E` modulo
`p^⌈n/e⌉`. Every coefficient of `E` must therefore be known to p-adic precision at least

```
max(⌈n/e⌉, 2)
```

(precision `2` is needed to see that the constant term has valuation exactly `1`); otherwise the
constructor throws an `ArgumentError`. A coefficient known to higher precision is replaced by an
integer lift (balanced, so that `-1` stays `-1`); the stored integers define an Eisenstein
polynomial presenting the same ring. The precision to which they agree with the input is recorded
as `R.Eprec` (`typemax(Int)` for exact input), and [`with_power`](@ref Syntomic.with_power)`(R, n′)`,
which changes the power `n` (as [`syntomic_sweep`](@ref) does for every `n` of a sweep), checks it
again: a ring given to low precision cannot silently be used for a larger `n`.

## The normalization `E(0) = p`

[AKN, §7.1, Input, p. 53] assumes `E(0) = p`. The input does not need to be normalized: `E` and
`E/u` with `u = E(0)/p ∈ W(F_q)^×` generate the same ideal, and the computation uses `E/u`
([`Syntomic.normalize_eisenstein`](@ref)). The stored coordinates `R.E` are those of the input.

## Descent to `Z_p`, and what the matrices refer to

If `f > 1` and the normalized polynomial `E/u` is congruent modulo `p^max(⌈n/e⌉, 2)` to a
polynomial `E′` with coefficients in `Z_p`, then `R = W(F_q) ⊗ Z_p[z]/(E′, z^n)`, and with
`descent = true` (the default) the matrices of the algorithm are computed with `f = 1` arithmetic
for `E′` and expanded to `W(F_q)` afterwards ([`Syntomic.descent_ring`](@ref)). The cohomology
does not depend on the presentation, but the matrices of [`syntomic_complex`](@ref) do: they
refer to the bases of [AKN, §7.1, Step 2, p. 54] for the Eisenstein polynomial actually used
(`E′` with descent, `E/u` otherwise), and to the `Z_p`-basis `x^d b_m` of `W(F_q)`-modules with
basis `b_1, b_2, …`, where `x^d b_m` has index `f(m-1) + d + 1`. Pass `descent = false` to use the
normalized input polynomial.

## Conventions for `n`, `i` and the results

- `n` is the power of the uniformizer and `i ≥ 1` the weight; the complex lives in F-weights
  `1, …, in - 1`, and its differentials `syn0`, `syn1` are `Z_p`-matrices of sizes
  `2f(in-1) × f(in-1)` and `f(in-1) × 2f(in-1)` ([AKN, §7.1, Return, p. 57]).
  `syn0, syn1 = syntomic_complex(R, i)` destructures a [`SyntomicComplex`](@ref).
- Cohomology is reported as sorted exponent lists: `h1 = [a_1, a_2, …]` means
  `H^1(Z_p(i)(R)) ≅ ⊕_j Z/p^{a_j}`, and `[]` is the zero group; `H^0 = 0`.
- `K_{2i-1}(R; Z_p) ≅ H^1(Z_p(i)(R))` for `i ≥ 1` and `K_{2i-2}(R; Z_p) ≅ H^2(Z_p(i)(R))` for
  `i ≥ 2` ([AKN, Cor. 2.16, p. 9]); `K_0(R; Z_p) ≅ Z_p`. The integral groups add the prime-to-p
  part `K_{2i-1}(F_q) = Z/(q^i - 1)` ([AKN, §1.1, (1), p. 2]): [`k_groups`](@ref) and
  [`k_theory_table`](@ref) with `integral = true`.
- `order(H^1)/order(H^2) = q^{i(n-1)}` ([AKN, Prop. 2.18, p. 9]); with `check = true` this is
  verified for every result.
