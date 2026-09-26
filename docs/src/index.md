# Syntomic.jl

```@docs
Syntomic
```

## Quick start

```jldoctest quickstart
julia> using Syntomic

julia> H = syntomic_cohomology(zpn(2, 2), 2);   # Z_2(2)(Z/4)

julia> (H.h1, H.h2)                              # K_3(Z/4; Z_2) = Z/2^3, K_2(Z/4; Z_2) = Z/2
([3], [1])

julia> k_theory_table(zpn(2, 2), 1:2; integral=true)
K-groups of Z/2^2:
  K_0(Z/2^2) ≅ Z
  K_1(Z/2^2) ≅ Z/2
  K_2(Z/2^2) ≅ Z/2
  K_3(Z/2^2) ≅ Z/2^3 ⊕ Z/3
```

Rings `O_K/ϖ^n` with residue field `F_q`, `q = p^f`, are described by an Eisenstein polynomial over
`W(F_q)`, for instance over a q-adic field of Nemo/Hecke/Oscar:

```jldoctest quickstart
julia> using Nemo, Hecke

julia> K, a = qadic_field(2, 2; precision=30);   # W(F_4)[1/2]

julia> Kz, z = polynomial_ring(K, :z);

julia> X = finite_chain_ring(z^2 + 2a*z + 2a, 5);   # O_K/ϖ^5 for a ramified quadratic extension

julia> syntomic_cohomology(X, 1).h1              # K_1 = principal units (1 + ϖO_K)/(1 + ϖ^5 O_K)
5-element Vector{Int64}:
 1
 1
 1
 2
 3
```

## Conventions

- `n` is the power of the uniformizer and `i` the motivic weight, as in [AKN].
- `K_{2i-1}(O_K/ϖ^n; Z_p) = H^1(Z_p(i))` for `i ≥ 1` and `K_{2i-2}(O_K/ϖ^n; Z_p) = H^2(Z_p(i))`
  for `i ≥ 2` ([AKN, Cor. 2.16, p. 9]); see [`k_groups`](@ref).
- Cohomology groups are reported as sorted exponent lists: `[a_1, a_2, …]` stands for
  `⊕ Z/p^{a_j}`. [`k_theory_table`](@ref) displays the primary decomposition of each K-group,
  and [`save_results`](@ref) writes results as JSON or CSV.

## Pages

- [Input conventions](@ref input): the input forms, the presentation of `W(F_q)`, the precision
  of `E` and what the results refer to.
- [Precision](@ref precision): the precision policies and the certificate.
- [Parallelism](@ref parallelism): threads and `Distributed`.
- [Algorithm map](@ref): which function implements which step of [AKN, §7.1].
- [API](@ref api): the public interface.
- [Reproducing the fixtures](@ref fixtures): the reference data of the tests and the Sage scripts
  that generate them.
