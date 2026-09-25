# Syntomic.jl

[![CI](https://github.com/wgabrielong/Syntomic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/wgabrielong/Syntomic.jl/actions/workflows/CI.yml)

Syntomic cohomology `Z_p(i)(O_K/ϖ^n)` and p-adic algebraic K-theory of the finite chain rings
`O_K/ϖ^n`, where `K/Q_p` is a finite extension with residue field `F_q` (`q = p^f`), built on the
[Oscar](https://www.oscar-system.org) number-theory stack ([Nemo](https://github.com/Nemocas/Nemo.jl)
and [Hecke](https://github.com/thofma/Hecke.jl)).

> **Status: under development.** The API below is the planned API.

This package reimplements, and extends, the algorithm of §7 of Antieau–Krause–Nikolaus,
*On the K-theory of Z/p^n* ([arXiv:2405.04329](https://arxiv.org/abs/2405.04329)), and their
SageMath implementation [antieau/kzpn](https://github.com/antieau/kzpn). Compared with that code it

- works over `W(F_q)` for any residue field `F_q`, not only `W(F_p)` (totally ramified case);
- is split into many small functions, each documented with the Step of the algorithm it implements;
- is parallel: threads within a computation, and `Distributed` workers across computations;
- shares the expensive `O_K`-side data across different `(n, i)`, and uses sharper, certified
  p-adic precision bookkeeping.

## Planned usage

```julia
using Oscar, Syntomic       # or: using Hecke, Syntomic

R = zpn(3, 4)                           # Z/81
H = syntomic_cohomology(R, 8)           # H¹, H² of Z_3(8)(Z/81)
k_groups(R, 8)                          # K_15(Z/81; Z_3), K_14(Z/81; Z_3)

K, a = qadic_field(2, 2)                # Q_4 (unramified, residue field F_4)
Kz, z = polynomial_ring(K, :z)
R = finite_chain_ring(z^2 + 2a*z + 2a, 5)   # O_K/ϖ^5 for a ramified quadratic extension of Q_4
k_groups(R, 1)
```

## Conventions

- `n` is the power of the uniformizer and `i` the motivic weight, as in the paper. (The SageMath code
  uses `k` for the power of the uniformizer.)
- `K_{2i-1}(O_K/ϖ^n; Z_p) = H¹(Z_p(i))` for `i ≥ 1` and `K_{2i-2}(O_K/ϖ^n; Z_p) = H²(Z_p(i))` for `i ≥ 2`.

## Testing

```sh
julia --project -t 8 -e 'using Pkg; Pkg.test()'                              # tier 1
SYNTOMIC_LONG_TESTS=1 julia --project -t 8 -e 'using Pkg; Pkg.test()'        # + tier 2 (long)
SYNTOMIC_DISTRIBUTED_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'      # + tier 3 (Distributed)
```

## License and attribution

Syntomic.jl is licensed under the GNU General Public License v3.0 (see `LICENSE`). It is a derivative
of [antieau/kzpn](https://github.com/antieau/kzpn) (GPL-3.0) by Benjamin Antieau, Achim Krause and
Thomas Nikolaus. Please cite their work (see `CITATION.cff`).
