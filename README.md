# Syntomic.jl

[![CI](https://github.com/wgabrielong/Syntomic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/wgabrielong/Syntomic.jl/actions/workflows/CI.yml)

Syntomic cohomology `Z_p(i)(O_K/ϖ^n)` and p-adic algebraic K-theory of the finite chain rings
`O_K/ϖ^n`, where `K/Q_p` is a finite extension with residue field `F_q` (`q = p^f`), built on the
[Oscar](https://www.oscar-system.org) number-theory stack ([Nemo](https://github.com/Nemocas/Nemo.jl)
and [Hecke](https://github.com/thofma/Hecke.jl)).

This package reimplements, and extends, the algorithm of [AKN, §7.1] and the authors' SageMath
implementation [antieau/kzpn](https://github.com/antieau/kzpn). Compared with that code it

- works over `W(F_q)` for every residue field `F_q`, not only over `W(F_p)`;
- is split into small functions, each documented with the step of [AKN, §7.1] it implements
  (see [`docs/src/algorithm_map.md`](docs/src/algorithm_map.md); build the documentation with
  `julia --project=docs docs/make.jl`, which also has pages on the
  [input conventions](docs/src/input.md) and on [reproducing the fixtures](docs/src/fixtures.md));
- is parallel (threads within a computation, `Distributed` across computations) and fast;
- certifies every result ([AKN, Lemma 7.1, Remark 7.3]) and offers sharper precision policies.

## Usage

```julia
using Syntomic

H = syntomic_cohomology(zpn(3, 4), 8)       # Z_3(8)(Z/81)
H.h1, H.h2                                  # ([1, 1, 1, 2, 19], [])  i.e. K_15(Z/81; Z_3)
k_groups(zpn(2, 2), 2; integral=true)       # K_3(Z/4) = Z/24, K_2(Z/4) = Z/2
syn0, syn1 = syntomic_complex(zpn(2, 2), 2) # the two differentials, as Z_p-matrices
```

`k_theory_table` displays the primary decomposition of each K-group:

```julia
julia> k_theory_table(zpn(2, 2), 1:3; integral=true)
K-groups of Z/2^2:
  K_0(Z/2^2) ≅ Z
  K_1(Z/2^2) ≅ Z/2
  K_2(Z/2^2) ≅ Z/2
  K_3(Z/2^2) ≅ Z/2^3 ⊕ Z/3
  K_4(Z/2^2) ≅ 0
  K_5(Z/2^2) ≅ Z/2^3 ⊕ Z/7
```

Other residue fields and ramified extensions are given by an Eisenstein polynomial:

```julia
using Nemo, Hecke        # or: using Oscar
K, a = qadic_field(2, 2; precision=30)                  # the unramified extension Q_4
Kz, z = polynomial_ring(K, :z)
X = finite_chain_ring(z^2 + 2a*z + 2a, 5)                # O_L/ϖ^5 for L = Q_4(ϖ), ϖ² + 2aϖ + 2a = 0
syntomic_cohomology(X, 3)

truncated_witt_vectors(3, 2, 4)                          # W_4(F_9)
finite_chain_ring([[3], [0, 1]]; p=3, f=2, n=3, basis=:teichmuller)   # E = ζ z + 3, ζ a root of unity
residue_field(X)                                         # (F_4, reduction map)
```

The input forms, the presentation `W(F_q) = Z_p[x]/(h̃)`, the precision to which `E` must be known
and what the matrices of `syntomic_complex` refer to are described in
[`docs/src/input.md`](docs/src/input.md).

Sweeps over `(n, i)` share the expensive `O_K`-side data, and can run on Distributed workers:

```julia
results = syntomic_sweep(zpn(2, 2), 2:5, 1:12)           # 48 pairs (n, i), ~5 s on 8 threads
save_results("z4.csv", results)                          # or .json; also for k_theory_table
k_theory_table(zpn(3, 3), 1:10; integral=true)           # K_0, …, K_19 of Z/27

using Distributed
add_syntomic_workers(4; threads_per_worker=2)
syntomic_sweep([zpn(2, 2), zpn(3, 2)], 2:4, 1:16; distributed=true)
```

Every result is certified ([AKN, Lemma 7.1, Remark 7.3]); the precision policy
(`precision_policy = :adaptive` (default), `:exponent` or `:apriori`) only affects the speed.
Start Julia with several threads (`julia -t auto`) to use them.

Timings on an Apple M2 Max, `julia -t 8`:

| ring, weight | p-adic K-group | Syntomic.jl | SageMath code |
|:-------------|:---------|------------:|--------------:|
| `Z/8`, `i = 8` | `K_15(Z/8; Z_2) = (Z/2)^⊕2 ⊕ Z/2^6 ⊕ Z/2^8` | 0.005 s | 9.9 s |
| `Z/81`, `i = 8` | `K_15(Z/81; Z_3) = (Z/3)^⊕3 ⊕ Z/3^2 ⊕ Z/3^19` | 0.008 s | 116 s |
| `Z/4`, `i = 30` | `K_59(Z/4; Z_2) = Z/2 ⊕ (Z/2^3)^⊕3 ⊕ (Z/2^4)^⊕2 ⊕ Z/2^5 ⊕ Z/2^7` | 0.08 s | — |

At the working precision of the SageMath runs (so that both certify the same result), the ten
slowest cases of the Sage fixtures run 340 to 7200 times faster. With 8 threads, single
computations run 1.8 to 2.4 times faster than on one thread (about 6 times for the largest ones),
and sweeps 2.5 to 4 times faster
(`julia --project=benchmark benchmark/report.jl`; see the Parallelism page of the documentation).

## Conventions

- `n` is the power of the uniformizer and `i` the motivic weight, as in [AKN].
- `K_{2i-1}(O_K/ϖ^n; Z_p) = H^1(Z_p(i))` for `i ≥ 1` and `K_{2i-2}(O_K/ϖ^n; Z_p) = H^2(Z_p(i))`
  for `i ≥ 2` ([AKN, Cor. 2.16]).
- Cohomology is reported as sorted exponent lists: `[a_1, a_2, …]` means `⊕ Z/p^{a_j}`.

## Testing

```sh
julia --project -t 8 -e 'using Pkg; Pkg.test()'                              # tier 1
SYNTOMIC_LONG_TESTS=1 julia --project -t 8 -e 'using Pkg; Pkg.test()'        # + tier 2 (long)
SYNTOMIC_DISTRIBUTED_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'      # + tier 3 (Distributed)
```

The tests compare against the tables of [AKN, Appendix A and Fig. 1], against the original
SageMath code (`test/fixtures/sage_f1.json`) and a version of it extended to `W(F_q)`
(`test/fixtures/sage_fq.json`), and against independent oracles (`test/oracles`).

The Sage fixtures are generated with SageMath 10.3 from a clean checkout of
[antieau/kzpn](https://github.com/antieau/kzpn) at commit `ed381b4bf8df`; their provenance
records that commit and the SHA-256 of the upstream files. To regenerate them:

```sh
git clone https://github.com/antieau/kzpn /path/to/kzpn
git -C /path/to/kzpn checkout ed381b4bf8df9522a12829ee0cbc9a2374949231
export KZPN_DIR=/path/to/kzpn
sage -python scripts/sage/make_fixtures_f1.py --date YYYY-MM-DD   # test/fixtures/sage_f1.json
sage -python scripts/sage/make_fixtures_fq.py --date YYYY-MM-DD   # test/fixtures/sage_fq.json
```

The scripts patch the upstream sources in memory only and never write to `KZPN_DIR`; see
[`docs/src/fixtures.md`](docs/src/fixtures.md) for what each fixture contains and which tests use
it.

## References

- [AKN] B. Antieau, A. Krause, T. Nikolaus, *On the K-theory of Z/p^n*,
  [arXiv:2405.04329](https://arxiv.org/abs/2405.04329).
- B. Antieau, A. Krause, T. Nikolaus, *Prismatic cohomology relative to δ-rings*,
  [arXiv:2310.12770](https://arxiv.org/abs/2310.12770).

## License and attribution

Syntomic.jl is licensed under the GNU General Public License v3.0 (see `LICENSE`). It is a derivative
of [antieau/kzpn](https://github.com/antieau/kzpn) (GPL-3.0) by Benjamin Antieau, Achim Krause and
Thomas Nikolaus. Please cite their work (see `CITATION.cff`).
