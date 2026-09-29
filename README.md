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
- is parallel (threads within a computation, `Distributed` across computations) and fast (see
  [Performance](#performance));
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

See [Performance](#performance) for timings against the SageMath code.

## Performance

Syntomic.jl against the SageMath code of [AKN] ([antieau/kzpn](https://github.com/antieau/kzpn) at
commit `ed381b4`), on one machine: Apple M2 Max (8 performance and 4 efficiency cores, 64 GB),
macOS 26.6, Julia 1.12.7 with Nemo 0.56.1 (FLINT 3.6.0), SageMath 10.3.

### The same computation

Both programs compute `H^1` and `H^2` of `Z_p(i)(Z/p^n)`, i.e. `K_{2i-1}` and `K_{2i-2}` of
`Z/p^n`, and certify the result. Syntomic.jl runs at the working precision `N` (p-adic digits)
that the SageMath code uses, so both carry out the same certified computation, and their groups
agree in every case that both finish. The last column is Syntomic.jl at its default precision
policy (`:adaptive`, also certified), which needs fewer digits.

| ring | weight i | N | SageMath | Syntomic.jl, 1 thread | Syntomic.jl, 8 threads | Syntomic.jl, 8 threads, default precision |
|:--|--:|--:|--:|--:|--:|--:|
| `Z/2^2` | 5 | 52 | 0.30 s | 0.61 ms (×493) | 0.93 ms (×321) | 0.86 ms |
| `Z/2^2` | 10 | 193 | 6.42 s | 5.2 ms (×1,225) | 6.6 ms (×966) | 3.2 ms |
| `Z/2^2` | 15 | 427 | 1.1 min | 40.9 ms (×1,676) | 21.6 ms (×3,177) | 8.1 ms |
| `Z/2^2` | 20 | 764 | 27.0 min | 278.5 ms (×5,812) | 83.8 ms (×19,313) | 21.8 ms |
| `Z/2^2` | 25 | 1196 | > 30 min | 1.20 s (> ×1,498) | 277.1 ms (> ×6,495) | 39.2 ms |
| `Z/2^2` | 30 | 1722 | not run¹ | 4.57 s | 873.7 ms | 61.4 ms |
| `Z/2^3` | 4 | 58 | 0.35 s | 0.86 ms (×413) | 1.2 ms (×284) | 1.2 ms |
| `Z/2^3` | 6 | 122 | 3.85 s | 2.9 ms (×1,326) | 3.6 ms (×1,077) | 2.5 ms |
| `Z/2^3` | 8 | 199 | 9.34 s | 8.1 ms (×1,154) | 9.6 ms (×971) | 4.5 ms |
| `Z/2^4` | 6 | 165 | 9.20 s | 6.8 ms (×1,359) | 8.4 ms (×1,093) | 4.3 ms |
| `Z/3^3` | 5 | 67 | 0.59 s | 1.3 ms (×470) | 1.8 ms (×327) | 1.3 ms |
| `Z/3^3` | 7 | 121 | 1.86 s | 3.7 ms (×506) | 4.7 ms (×393) | 2.8 ms |
| `Z/3^3` | 9 | 190 | 4.33 s | 10.6 ms (×409) | 8.5 ms (×507) | 5.4 ms |
| `Z/3^3` | 11 | 284 | 2.7 min | 55.5 ms (×2,909) | 24.4 ms (×6,611) | 9.7 ms |
| `Z/3^3` | 13 | 392 | 6.2 min | 141.3 ms (×2,651) | 46.6 ms (×8,038) | 15.3 ms |
| `Z/3^3` | 15 | 515 | 16.4 min | 346.6 ms (×2,836) | 93.0 ms (×10,567) | 28.3 ms |
| `Z/3^3` | 17 | 656 | > 30 min | 857.4 ms (> ×2,099) | 193.9 ms (> ×9,284) | 41.7 ms |
| `Z/3^4` | 8 | 216 | 1.9 min | 27.2 ms (×4,222) | 15.9 ms (×7,209) | 8.6 ms |
| `Z/5^3` | 8 | 133 | 1.43 s | 2.9 ms (×493) | 3.4 ms (×417) | 2.3 ms |
| `Z/7^3` | 9 | 156 | 10.94 s | 5.2 ms (×2,093) | 4.3 ms (×2,518) | 3.5 ms |

¹ Not run: the SageMath code had already exceeded 30 minutes at weight 25.

At the same precision, Syntomic.jl is 400 to 5,800 times faster on one thread and up to 19,000
times faster with 8 threads, and the factor grows with the size of the computation: at weight 20,
`Z/4` takes 27 minutes in the SageMath code and 84 ms in Syntomic.jl (22 ms at its default
precision). Below about 10 ms, 8 threads are slower than one, because starting the tasks costs more
than they save.

The SageMath times are those of `syntomic_matrices` and `syntomic_cohomology` (without starting
Sage), one fresh process per run and one run at a time, the minimum of 3 runs for runs under a
minute; runs were stopped after 30 minutes. The Syntomic.jl times are minima of repeated runs, in
one fresh process with 1 thread and one with 8, after a first run of each case that compiles the
code for it (the first computation in a new Julia session also spends about 7 s compiling).

### Beyond the SageMath code

[AKN, Appendix A] tabulates `K_r(Z/p^n; Z_p)` for `p ≤ 7`: 1,293 groups, of which 494 are marked
as not computed within 48 hours. Syntomic.jl computes all 1,293 in 9.4 minutes on 8 threads (each
ring in a fresh process, with `k_theory_table`) and agrees with all 799 groups that the paper lists:

| ring | degrees r | groups in [AKN] | marked not computed within 48 h | Syntomic.jl, 8 threads | peak memory |
|:--|--:|--:|--:|--:|--:|
| `Z/2^2` | 1–59 | 59 | — | 2.34 s | 1.0 GiB |
| `Z/2^3` | 1–59 | 41 | 18 | 3.84 s | 1.3 GiB |
| `Z/2^4` | 1–59 | 31 | 28 | 8.61 s | 2.3 GiB |
| `Z/2^5` | 1–59 | 23 | 36 | 19.48 s | 3.7 GiB |
| `Z/3^2` | 1–79 | 79 | — | 2.65 s | 1.1 GiB |
| `Z/3^3` | 1–79 | 53 | 26 | 8.01 s | 1.9 GiB |
| `Z/3^4` | 1–79 | 39 | 40 | 24.42 s | 3.5 GiB |
| `Z/3^5` | 1–79 | 31 | 48 | 1.2 min | 6.5 GiB |
| `Z/5^2` | 1–87 | 85 | 2 | 3.07 s | 1.3 GiB |
| `Z/5^3` | 1–87 | 65 | 22 | 8.42 s | 2.0 GiB |
| `Z/5^4` | 1–87 | 49 | 38 | 27.16 s | 3.3 GiB |
| `Z/5^5` | 1–87 | 41 | 46 | 1.3 min | 6.9 GiB |
| `Z/5^6`² | 1–87 | 37 | 50 | 4.1 min | 13.4 GiB |
| `Z/7^2` | 1–51 | 51 | — | 1.99 s | 0.9 GiB |
| `Z/7^3` | 1–51 | 35 | 16 | 2.38 s | 1.1 GiB |
| `Z/7^4` | 1–51 | 27 | 24 | 3.74 s | 1.4 GiB |
| `Z/7^5` | 1–51 | 21 | 30 | 8.03 s | 1.8 GiB |
| `Z/7^6` | 1–51 | 17 | 34 | 15.13 s | 2.7 GiB |
| `Z/7^7` | 1–51 | 15 | 36 | 26.52 s | 3.7 GiB |

² The memory model of sweeps, which is deliberately conservative, estimates 100 GiB for `Z/5^6`,
more than the machine has, so this ring was run with `check_memory = false`.

Larger ranges are within reach too: `K_n(Z/4)` for all `n ≤ 200` (weights up to 101) takes
2 minutes on 8 threads, with a peak memory of 10.5 GiB.

### Reproducing

```sh
git clone https://github.com/antieau/kzpn /path/to/kzpn && git -C /path/to/kzpn checkout ed381b4
KZPN_DIR=/path/to/kzpn sage -python scripts/sage/time_kzpn.py --timeout 1800 --output kzpn_times.json
julia --project=benchmark benchmark/vs_sage.jl sage kzpn_times.json    # the first table
julia --project=benchmark benchmark/vs_sage.jl appendix                 # the second table
```

The SageMath timings above (about 2 hours of computation) are in `benchmark/kzpn_times.json`, so
the first table can be recomputed without Sage.

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
