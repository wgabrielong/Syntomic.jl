# Benchmark of the multiplication methods of `Syntomic.wpoly_mullow` (products of polynomials in z
# over W(F_q)/p^N, including the reduction modulo h̃): schoolbook and Karatsuba in x against
# Kronecker packing, for full operands of z-length `len` truncated at `len`; and, with the
# argument `kernel`, the Kronecker substitution kernel `Syntomic.kron_mullow!` for products over
# Z/p^N against FLINT's `fmpz_mod_poly_mullow`.
#
#   julia --project=benchmark -t 1 benchmark/kernels.jl            # the full grid
#   julia --project=benchmark -t 1 benchmark/kernels.jl 3 4        # only f = 3, 4
#   julia --project=benchmark -t 1 benchmark/kernels.jl kernel     # the kernel grid
#   julia --project=benchmark -t 1 benchmark/kernels.jl low        # its low-half products
#
# The environment variables KERNEL_PS, KERNEL_BITS and KERNEL_LENS (comma-separated) replace the
# default primes, sizes of p^N in bits and lengths; for the kernel grid, KERNEL_PS, KERNEL_LIMBS
# and KERNEL_LENS, and KERNEL_ROUNDS the number of interleaved rounds (3).
#
# Prints, for each (p, f, bits of p^N, len), the time per product of each method (μs, minimum
# over repetitions) and the ratio Kronecker / (best of schoolbook and Karatsuba). The rule
# `Syntomic.KRONECKER_RULE` is based on this table and on the products of actual computations
# (see its docstring). The kernel grid prints, for each (p, limbs of p^N, len), the time per
# product of FLINT and of the kernel (two full operands of length `len` truncated at `len`, and
# the square of one) and the ratio FLINT / kernel, on which `Syntomic.KRON_KERNEL_RULE` is based;
# the low-half grid prints, for truncated products, the time of the kernel with the whole
# product of the packed operands and with its low half, on which `Syntomic.KRON_LOW_RULE` is
# based.
using Syntomic, Nemo, Random
using Printf

const S = Syntomic

# A Galois ring W(F_{p^f})/p^N with a Conway polynomial as h̃.
function galois_ring(p, f, N)
    K, _ = qadic_field(p, f; precision=1, cached=false)
    h = [lift(ZZ, c) for c in coefficients(defining_polynomial(K))]
    return S.GaloisRing(p, f == 1 ? [0, 1] : h, N)
end

rand_poly(rng, R, len) = R.Znz([R.Zn(rand(rng, 0:(R.pN - 1))) for _ in 1:len])

# Minimum over `reps` batches of the time per call (μs), with batches of about 10 ms.
function timeit(g; reps=5)
    g()
    t1 = @elapsed g()
    k = max(1, round(Int, 0.01 / max(t1, 1e-7)))
    best = Inf
    for _ in 1:reps
        t = @elapsed for _ in 1:k
            g()
        end
        best = min(best, t / k)
    end
    return 1e6 * best
end

function run_grid(;
    ps=(2, 3), fs=2:6, bitss=(32, 64, 128, 256, 512), lens=(4, 8, 16, 32, 64, 128), reps=5
)
    rng = Xoshiro(1)
    @printf(
        "%3s %3s %6s %5s %12s %12s %12s %8s\n",
        "p",
        "f",
        "bits",
        "len",
        "schoolbook",
        "karatsuba",
        "kronecker",
        "kron/best"
    )
    for p in ps, f in fs, bits in bitss
        N = ceil(Int, bits / log2(p))
        R = galois_ring(p, f, N)
        for len in lens
            A = [rand_poly(rng, R, len) for _ in 1:f]
            B = [rand_poly(rng, R, len) for _ in 1:f]
            ws = S.WScratch(R)
            P = [zero(R.Znz) for _ in 1:(2 * f - 1)]
            tm(m) = timeit(() -> S.wpoly_mul!(R, P, A, B, len, ws; method=m); reps=reps)
            # interleave the methods to reduce the effect of load changes
            ts, tk, tkr = Inf, Inf, Inf
            for _ in 1:2
                ts = min(ts, tm(:schoolbook))
                f <= 4 && (tk = min(tk, tm(:karatsuba)))
                tkr = min(tkr, tm(:kronecker))
            end
            best = min(ts, tk)
            @printf(
                "%3d %3d %6d %5d %12.2f %12.2f %12.2f %8.2f\n",
                p,
                f,
                bits,
                len,
                ts,
                tk,
                tkr,
                tkr / best
            )
        end
    end
end

# The Kronecker substitution kernel against FLINT, for p^N of `limbs` limbs (the largest N with
# bits(p^N) ≤ 64·limbs - 8): FLINT's `fmpz_mod_poly_mullow` (`Syntomic.poly_mullow!` without a
# scratch space) and `Syntomic.kron_mullow!` in a reused scratch space, both into a reused output.
function run_kernel_grid(;
    ps=(2, 3),
    limbs=1:8,
    lens=(4, 6, 7, 8, 16, 24, 32, 48, 64, 96, 100, 128, 256),
    reps=5,
    rounds=3,
)
    rng = Xoshiro(1)
    @printf(
        "%3s %6s %6s %5s %10s %10s %7s %10s %10s %7s\n",
        "p",
        "limbs",
        "bits",
        "len",
        "flint mul",
        "kernel mul",
        "ratio",
        "flint sqr",
        "kernel sqr",
        "ratio"
    )
    for p in ps, K in limbs
        N = 1
        while nbits(ZZ(p)^(N + 1)) <= 64 * K - 8
            N += 1
        end
        R = S.GaloisRing(p, N)
        ws = S.WScratch(R)
        for len in lens
            x, y = rand_poly(rng, R, len), rand_poly(rng, R, len)
            z = zero(R.Znz)
            tf(a, b) = timeit(() -> S.poly_mullow!(z, a, b, len); reps=reps)
            tk(a, b) = timeit(() -> S.kron_mullow!(z, a, b, len, ws); reps=reps)
            # interleave the methods to reduce the effect of load changes
            fm, km, fs, ks = Inf, Inf, Inf, Inf
            for _ in 1:rounds
                fm = min(fm, tf(x, y))
                km = min(km, tk(x, y))
                fs = min(fs, tf(x, x))
                ks = min(ks, tk(x, x))
            end
            @printf(
                "%3d %6d %6d %5d %10.2f %10.2f %7.2f %10.2f %10.2f %7.2f\n",
                p,
                K,
                nbits(R.pN),
                len,
                fm,
                km,
                fm / km,
                fs,
                ks,
                fs / ks
            )
        end
    end
end

# The low-half product of the kernel against the whole product, for truncated products: two
# operands of length `len` truncated at `len` (the common case: `len` of the `2·len - 1`
# coefficients are needed), and the square of one; FLINT for comparison. `ml` is the number of
# limbs of the low half.
function run_low_grid(;
    ps=(2, 3),
    limbs=2:8,
    lens=(8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 200, 256),
    reps=5,
    rounds=3,
)
    rng = Xoshiro(1)
    @printf(
        "%3s %6s %5s %6s %9s %9s %9s %6s %9s %9s %6s\n",
        "p",
        "limbs",
        "len",
        "ml",
        "flint",
        "whole",
        "low",
        "ratio",
        "whole sq",
        "low sq",
        "ratio"
    )
    for p in ps, K in limbs
        N = 1
        while nbits(ZZ(p)^(N + 1)) <= 64 * K - 8
            N += 1
        end
        R = S.GaloisRing(p, N)
        kb = S._kron!(S.WScratch(R), R)
        for len in lens
            x, y = rand_poly(rng, R, len), rand_poly(rng, R, len)
            z = zero(R.Znz)
            bits = 2 * nbits(R.pN) + ndigits(len; base=2) + 1
            tk(a, b, low) = timeit(() -> S._kron_mullow!(kb, z, a, b, len, low); reps=reps)
            tf, tw, tl, tws, tls = Inf, Inf, Inf, Inf, Inf
            for _ in 1:rounds
                tf = min(tf, timeit(() -> S.poly_mullow!(z, x, y, len); reps=reps))
                tw = min(tw, tk(x, y, :never))
                tl = min(tl, tk(x, y, :always))
                tws = min(tws, tk(x, x, :never))
                tls = min(tls, tk(x, x, :always))
            end
            @printf(
                "%3d %6d %5d %6d %9.2f %9.2f %9.2f %6.2f %9.2f %9.2f %6.2f\n",
                p,
                K,
                len,
                cld(len * bits, 64),
                tf,
                tw,
                tl,
                tw / tl,
                tws,
                tls,
                tws / tls
            )
        end
    end
end

# A comma-separated list of integers from the environment variable `name`, or `default`.
env_ints(name, default) = haskey(ENV, name) ? parse.(Int, split(ENV[name], ",")) : default

println(
    "threads: ",
    Threads.nthreads(),
    "  FLINT_INPLACE: ",
    S.FLINT_INPLACE,
    "  KRONECKER_KERNEL: ",
    S.KRONECKER_KERNEL,
)
if !isempty(ARGS) && ARGS[1] == "low"
    run_low_grid(;
        ps=env_ints("KERNEL_PS", (2, 3)),
        limbs=env_ints("KERNEL_LIMBS", 2:8),
        lens=env_ints(
            "KERNEL_LENS", (8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 200, 256)
        ),
        rounds=only(env_ints("KERNEL_ROUNDS", 3)),
    )
elseif !isempty(ARGS) && ARGS[1] == "kernel"
    run_kernel_grid(;
        ps=env_ints("KERNEL_PS", (2, 3)),
        limbs=env_ints("KERNEL_LIMBS", 1:8),
        lens=env_ints("KERNEL_LENS", (4, 6, 7, 8, 16, 24, 32, 48, 64, 96, 100, 128, 256)),
        rounds=only(env_ints("KERNEL_ROUNDS", 3)),
    )
else
    run_grid(;
        fs=isempty(ARGS) ? (2:6) : parse.(Int, ARGS),
        ps=env_ints("KERNEL_PS", (2, 3)),
        bitss=env_ints("KERNEL_BITS", (32, 64, 128, 256, 512)),
        lens=env_ints("KERNEL_LENS", (4, 8, 16, 32, 64, 128)),
    )
end
