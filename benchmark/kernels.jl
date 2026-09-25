# Benchmark of the multiplication methods of `Syntomic.wpoly_mullow` (products of polynomials in z
# over W(F_q)/p^N, including the reduction modulo h̃): schoolbook and Karatsuba in x against
# Kronecker packing, for full operands of z-length `len` truncated at `len`.
#
#   julia --project=benchmark -t 1 benchmark/kernels.jl            # the full grid
#   julia --project=benchmark -t 1 benchmark/kernels.jl 3 4        # only f = 3, 4
#
# The environment variables KERNEL_PS, KERNEL_BITS and KERNEL_LENS (comma-separated) replace the
# default primes, sizes of p^N in bits and lengths.
#
# Prints, for each (p, f, bits of p^N, len), the time per product of each method (μs, minimum
# over repetitions) and the ratio Kronecker / (best of schoolbook and Karatsuba). The rule
# `Syntomic.KRONECKER_RULE` is based on this table and on the products of actual computations
# (see its docstring).
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

# A comma-separated list of integers from the environment variable `name`, or `default`.
env_ints(name, default) = haskey(ENV, name) ? parse.(Int, split(ENV[name], ",")) : default

println("threads: ", Threads.nthreads(), "  FLINT_INPLACE: ", S.FLINT_INPLACE)
run_grid(;
    fs=isempty(ARGS) ? (2:6) : parse.(Int, ARGS),
    ps=env_ints("KERNEL_PS", (2, 3)),
    bitss=env_ints("KERNEL_BITS", (32, 64, 128, 256, 512)),
    lens=env_ints("KERNEL_LENS", (4, 8, 16, 32, 64, 128)),
)
