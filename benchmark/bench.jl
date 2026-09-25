# Benchmarks for Syntomic.jl.
#
#   julia --project=benchmark -t 8 benchmark/bench.jl            # default cases
#   julia --project=benchmark -t 1 benchmark/bench.jl            # single-threaded
#
# Prints, for each case, the working precision and the time spent building the W-matrices
# (Steps 3–6, 8, 9), in the square completions (Steps 7, 8, 10) and in the Smith normal forms.
using Syntomic, Nemo, Hecke
using Printf

const CASES = [
    ("Z/8, i=8", () -> zpn(2, 3), 8),
    ("Z/81, i=8", () -> zpn(3, 4), 8),
    ("Z/4, i=24", () -> zpn(2, 2), 24),
    ("Z/8, i=16", () -> zpn(2, 3), 16),
    ("Z/27, i=15", () -> zpn(3, 3), 15),
    ("Z/125, i=12", () -> zpn(5, 3), 12),
    ("W_3(F_9), i=10", () -> truncated_witt_vectors(3, 2, 3), 10),
    ("Q_4(√(-2a)), n=4, i=8", () -> begin
        K, a = qadic_field(2, 2; precision=10)
        Kz, z = polynomial_ring(K, :z)
        finite_chain_ring(z^2 + 2a * z + 2a, 4)
    end, 8),
]

function run(policy::Symbol)
    @printf(
        "%-24s %-9s %6s %9s %9s %9s %9s   %s\n",
        "case",
        "policy",
        "N",
        "wmatrix",
        "square",
        "snf",
        "total",
        "H^1 / H^2"
    )
    for (name, mk, i) in CASES
        X = mk()
        syntomic_cohomology(X, 1)                         # warm up
        t = @elapsed H = syntomic_cohomology(X, i; precision_policy=policy)
        tm = H.timings
        @printf(
            "%-24s %-9s %6d %9.3f %9.3f %9.3f %9.3f   %s / %s\n",
            name,
            policy,
            H.N,
            get(tm, :wmatrices, NaN),
            get(tm, :square, NaN),
            get(tm, :snf, NaN),
            t,
            H.h1,
            H.h2
        )
    end
end

println("threads: ", Threads.nthreads())
for policy in (:apriori, :exponent, :adaptive)
    run(policy)
end
