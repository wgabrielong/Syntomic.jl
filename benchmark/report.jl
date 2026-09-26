# Performance report: comparison with the SageMath implementation at equal working precision,
# and thread scaling.
#
#   julia --project=benchmark benchmark/report.jl            # both parts
#   julia --project=benchmark benchmark/report.jl sage       # Sage comparison only (-t 8)
#   julia --project=benchmark benchmark/report.jl scaling    # thread scaling only
#
# The Sage timings are the wall-clock times recorded in test/fixtures/sage_f1.json (the original
# SageMath code at the working precision stored with each case). Syntomic.jl runs the same ring and
# weight at that same precision N (so both certify the same result), threaded and serial, and with
# its default :adaptive precision policy. The thread-scaling part runs every case in a fresh Julia
# process with 1, 2, 4 and 8 threads. All figures are minima of repeated runs after a warm-up.
using Syntomic
using JSON3
using Printf

const S = Syntomic
const REPEATS = 3
const ROOT = normpath(joinpath(@__DIR__, ".."))

# --- the Sage comparison ----------------------------------------------------------------------

# The fixture cases with a recorded Sage time of at least `tmin` seconds, slowest first.
function sage_cases(tmin)
    fx = JSON3.read(read(joinpath(ROOT, "test", "fixtures", "sage_f1.json"), String))
    out = []
    for c in fx.cases
        secs = get(c, :sage_seconds, nothing)
        (secs === nothing || secs < tmin) && continue
        E = [[parse(BigInt, x) for x in coeff] for coeff in c.E]
        h = [parse(BigInt, x) for x in c.h_tilde]
        push!(out, (p=c.p, n=c.n, i=c.i, N=c.N, h=h, E=E, sage=Float64(secs)))
    end
    return sort(out; by=c -> -c.sage)
end

besttime(f) = minimum(@elapsed(f()) for _ in 1:REPEATS)

function sage_report(; tmin=0.3)
    println("Comparison with the SageMath implementation (threads: $(Threads.nthreads()))")
    @printf(
        "%-26s %5s %9s %11s %11s %13s %9s\n",
        "case (p, n, i; E)",
        "N",
        "Sage",
        "equal N",
        "equal N,",
        ":adaptive",
        "speedup"
    )
    @printf(
        "%-26s %5s %9s %11s %11s %13s %9s\n",
        "",
        "",
        "",
        "threaded",
        "serial",
        "threaded",
        ""
    )
    for c in sage_cases(tmin)
        X = finite_chain_ring(c.p, c.h, c.E, c.n)
        syntomic_cohomology(X, c.i; N=c.N)                       # warm-up
        t = besttime(() -> syntomic_cohomology(X, c.i; N=c.N))
        ts = besttime(() -> syntomic_cohomology(X, c.i; N=c.N, threaded=false))
        ta = besttime(() -> syntomic_cohomology(X, c.i))
        label = "($(c.p), $(c.n), $(c.i)); deg E = $(length(c.E) - 1)"
        @printf(
            "%-26s %5d %8.2fs %10.3fs %10.3fs %12.3fs %8.0fx\n",
            label,
            c.N,
            c.sage,
            t,
            ts,
            ta,
            c.sage / t
        )
    end
end

# --- thread scaling ---------------------------------------------------------------------------

const SCALING_CASES = [
    ("K_59(Z/4): zpn(2,2), i=30", () -> syntomic_cohomology(zpn(2, 2), 30)),
    ("zpn(2,2), i=40", () -> syntomic_cohomology(zpn(2, 2), 40)),
    ("zpn(3,4), i=8", () -> syntomic_cohomology(zpn(3, 4), 8)),
    (
        "z²+2az+2a over Q_4, n=4, i=12",
        () -> syntomic_cohomology(
            finite_chain_ring([[0, 2], [0, 2], [1]]; p=2, f=2, n=4), 12
        ),
    ),
    ("sweep zpn(2,2), 2:5 × 1:12", () -> syntomic_sweep(zpn(2, 2), 2:5, 1:12)),
    ("sweep zpn(3,2), 2:3 × 1:24", () -> syntomic_sweep(zpn(3, 2), 2:3, 1:24)),
]

# In a subprocess: the minimum time of every scaling case, as a Julia vector literal.
function scaling_worker()
    syntomic_cohomology(zpn(2, 2), 6)                             # warm-up
    for (_, f) in SCALING_CASES
        f()
    end
    return println([besttime(f) for (_, f) in SCALING_CASES])
end

function scaling_report(threads=(1, 2, 4, 8))
    project = dirname(Base.active_project())
    times = Dict{Int,Vector{Float64}}()
    for t in threads
        cmd = `$(Base.julia_cmd()) --project=$project -t $t $(@__FILE__) scaling-worker`
        out = read(cmd, String)
        times[t] = eval(Meta.parse(strip(last(split(strip(out), '\n')))))
    end
    println("Thread scaling (seconds; speedup over 1 thread in parentheses)")
    @printf("%-32s", "case")
    for t in threads
        @printf(" %14s", "$t thread" * (t == 1 ? "" : "s"))
    end
    println()
    for (k, (name, _)) in enumerate(SCALING_CASES)
        @printf("%-32s", name)
        for t in threads
            @printf(" %7.3f (%4.1fx)", times[t][k], times[1][k] / times[t][k])
        end
        println()
    end
end

if isempty(ARGS)
    sage_report()
    println()
    scaling_report()
elseif ARGS[1] == "sage"
    sage_report()
elseif ARGS[1] == "scaling"
    scaling_report()
elseif ARGS[1] == "scaling-worker"
    scaling_worker()
end
