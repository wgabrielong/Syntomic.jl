# Calibration of the memory model of sweeps (src/memory.jl).
#
#   julia --project=benchmark -t 8 benchmark/memory.jl                    # all parts
#   julia --project=benchmark -t 8 benchmark/memory.jl calibrate pair sweep
#
# Every measurement runs in a fresh Julia process: after loading Syntomic.jl and a small warm-up,
# the process records its resident memory, runs one part of a sweep, and reports the increase of
# its peak resident memory (Sys.maxrss) over that baseline, next to the counted table sizes. The
# ratio "measured / counted" is the constant factor of the model:
#
#   shared: Phases A and B, `shared_nabla_OK` at (p, f, L, N)   vs  f · g_table_entries · L² · B
#   pair:   one job `run_sweep_job` (∇_{O_K} precomputed)        vs  f_c · L³ · B (work) and
#           the f-envelope table f_c · (c · B + 8 · e · L), (e, c) = f_table_entries
#   sweep:  a whole `syntomic_sweep`                             vs  `sweep_memory_estimate(…).total`
#
# where B = coefficient_bytes(p, N). The table also lists the number of cached normal forms of
# the g-envelope (`length(genv.nf_cache)`) next to the counted `g_table_entries`.
using Syntomic, Serialization, Printf

const S = Syntomic
const PROJECT = dirname(Base.active_project())

rss_bytes() = parse(Int, strip(read(`ps -o rss= -p $(getpid())`, String))) * 1024

function ring(p, f, n)
    f == 1 && return zpn(p, n)
    return truncated_witt_vectors(p, f, n)
end

# --- the measurements (run in a subprocess) ---------------------------------------------------

function measure_shared(p, f, L, N, nweights)
    X = ring(p, f, 2)
    S.shared_nabla_OK(X, 1:2, 8, 20)                      # warm-up
    GC.gc()
    r0 = rss_bytes()
    prm = S.SyntomicParams(p, f, L, 1)
    R = S.GaloisRing(p, X.htilde, N)
    ed = S.EisensteinData(R, X, L)
    lam = S.LambdaData(ed, max(prm.ng, 2) - 2)
    genv = S.GEnvelope(prm, R, ed, lam)
    wv = S.bk_unit_v(genv, S.bk_unit_u(genv))
    nab = [
        S.to_plain(S.nabla_OK_matrix(genv, S.bk_unit_power(genv, wv, i))) for
        i in 1:nweights
    ]
    peak = Sys.maxrss() - r0
    return (peak=peak, cached=length(genv.nf_cache) + length(genv.phimono_cache))
end

function measure_pair(file)
    job = deserialize(file)
    X = first(job.rings)
    small = first(S.syntomic_sweep(S.with_power(X, 2), [2], [2]; check_memory=false))
    GC.gc()
    r0 = rss_bytes()
    S.run_sweep_job(job)
    return (peak=Sys.maxrss() - r0,)
end

function measure_sweep(p, fs, ns, is)
    Xs = [ring(p, f, maximum(ns)) for f in fs]
    S.syntomic_sweep(Xs, [2], [2]; check_memory=false)
    GC.gc()
    r0 = rss_bytes()
    S.syntomic_sweep(Xs, ns, is; check_memory=false)
    return (peak=Sys.maxrss() - r0,)
end

# --- the driver -------------------------------------------------------------------------------

function subprocess(args...)
    flush(stdout)
    script = @__FILE__
    cmd = `$(Base.julia_cmd()) --project=$PROJECT -t $(Threads.nthreads()) $script $args`
    out = read(cmd, String)
    return eval(Meta.parse(strip(last(split(strip(out), '\n')))))
end

mib(b) = b / 2^20

function calibrate_shared()
    println("shared (Phases A and B): peak MiB, counted coefficient-bytes MiB, ratio")
    for (p, f, L, N) in (
        (2, 1, 40, 60),
        (2, 1, 40, 140),
        (2, 1, 60, 200),
        (2, 1, 80, 100),
        (2, 1, 80, 300),
        (2, 1, 100, 130),
        (3, 1, 60, 80),
        (3, 1, 80, 200),
        (5, 1, 60, 80),
        (5, 1, 100, 150),
        (3, 2, 40, 100),
    )
        m = subprocess("shared", p, f, L, N, 3)
        B = S.coefficient_bytes(p, N)
        counted = f * S.g_table_entries(p, L) * L^2 * B
        @printf(
            "  p=%d f=%d L=%3d N=%3d  cached %4d / counted %4d  peak %8.1f  counted %8.1f  ratio %.2f\n",
            p,
            f,
            L,
            N,
            m.cached,
            S.g_table_entries(p, L),
            mib(m.peak),
            mib(counted),
            m.peak / counted
        )
    end
end

function calibrate_pairs()
    println("pair (one job): peak MiB, work and f-table coefficient-bytes MiB, model MiB")
    for (p, f, n, i, desc) in (
        (2, 1, 2, 20, true),
        (2, 1, 2, 40, true),
        (2, 1, 3, 20, true),
        (2, 1, 4, 25, true),
        (3, 1, 3, 15, true),
        (5, 1, 3, 12, true),
        (2, 3, 2, 20, false),
        (3, 2, 3, 10, false),
        (2, 1, 2, 60, true),
    )
        X = ring(p, f, n)
        Xc = S.computing_ring(X, desc)
        er, ec = S.exponent_phase(Xc, i)
        N = S.exponent_working_precision(p, n, i, er, ec)
        L = n * i
        Nshare = N + S.num_g_variables(p, L) + 1
        nab = S.shared_nabla_OK(Xc, [i], L, Nshare)[i]
        job = S.SweepJob(
            [X], Xc, i, N, :exponent, S.plain_top_left(nab, L - 1, L - 1), true
        )
        file = tempname()
        serialize(file, job)
        m = subprocess("pair", file)
        rm(file)
        B = S.coefficient_bytes(p, N)
        work = Xc.f * L^3 * B
        e, c = S.f_table_entries(p, n, i)
        table = Xc.f * (c * B + 8 * e * L)
        model = S.pair_memory_estimate(p, Xc.f, f, n, i, N)
        @printf(
            "  p=%d f=%d n=%d i=%2d N=%3d  peak %8.1f  work %8.1f (ratio %.2f)  table %8.1f (ratio %.2f)  model %8.1f  peak/model %.2f\n",
            p,
            f,
            n,
            i,
            N,
            mib(m.peak),
            mib(work),
            m.peak / work,
            mib(table),
            m.peak / table,
            mib(model),
            m.peak / model
        )
    end
end

function calibrate_sweeps()
    println("sweeps: peak MiB vs the estimate")
    for (p, fs, ns, is) in (
        (2, [1], 2:5, 1:12),
        (2, [1], 2:2, 1:40),
        (3, [1], 2:4, 1:15),
        (2, [3], 2:3, 1:10),
        (2, [1], 2:3, 1:30),
        (3, [1, 2, 3], 2:3, 1:12),
        (2, [1], 2:2, 1:50),
    )
        m = subprocess("sweep", p, repr(fs), repr(ns), repr(is))
        X = ring(p, maximum(fs), maximum(ns))
        Xc = S.computing_ring(X, true)
        est = S.sweep_memory_estimate(
            Xc, maximum(fs), ns, is; concurrency=S._sweep_concurrency(true)
        )
        @printf(
            "  p=%d f∈%s n∈%s i∈%s  peak %8.1f  estimate %8.1f (phase 0 %.1f, shared %.1f, pairs %.1f)  peak/estimate %.2f\n",
            p,
            fs,
            ns,
            is,
            mib(m.peak),
            mib(est.total),
            mib(est.phase0),
            mib(est.shared),
            mib(est.total - est.nablas),
            m.peak / est.total
        )
    end
end

if isempty(ARGS) || ARGS[1] == "calibrate"
    println("threads: ", Threads.nthreads())
    parts = length(ARGS) > 1 ? ARGS[2:end] : ["shared", "pair", "sweep"]
    "shared" in parts && calibrate_shared()
    "pair" in parts && calibrate_pairs()
    "sweep" in parts && calibrate_sweeps()
elseif ARGS[1] == "shared"
    println(measure_shared(parse.(Int, ARGS[2:6])...))
elseif ARGS[1] == "pair"
    println(measure_pair(ARGS[2]))
elseif ARGS[1] == "sweep"
    p = parse(Int, ARGS[2])
    fs, ns, is = (eval(Meta.parse(a)) for a in ARGS[3:5])
    println(measure_sweep(p, fs, ns, is))
end
