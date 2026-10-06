# Syntomic.jl against the original SageMath code of [AKN24] (KZPN, https://github.com/antieau/kzpn,
# commit ed381b4), on the same machine.
#
#   KZPN_DIR=/path/to/kzpn sage -python scripts/sage/time_kzpn.py --output kzpn_times.json
#   julia --project=benchmark benchmark/vs_sage.jl sage kzpn_times.json
#   julia --project=benchmark benchmark/vs_sage.jl appendix [p,n ...] [--threads=8]
#
# sage:     for every case timed by time_kzpn.py, Syntomic.jl computes the same ring and weight at
#           the same working precision N (so that both certify the same groups), in fresh
#           processes with 1 and with 8 threads, and at its default precision policy with 8
#           threads. H^1 and H^2 must agree with Sage's. Prints a Markdown table.
# appendix: the p-adic K-groups K_r(Z/p^n; Z_p) of [AKN24, Appendix A, pp. 61–64], including the
#           cells that the paper marks as not computed within 48 hours: each ring in a fresh
#           process (k_theory_table), compared with the paper's cells. Prints a Markdown table
#           with the time and the peak memory of each ring.
#
# Times are wall-clock minima over repeated runs after a warm-up (compilation excluded).
using Syntomic
using JSON3
using Printf

const S = Syntomic
const ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECT = @__DIR__

ring(p, n, E) = finite_chain_ring(p, [0, 1], [[BigInt(e)] for e in E], n)
ringname(p, n) = "Z/$p^$n"

# The minimum over repeated runs of f() after one run that is not counted (it compiles the code
# for the sizes of this case); long runs are repeated fewer times.
function besttime(f)
    first = @elapsed f()
    reps = if first < 2
        5
    elseif first < 30
        3
    else
        1
    end
    return minimum(1:reps) do _
        GC.gc()
        @elapsed f()
    end
end

function subprocess(args...; threads)
    return readchomp(
        `$(Base.julia_cmd()) --project=$PROJECT -t $threads $(@__FILE__) $args`
    )
end

# --- against Sage -----------------------------------------------------------------------------

function sage_worker(file, out)
    cases = JSON3.read(read(file, String)).cases
    syntomic_cohomology(zpn(2, 2), 4; N=40)                       # warm-up
    default = Threads.nthreads() > 1
    res = map(cases) do c
        X = ring(c.p, c.n, c.E)
        H = syntomic_cohomology(X, c.i; N=c.N)
        agree = if get(c, :status, "") == "ok"
            (H.h1, H.h2) == (Vector{Int}(c.h1), Vector{Int}(c.h2))
        else
            nothing
        end
        t = besttime(() -> syntomic_cohomology(X, c.i; N=c.N))
        td = default ? besttime(() -> syntomic_cohomology(X, c.i)) : nothing
        Dict("t_equal" => t, "t_default" => td, "agree" => agree)
    end
    write(out, JSON3.write(res))
    return nothing
end

function sage_report(file)
    data = JSON3.read(read(file, String))
    runs = Dict{Int,Any}()
    for t in (1, 8)
        out = tempname()
        subprocess("sage-worker", file, out; threads=t)
        runs[t] = JSON3.read(read(out, String))
        rm(out)
    end
    pv = data.provenance
    println("Sage $(pv.sage_version), KZPN $(pv.kzpn_upstream_commit[1:7]); $(pv.timing)")
    println()
    println(
        "| ring | i | L | N | Sage | Syntomic.jl, 1 thread | Syntomic.jl, 8 threads | Syntomic.jl, default precision, 8 threads |",
    )
    println("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for (k, c) in enumerate(data.cases)
        r1, r8 = runs[1][k], runs[8][k]
        all(r -> r.agree !== false, (r1, r8)) ||
            error("H^1, H^2 differ from Sage for $(ringname(c.p, c.n)), i = $(c.i)")
        status = get(c, :status, "")
        sage = status == "ok" ? Float64(c.seconds) : Float64(get(c, :timeout_seconds, NaN))
        sagestr = if status == "ok"
            fmt(sage)
        elseif status == "timeout"
            "> $(fmt(sage))"
        else
            "not run"
        end
        function cell(t)
            status == "ok" && return "$(fmt(t)) (×$(round(Int, sage / t)))"
            status == "timeout" && return "$(fmt(t)) (> ×$(round(Int, sage / t)))"
            return fmt(t)
        end
        println(
            "| `$(ringname(c.p, c.n))` | $(c.i) | $(c.n * c.i) | $(c.N) | $sagestr | ",
            cell(r1.t_equal),
            " | ",
            cell(r8.t_equal),
            " | ",
            fmt(r8.t_default),
            " |",
        )
    end
end

function fmt(t)
    t >= 3600 && return @sprintf("%.1f h", t / 3600)
    t >= 60 && return @sprintf("%.1f min", t / 60)
    t >= 1 && return @sprintf("%.2f s", t)
    t >= 0.001 && return @sprintf("%.1f ms", 1000 * t)
    return @sprintf("%.2f ms", 1000 * t)
end

# --- [AKN24, Appendix A] ----------------------------------------------------------------------

function appendix_rows()
    fx = JSON3.read(read(joinpath(ROOT, "test", "fixtures", "akn_tables.json"), String))
    paper = Dict{Tuple{Int,Int},Dict{Int,Vector{Int}}}()
    for r in fx.appendix_A
        get!(paper, (r.p, r.n), Dict{Int,Vector{Int}}())[r.r] = sort(
            Vector{Int}(r.exponents)
        )
    end
    notdone = Dict(
        (Int(x.p), Int(x.n)) => Vector{Int}(x.r) for
        x in fx.provenance.appendix_A_not_computed
    )
    return paper, notdone
end

function appendix_worker(p, n, check_memory)
    p, n = parse(Int, p), parse(Int, n)
    paper, notdone = appendix_rows()
    cells = paper[(p, n)]
    maxdeg = maximum(vcat(collect(keys(cells)), get(notdone, (p, n), Int[])))
    imax = div(maxdeg + 2, 2)
    k_theory_table(zpn(p, n), 1:3)                                 # warm-up
    GC.gc()
    tab = nothing
    t = try
        @elapsed tab = k_theory_table(zpn(p, n), 1:imax; check_memory=check_memory == "true")
    catch err
        err isa ArgumentError || rethrow()
        return println(JSON3.write((; error=sprint(showerror, err), imax, maxdeg)))
    end
    rows = Dict(r.degree => r for r in tab if 1 <= r.degree <= maxdeg)
    mismatches = count(d -> sort(rows[d].exponents) != cells[d], keys(cells))
    return println(JSON3.write((; t, rss=Sys.maxrss(), imax, maxdeg, mismatches)))
end

function appendix_report(args)
    threads = 8
    check_memory = true
    wanted = Tuple{Int,Int}[]
    for a in args
        if startswith(a, "--threads=")
            threads = parse(Int, split(a, '=')[2])
        elseif a == "--no-memory-check"
            check_memory = false
        else
            push!(wanted, Tuple(parse.(Int, split(a, ','))))
        end
    end
    paper, notdone = appendix_rows()
    keys_ = isempty(wanted) ? sort(collect(keys(paper))) : wanted
    println(
        "| ring | degrees | computed in [AKN24] | not computed within 48 h | Syntomic.jl, $threads threads | peak memory |",
    )
    println("|:--|--:|--:|--:|--:|--:|")
    total = 0.0
    for (p, n) in keys_
        out = subprocess("appendix-worker", p, n, check_memory; threads=threads)
        r = JSON3.read(last(split(out, '\n')))
        nc = length(get(notdone, (p, n), Int[]))
        head = "| `$(ringname(p, n))` | 1–$(r.maxdeg) | $(length(paper[(p, n)])) | $(nc == 0 ? "—" : nc) |"
        if haskey(r, :error)
            println(head, " not run: ", r.error, " | |")
            continue
        end
        r.mismatches == 0 ||
            error("Z/$p^$n: $(r.mismatches) degrees differ from [AKN24, Appendix A]")
        total += r.t
        @printf("%s %s | %.1f GiB |\n", head, fmt(r.t), r.rss / 2^30)
        flush(stdout)
    end
    return println("\ntotal: ", fmt(total))
end

if !isempty(ARGS)
    mode = ARGS[1]
    if mode == "sage"
        sage_report(ARGS[2])
    elseif mode == "sage-worker"
        sage_worker(ARGS[2], ARGS[3])
    elseif mode == "appendix"
        appendix_report(ARGS[2:end])
    elseif mode == "appendix-worker"
        appendix_worker(ARGS[2:4]...)
    else
        error("unknown mode $mode (use sage FILE or appendix)")
    end
end
