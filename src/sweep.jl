# Sweeps over (n, i) sharing the O_K-side data, and distributed computations.
#
# The envelope of O_K over W(F_q)[[z_0, z_1]], the unit w(v) and the matrices ∇_{O_K} depend only on
# K (through E) and on the F-precision L = n·i, not on n itself: ∇_{O_K} on F^{[1,L-1]} is the
# top-left block of ∇_{O_K} computed on F^{[1,L'-1]} for any L' ≥ L ([AKN, Remark 4.67, p. 46];
# [AKN, §7.1, Step 5, p. 56]). A sweep therefore
#   A. builds the g-envelope and w(v) once, at the largest F-precision and working precision;
#   B. computes w(v^i) and ∇_{O_K}(i) once for each weight i;
#   C. runs the remaining steps (the envelope of O_K/ϖ^n, Steps 6–10 and the cohomology) for every
#      pair (n, i), in parallel over threads or Distributed workers.

"""
    SweepJob

A pair `(n, i)` of a sweep together with its working precision and (plain) `∇_{O_K}`.
"""
struct SweepJob
    ring::FiniteChainRing
    i::Int
    N::Int
    nablaOK::Union{Nothing,PlainWMatrix}
    descent::Bool
    check::Bool
end

"""
    run_sweep_job(job::SweepJob) -> SyntomicCohomology

Phase C of a sweep for one pair `(n, i)` (runs serially; also the function executed on
Distributed workers, which receive only plain data).
"""
function run_sweep_job(job::SweepJob)
    timings = Dict{Symbol,Float64}()
    C = syntomic_complex(
        job.ring,
        job.i;
        N=job.N,
        threaded=false,
        descent=job.descent,
        timings=timings,
        nablaOK=job.nablaOK,
    )
    return syntomic_cohomology(
        C; check=job.check, threaded=false, policy=:exponent, timings=timings
    )
end

"""
    sweep_precisions(X, ns, is; precision_policy=:exponent) -> Dict{Tuple{Int,Int},Int}

The working precision of every pair `(n, i)` of a sweep (`:apriori` or `:exponent`).
"""
function sweep_precisions(
    X::FiniteChainRing, ns, is; precision_policy::Symbol=:exponent, threaded::Bool=true
)
    precision_policy in (:apriori, :exponent) ||
        throw(ArgumentError("sweeps support the precision policies :apriori and :exponent"))
    pairs = [(n, i) for n in ns for i in is]
    Ns = tmap(pairs; threaded=threaded) do (n, i)
        working_precision(
            with_power(X, n), i; precision_policy=precision_policy, threaded=false
        )
    end
    return Dict(zip(pairs, Ns))
end

"""
    shared_nabla_OK(X, is, Lmax, N; threaded=true) -> Dict{Int,PlainWMatrix}

Phases A and B of a sweep: the g-envelope and `w(v)` at F-precision `Lmax` and precision `N`, then
`∇_{O_K}` for each weight `i` in `is` (on `F^{[1,Lmax-1]}`). `X` must already be the ring used for
the computation (after descent).
"""
function shared_nabla_OK(
    X::FiniteChainRing, is, Lmax::Integer, N::Integer; threaded::Bool=true
)
    prm = SyntomicParams(X.p, X.f, Lmax, 1)          # the g-side only depends on L = Lmax
    R = GaloisRing(X.p, X.htilde, N)
    ed = EisensteinData(R, X, Lmax)
    lam = LambdaData(ed, max(prm.ng, 2) - 2)
    genv = GEnvelope(prm, R, ed, lam; threaded=threaded)             # Step 3 (O_K)
    wv = bk_unit_v(genv, bk_unit_u(genv))                              # Step 4
    out = Dict{Int,PlainWMatrix}()
    sorted = sort(unique(collect(is)))
    vi = nothing
    last = 0
    for i in sorted
        # w(v^i), incrementally along consecutive weights
        vi = if vi !== nothing && i == last + 1
            g_mul(genv, vi, wv)
        else
            bk_unit_power(genv, wv, i)
        end
        last = i
        out[i] = to_plain(nabla_OK_matrix(genv, vi; threaded=threaded))   # Step 5
    end
    return out
end

"""
    syntomic_sweep(X::FiniteChainRing, ns, is; precision_policy=:exponent, threaded=true,
                   distributed=false, check=true, descent=true) -> Vector{SyntomicCohomology}

The cohomology of `Z_p(i)(O_K/ϖ^n)` for all `n ∈ ns` and `i ∈ is` (the power `X.n` of `X` is
ignored), sharing the `O_K`-side data between all pairs. With `distributed = true` the pairs are
distributed over the Distributed workers (see [`add_syntomic_workers`](@ref)); otherwise they run
on threads. Results are returned sorted by `(n, i)`.
"""
function syntomic_sweep(
    X::FiniteChainRing,
    ns,
    is;
    precision_policy::Symbol=:exponent,
    threaded::Bool=true,
    distributed::Bool=false,
    check::Bool=true,
    descent::Bool=true,
)
    ns = sort(unique(collect(ns)))
    is = sort(unique(collect(is)))
    all(>=(1), ns) && all(>=(1), is) || throw(ArgumentError("n and i must be at least 1"))
    Ns = sweep_precisions(X, ns, is; precision_policy=precision_policy, threaded=threaded)
    Lmax = maximum(n * i for n in ns for i in is)
    # the g-side at Lmax loses up to M_g(Lmax) + 1 digits to δ ([AKN, Lemma 7.5(c)])
    Nshare = maximum(values(Ns)) + num_g_variables(X.p, Lmax) + 1
    Xd = descent ? descent_ring(X, Nshare) : nothing
    Xc = something(Xd, X)
    nablas = if Lmax > 1
        shared_nabla_OK(Xc, is, Lmax, Nshare; threaded=threaded)
    else
        Dict{Int,PlainWMatrix}()
    end
    jobs = SweepJob[]
    for n in ns, i in is
        L = n * i
        nab = L > 1 ? plain_top_left(nablas[i], L - 1, L - 1) : nothing
        push!(jobs, SweepJob(with_power(X, n), i, Ns[(n, i)], nab, descent, check))
    end
    # largest jobs first, for load balance
    order = sortperm(jobs; by=j -> -(j.ring.n * j.i)^3 * j.N)
    results = if distributed && Distributed.nprocs() > 1
        Distributed.pmap(run_sweep_job, jobs[order])
    else
        tmap(run_sweep_job, jobs[order]; threaded=threaded)
    end
    out = Vector{SyntomicCohomology}(undef, length(jobs))
    out[order] = results
    return out
end

"""
    syntomic_sweep(Xs::AbstractVector{FiniteChainRing}, ns, is; distributed=true, kwargs...)

Sweeps for several rings (e.g. several Eisenstein polynomials). With `distributed = true` each
ring is handled by one Distributed worker, which uses its own threads; the rings are sent as plain
data. Returns one vector of results per ring.
"""
function syntomic_sweep(
    Xs::AbstractVector{FiniteChainRing}, ns, is; distributed::Bool=true, kwargs...
)
    work(X) = syntomic_sweep(X, ns, is; distributed=false, kwargs...)
    if distributed && Distributed.nprocs() > 1
        return Distributed.pmap(work, Xs)
    else
        return map(work, Xs)
    end
end

"""
    add_syntomic_workers(k; threads_per_worker=1) -> Vector{Int}

Start `k` Distributed worker processes, each with `threads_per_worker` threads, and load
Syntomic.jl on them. Warns if `k · threads_per_worker` exceeds the number of CPU threads.
"""
function add_syntomic_workers(k::Integer; threads_per_worker::Integer=1)
    total = k * threads_per_worker
    total <= Sys.CPU_THREADS || @warn(
        "starting $k workers with $threads_per_worker threads each oversubscribes the $(Sys.CPU_THREADS) CPU threads"
    )
    ids = Distributed.addprocs(k; exeflags=`--threads=$threads_per_worker`)
    Distributed.remotecall_eval(Main, ids, :(using Syntomic))
    return ids
end

"""
    k_theory_table(X::FiniteChainRing, is; integral=false, kwargs...) -> Vector{NamedTuple}

The K-groups `K_{2i-1}(X)` and `K_{2i-2}(X)` for all weights `i ∈ is`, computed with a sweep
(p-adic by default; `integral = true` adds the prime-to-p part `Z/(q^i - 1)`). Each entry is
`(degree = r, exponents = [a_1, …], prime_to_p_order = m)`, meaning `K_r ≅ Z/m ⊕ ⊕_j Z/p^{a_j}`
(`K_0 = Z` is reported with `exponents = nothing`).
"""
function k_theory_table(X::FiniteChainRing, is; integral::Bool=false, kwargs...)
    results = syntomic_sweep(X, [X.n], is; kwargs...)
    q = big(X.p)^X.f
    rows = NamedTuple[]
    for H in sort(results; by=H -> H.i)
        i = H.i
        push!(
            rows,
            (degree=2i - 1, exponents=H.h1, prime_to_p_order=integral ? q^i - 1 : big(1)),
        )
        if i >= 2
            push!(rows, (degree=2i - 2, exponents=H.h2, prime_to_p_order=big(1)))
        else
            push!(rows, (degree=0, exponents=nothing, prime_to_p_order=big(1)))
        end
    end
    return sort(rows; by=r -> r.degree)
end
