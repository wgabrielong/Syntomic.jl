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

A pair `(n, i)` of a sweep: the ring `O_K/ϖ^n`, the ring used for the computation (the same for
all pairs of the sweep, see [`computing_ring`](@ref)), the working precision and the policy that
chose it, the (plain) `∇_{O_K}` and whether to run the final check.
"""
struct SweepJob
    ring::FiniteChainRing
    cring::FiniteChainRing
    i::Int
    N::Int
    policy::Symbol
    nablaOK::Union{Nothing,PlainWMatrix}
    check::Bool
end

"""
    run_sweep_job(job::SweepJob) -> SyntomicCohomology

Phase C of a sweep for one pair `(n, i)` (runs serially; also the function executed on
Distributed workers, which receive only plain data).
"""
function run_sweep_job(job::SweepJob)
    timings = Dict{Symbol,Float64}()
    C = _syntomic_complex(
        job.ring,
        job.cring,
        job.i,
        job.N;
        policy=job.policy,
        threaded=false,
        timings=timings,
        nablaOK=job.nablaOK,
    )
    return syntomic_cohomology(C; check=job.check, threaded=false, timings=timings)
end

"""
    sweep_precisions(Xc, ns, is; precision_policy=:exponent) -> Dict{Tuple{Int,Int},Int}

The working precision of every pair `(n, i)` of a sweep (`:apriori` or `:exponent`), for the
ring `Xc` used in the computation (see [`computing_ring`](@ref)).
"""
function sweep_precisions(
    Xc::FiniteChainRing, ns, is; precision_policy::Symbol=:exponent, threaded::Bool=true
)
    _check_policy(precision_policy, (:apriori, :exponent))
    pairs = [(n, i) for n in ns for i in is]
    Ns = tmap(pairs; threaded=threaded) do (n, i)
        working_precision(
            with_power(Xc, n),
            i;
            precision_policy=precision_policy,
            threaded=false,
            descent=false,
        )
    end
    return Dict(zip(pairs, Ns))
end

"""
    shared_nabla_OK(X, is, Lmax, N; threaded=true) -> Dict{Int,PlainWMatrix}

Phases A and B of a sweep: the g-envelope and `w(v)` at F-precision `Lmax` and precision `N`, then
`∇_{O_K}` for each weight `i` in `is` (on `F^{[1,Lmax-1]}`) ([AKN, §7.1, Steps 3–5, pp. 55–56;
Remark 4.67, p. 46]). `X` must already be the ring used for the computation (after descent).
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

# `pmap` rethrowing the package's exceptions (such as `PrecisionExhausted`) unwrapped.
function _pmap(f, xs)
    try
        return Distributed.pmap(f, xs)
    catch err
        rethrow_unwrapped(err)
    end
end

"""
    syntomic_sweep(X::FiniteChainRing, ns, is; precision_policy=:exponent, threaded=true,
                   distributed=false, check=true, descent=true) -> Vector{SyntomicCohomology}

The cohomology of `Z_p(i)(O_K/ϖ^n)` for all `n ∈ ns` and `i ∈ is` (the power `X.n` of `X` is
ignored), sharing the `O_K`-side data between all pairs. With `distributed = true` the pairs are
distributed over the Distributed workers (see [`add_syntomic_workers`](@ref)); otherwise they run
on threads. Results are returned sorted by `(n, i)`; empty `ns` or `is` give an empty vector.

The precision policy is `:exponent` or `:apriori` (see [`working_precision`](@ref)); `:adaptive`
is not supported, since the shared data must be computed at a precision fixed in advance. The
Eisenstein polynomial of `X` must be known to the precision required by `maximum(ns)` (see
[`with_power`](@ref)).
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
    _check_policy(precision_policy, (:apriori, :exponent))
    ns = sort(unique(collect(ns)))
    is = sort(unique(collect(is)))
    (isempty(ns) || isempty(is)) && return SyntomicCohomology[]
    all(>=(1), ns) && all(>=(1), is) || throw(ArgumentError("n and i must be at least 1"))
    rings = Dict(n => with_power(X, n) for n in ns)          # checks the precision of E
    # one presentation for all pairs: the shared O_K-side data depend on E itself
    Xc = computing_ring(rings[last(ns)], descent)
    Ns = sweep_precisions(Xc, ns, is; precision_policy=precision_policy, threaded=threaded)
    Lmax = last(ns) * last(is)
    # the g-side at Lmax loses up to M_g(Lmax) + 1 digits to δ ([AKN, Lemma 7.5(c)])
    Nshare = maximum(values(Ns)) + num_g_variables(X.p, Lmax) + 1
    nablas = if Lmax > 1
        shared_nabla_OK(Xc, is, Lmax, Nshare; threaded=threaded)
    else
        Dict{Int,PlainWMatrix}()
    end
    jobs = SweepJob[]
    for n in ns, i in is
        L = n * i
        nab = L > 1 ? plain_top_left(nablas[i], L - 1, L - 1) : nothing
        cring = with_power(Xc, n)
        push!(jobs, SweepJob(rings[n], cring, i, Ns[(n, i)], precision_policy, nab, check))
    end
    # largest jobs first, for load balance
    order = sortperm(jobs; by=j -> -(j.ring.n * j.i)^3 * j.N)
    results = if distributed && Distributed.nprocs() > 1
        _pmap(run_sweep_job, jobs[order])
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
        return _pmap(work, Xs)
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
    k_theory_table(X::FiniteChainRing, is; integral=false, kwargs...) -> KTheoryTable

The K-groups `K_{2i-1}(X)` and `K_{2i-2}(X)` for all weights `i ∈ is`, computed with a sweep
(keyword arguments are passed on to [`syntomic_sweep`](@ref)) and [AKN, Cor. 2.16, p. 9]
(p-adic by default; `integral = true` adds the prime-to-p part `Z/(q^i - 1)` of
[AKN, §1.1, (1), p. 2]). The result is a [`KTheoryTable`](@ref): a vector of rows
`(degree = r, exponents = [a_1, …], prime_to_p_order = m)`, sorted by degree, meaning
`K_r ≅ Z/m ⊕ ⊕_j Z/p^{a_j}` (`K_0 = Z` is reported with `exponents = nothing`), displayed as the
primary decomposition of each group.
"""
function k_theory_table(X::FiniteChainRing, is; integral::Bool=false, kwargs...)
    results = syntomic_sweep(X, [X.n], is; kwargs...)
    q = big(X.p)^X.f
    rows = KTheoryRow[]
    for H in sort(results; by=H -> H.i)
        i = H.i
        push!(rows, KTheoryRow((2 * i - 1, H.h1, integral ? q^i - 1 : big(1))))
        if i >= 2
            push!(rows, KTheoryRow((2 * i - 2, H.h2, big(1))))
        else
            push!(rows, KTheoryRow((0, nothing, big(1))))
        end
    end
    return KTheoryTable(X, integral, sort!(rows; by=r -> r.degree))
end
