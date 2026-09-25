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
# Rings whose computations use the same ring (see `computing_ring`), such as W_n(F_q) for several
# q = p^f, form a group: A, B, the working precisions and the W-matrices of each pair are computed
# once for the group; only Steps 7–10, the expansion to Z_p-matrices and the cohomology are
# repeated for each ring of the group.

"""
    SweepJob

A pair `(n, i)` of a sweep: the rings `O_K/ϖ^n` of a group of rings computed together (`rings`,
e.g. several residue degrees), the ring used for the computation (the same for all pairs and all
rings of the group, see [`computing_ring`](@ref)), the working precision and the policy that chose
it, the (plain) `∇_{O_K}` and whether to run the final check.
"""
struct SweepJob
    rings::Vector{FiniteChainRing}
    cring::FiniteChainRing
    i::Int
    N::Int
    policy::Symbol
    nablaOK::Union{Nothing,PlainWMatrix}
    check::Bool
end

"""
    run_sweep_job(job::SweepJob) -> Vector{SyntomicCohomology}

Phase C of a sweep for one pair `(n, i)`: the W-matrices of the computing ring once, then the
square, its expansion and the cohomology for each ring of the job (in the order of `job.rings`).
Runs serially; also the function executed on Distributed workers, which receive only plain data.
"""
function run_sweep_job(job::SweepJob)
    complexes = _syntomic_complexes(
        job.rings,
        job.cring,
        job.i,
        job.N;
        policy=job.policy,
        threaded=false,
        nablaOK=job.nablaOK,
    )
    return [
        syntomic_cohomology(C; check=job.check, threaded=false, timings=t) for
        (C, t) in complexes
    ]
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
    shared_nabla_OK(X, is, Lmax, N; threaded=true, nmax=⌈Lmax/maximum(is)⌉)
        -> Dict{Int,PlainWMatrix}

Phases A and B of a sweep: the g-envelope at F-precision `Lmax` and precision `N`, then `w(v^i)`
and `∇_{O_K}` for each weight `i` in `is` on `F^{[1,L_i-1]}`, `L_i = min(Lmax, nmax·i)` (which
contains the `∇_{O_K}` of every pair `(n, i)` with `n ≤ nmax` as its top-left block)
([AKN, §7.1, Steps 3–5, pp. 55–56; Remark 4.67, p. 46]). The units `w(v^i)` are computed only
modulo `F^{≥L_i}`, by the method of `bk_unit_v_powers` with the smallest modelled cost, and the
transposed chain of Step 5 once for all weights. `X` must already be the ring used for the
computation (after descent).
"""
function shared_nabla_OK(
    X::FiniteChainRing,
    is,
    Lmax::Integer,
    N::Integer;
    threaded::Bool=true,
    nmax::Integer=isempty(is) ? 1 : cld(Lmax, maximum(is)),
)
    prm = SyntomicParams(X.p, X.f, Lmax, 1)          # the g-side only depends on L = Lmax
    R = GaloisRing(X.p, X.htilde, N)
    ed = EisensteinData(R, X, Lmax)
    lam = LambdaData(ed, max(prm.ng, 2) - 2)
    genv = GEnvelope(prm, R, ed, lam; threaded=threaded)             # Step 3 (O_K)
    sorted = sort(unique(collect(is)))
    Ls = Dict(i => Int(min(Lmax, nmax * i)) for i in sorted)
    powers = bk_unit_v_powers(genv, bk_unit_u(genv), sorted, Ls)     # Step 4: w(v^i) mod F^{≥L_i}
    chain = theta_chain(genv; threaded=threaded)                      # Step 5, independent of i
    # Step 5 for each weight: largest first, in parallel when `threaded`
    order = reverse(sorted)
    mats = tmap(order; threaded=threaded) do i
        to_plain(nabla_OK_matrix(genv, powers[i]; threaded=threaded, chain=chain))
    end
    return Dict{Int,PlainWMatrix}(zip(order, mats))
end

# `pmap` rethrowing the package's exceptions (such as `PrecisionExhausted`) unwrapped.
function _pmap(f, xs)
    try
        return Distributed.pmap(f, xs)
    catch err
        rethrow_unwrapped(err)
    end
end

_normalize_range(xs) = sort(unique(collect(Int, xs)))

"""
    syntomic_sweep(X::FiniteChainRing, ns, is; precision_policy=:exponent, threaded=true,
                   distributed=false, check=true, descent=true, check_memory=true)
        -> Vector{SyntomicCohomology}

The cohomology of `Z_p(i)(O_K/ϖ^n)` for all `n ∈ ns` and `i ∈ is` (the power `X.n` of `X` is
ignored), sharing the `O_K`-side data between all pairs. With `distributed = true` the pairs are
distributed over the Distributed workers (see [`add_syntomic_workers`](@ref)); otherwise they run
on threads. Results are returned sorted by `(n, i)`; empty `ns` or `is` give an empty vector.

The precision policy is `:exponent` or `:apriori` (see [`working_precision`](@ref)); `:adaptive`
is not supported, since the shared data must be computed at a precision fixed in advance. The
Eisenstein polynomial of `X` must be known to the precision required by `maximum(ns)` (see
[`with_power`](@ref)).

With `check_memory = true` the peak memory is estimated first (see
[`sweep_memory_estimate`](@ref)), and an `ArgumentError` is thrown if it exceeds the available
memory.
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
    check_memory::Bool=true,
)
    return only(
        _sweep_group(
            [X],
            ns,
            is;
            precision_policy=precision_policy,
            threaded=threaded,
            distributed=distributed,
            check=check,
            descent=descent,
            check_memory=check_memory,
        ),
    )
end

"""
    syntomic_sweep(Xs::AbstractVector{FiniteChainRing}, ns, is; distributed=true, kwargs...)
        -> Vector{Vector{SyntomicCohomology}}

Sweeps for several rings (e.g. several Eisenstein polynomials, or several residue degrees);
returns one vector of results per ring, in the order of `Xs`, each as for
`syntomic_sweep(X, ns, is; kwargs...)`.

Rings whose computations use the same ring (see [`computing_ring`](@ref) of the ring with power
`maximum(ns)`) are computed together: for example, with `descent = true`, the truncated Witt
vectors `W_n(F_q)` for `q = p, p^2, p^3, …` all use `Z/p^n`, and so do the rings with
Eisenstein polynomials in `Z_p[z]` over different residue fields. For such a group the shared
`O_K`-side data, the working precisions and the W-matrices of each pair `(n, i)` are computed
once; only Steps 7–10, the expansion to `Z_p`-matrices and the Smith normal forms are computed for
each ring ([AKN, §7.1, pp. 53–57]). The results are identical to separate sweeps.

With `distributed = true` and several Distributed workers, the groups are distributed over the
workers (each uses its own threads) if there are at least as many groups as workers, and
otherwise the pairs of each group are; only plain data is sent. The keyword arguments are those of
the sweep for one ring.
"""
function syntomic_sweep(
    Xs::AbstractVector{FiniteChainRing},
    ns,
    is;
    distributed::Bool=true,
    precision_policy::Symbol=:exponent,
    threaded::Bool=true,
    check::Bool=true,
    descent::Bool=true,
    check_memory::Bool=true,
)
    _check_policy(precision_policy, (:apriori, :exponent))
    ns, is = _normalize_range(ns), _normalize_range(is)
    out = [SyntomicCohomology[] for _ in Xs]
    (isempty(Xs) || isempty(ns) || isempty(is)) && return out
    groups = sweep_groups(Xs, maximum(ns); descent=descent)
    kw = (;
        precision_policy=precision_policy,
        check=check,
        descent=descent,
        check_memory=check_memory,
    )
    nw = distributed ? Distributed.nworkers() : 0
    results = if Distributed.nprocs() > 1 && nw > 1 && length(groups) >= nw
        # one group per worker, each on the worker's threads (only plain rings are sent)
        shares = _worker_memory_shares()
        _pmap(groups) do g
            _sweep_group(
                Xs[g],
                ns,
                is;
                threaded=threaded,
                distributed=false,
                memory_share=get(shares, Distributed.myid(), 1.0),
                kw...,
            )
        end
    else
        [
            _sweep_group(Xs[g], ns, is; threaded=threaded, distributed=distributed, kw...) for g in groups
        ]
    end
    for (g, res) in zip(groups, results)
        out[g] = res
    end
    return out
end

"""
    sweep_groups(Xs, nmax; descent=true) -> Vector{Vector{Int}}

The indices of the rings of `Xs` grouped by the ring used for their computation,
[`computing_ring`](@ref)`(with_power(X, nmax), descent)`, in the order of first appearance.
Throws an `ArgumentError` if some Eisenstein polynomial is not known to the precision that
`nmax` requires.
"""
function sweep_groups(
    Xs::AbstractVector{FiniteChainRing}, nmax::Integer; descent::Bool=true
)
    keys = FiniteChainRing[]
    groups = Vector{Int}[]
    for (k, X) in enumerate(Xs)
        Xc = computing_ring(with_power(X, nmax), descent)
        j = findfirst(==(Xc), keys)
        if j === nothing
            push!(keys, Xc)
            push!(groups, [k])
        else
            push!(groups[j], k)
        end
    end
    return groups
end

# The sweep for a group of rings with the same computing ring: one vector of results (sorted by
# (n, i)) per ring of `Xs`.
function _sweep_group(
    Xs::AbstractVector{FiniteChainRing},
    ns,
    is;
    precision_policy::Symbol=:exponent,
    threaded::Bool=true,
    distributed::Bool=false,
    check::Bool=true,
    descent::Bool=true,
    check_memory::Bool=true,
    memory_share::Real=1.0,
)
    _check_policy(precision_policy, (:apriori, :exponent))
    ns, is = _normalize_range(ns), _normalize_range(is)
    (isempty(ns) || isempty(is)) && return [SyntomicCohomology[] for _ in Xs]
    all(>=(1), ns) && all(>=(1), is) || throw(ArgumentError("n and i must be at least 1"))
    rings = [Dict(n => with_power(X, n) for n in ns) for X in Xs]     # checks the precision of E
    # one presentation for all pairs and rings: the shared O_K-side data depend on E itself
    Xc = computing_ring(rings[1][last(ns)], descent)
    all(computing_ring(r[last(ns)], descent) == Xc for r in rings) ||
        throw(ArgumentError("the rings of a group must have the same computing ring"))
    parallel = distributed && Distributed.nprocs() > 1
    if check_memory
        est = sweep_memory_estimate(
            Xc,
            maximum(X.f for X in Xs),
            ns,
            is;
            precision_policy=precision_policy,
            concurrency=_sweep_concurrency(threaded),
        )
        check_sweep_memory(est; distributed=parallel, share=memory_share)
    end
    Ns = sweep_precisions(Xc, ns, is; precision_policy=precision_policy, threaded=threaded)
    Lmax = last(ns) * last(is)
    # the g-side at Lmax loses up to M_g(Lmax) + 1 digits to δ ([AKN, Lemma 7.5(c)])
    Nshare = maximum(values(Ns)) + num_g_variables(Xc.p, Lmax) + 1
    nablas = if Lmax > 1
        shared_nabla_OK(Xc, is, Lmax, Nshare; threaded=threaded)
    else
        Dict{Int,PlainWMatrix}()
    end
    jobs = SweepJob[]
    for n in ns, i in is
        L = n * i
        nab = L > 1 ? plain_top_left(nablas[i], L - 1, L - 1) : nothing
        targets = [r[n] for r in rings]
        cring = with_power(Xc, n)
        push!(jobs, SweepJob(targets, cring, i, Ns[(n, i)], precision_policy, nab, check))
    end
    # largest jobs first, for load balance
    order = sortperm(jobs; by=j -> -(j.cring.n * j.i)^3 * j.N)
    results = if parallel
        _pmap(run_sweep_job, jobs[order])
    else
        tmap(run_sweep_job, jobs[order]; threaded=threaded)
    end
    byjob = Vector{Vector{SyntomicCohomology}}(undef, length(jobs))
    byjob[order] = results
    return [[r[k] for r in byjob] for k in eachindex(Xs)]
end

# The number of pairs of a sweep that run at the same time in this process.
function _sweep_concurrency(threaded::Bool)
    return (threaded && threads_available()) ? Threads.nthreads() : 1
end

"""
    add_syntomic_workers(k; threads_per_worker=1, oversubscribe=false) -> Vector{Int}

Start `k` Distributed worker processes, each with `threads_per_worker` threads, and load
Syntomic.jl on them. Throws an `ArgumentError`, before starting any process, if
`k · threads_per_worker` exceeds the number `Sys.CPU_THREADS` of CPU threads, unless
`oversubscribe = true`.
"""
function add_syntomic_workers(
    k::Integer; threads_per_worker::Integer=1, oversubscribe::Bool=false
)
    k >= 1 || throw(ArgumentError("the number of workers must be positive, got $k"))
    threads_per_worker >= 1 || throw(
        ArgumentError(
            "the number of threads per worker must be positive, got $threads_per_worker"
        ),
    )
    total = k * threads_per_worker
    (total <= Sys.CPU_THREADS || oversubscribe) || throw(
        ArgumentError(
            "$k workers with $threads_per_worker thread(s) each need $total threads, but there " *
            "are only $(Sys.CPU_THREADS) CPU threads; use fewer workers or threads, or pass " *
            "oversubscribe=true",
        ),
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
