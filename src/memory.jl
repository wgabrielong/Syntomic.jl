# Memory estimates for sweeps, from the sizes of the tables that are built.
#
# A sweep (see sweep.jl) runs, in this order:
#   * Phase 0 (policy :exponent): the exponents of red and can1 for every pair, from the f-envelope
#     of O_K/ϖ^n at a small precision (`exponent_phase`), several pairs at a time;
#   * Phases A and B (shared): the g-envelope at the largest F-precision L = Lmax and the precision
#     Nshare, whose tables hold the normal forms of the non-reduced monomials ∏ g_u^{e_u} and the
#     values φ(g^{μ(t)}), each an element Σ_{t<L} c_t(z_0) g^{μ(t)} of about L² coefficients;
#     then the matrices ∇_{O_K}(i), kept as plain integers while the pairs run;
#   * Phase C: for each pair (n, i) that is running, the f-envelope of O_K/ϖ^n at L = n·i (its memo
#     table holds the normal forms of the monomials z_0^a ∏ f_u^{e_u} of F-weight < L), the W-matrices,
#     and the Z_p-matrices of the square and of the Smith normal forms.
# The model counts coefficients and multiplies them by the memory of one coefficient modulo p^N.
# The peak resident memory of a pair is dominated either by the table of the f-envelope (n = 2 and
# large i) or by the memory allocated between two garbage collections, which grows like the number
# L³ of coefficients handled when the L columns of the matrices are built. The constant factors were
# calibrated against the peak resident memory of the parts of sweeps and of whole sweeps
# (benchmark/memory.jl); they cover the temporaries and the allocator's overhead.

"""
    coefficient_bytes(p, N) -> Int

The memory of one coefficient modulo `p^N` in a FLINT polynomial or matrix: 8 bytes if
`p^N < 2^62` (an inline integer), and otherwise 8 bytes plus a GMP integer, i.e. 40 bytes of
header and allocation overhead and 8 bytes per 64-bit limb.
"""
function coefficient_bytes(p::Integer, N::Integer)
    bits = N * log2(p)
    bits <= 62 && return 8
    return 48 + 8 * cld(ceil(Int, bits), 64)
end

# The number of vectors (e_0, …, e_{m-1}) with 0 ≤ e_u ≤ dmax (dmax = nothing: unbounded) and
# Σ e_u p^u = w, for 0 ≤ w < W (as floating-point numbers: they grow quickly with w).
function _digit_representations(p::Integer, m::Integer, W::Integer, dmax)
    W <= 0 && return Float64[]
    counts = zeros(W)
    counts[1] = 1
    for u in 0:(m - 1)
        step = big(p)^u
        step >= W && break
        s = Int(step)
        new = copy(counts)
        # new[v] = Σ_{0 ≤ d ≤ dmax} counts[v - d·s], as a running sum over d
        for v in (s + 1):W
            new[v] += new[v - s]
            if dmax !== nothing && v > (dmax + 1) * s
                new[v] -= counts[v - (dmax + 1) * s]
            end
        end
        counts = new
    end
    return counts
end

"""
    g_table_entries(p, L) -> Int

The number of elements of the g-envelope at F-precision `L` that the model counts: the
non-reduced monomials `∏ g_u^{e_u}` of F-weight `< L` with digits `0 ≤ e_u ≤ 2p-2` (products of
two basis monomials, whose normal forms are tabulated, [AKN, §7.1, Step 3, p. 55]), plus the
`⌈L/p⌉` values `φ(g^{μ(t)})`, `t < L/p`.
"""
function g_table_entries(p::Integer, L::Integer)
    L <= 1 && return 0
    reps = _digit_representations(p, num_g_variables(p, L), L, 2 * p - 2)
    return _bytes(sum(reps) - L + cld(L, p))  # one reduced monomial per F-weight
end

"""
    f_table_entries(p, n, i) -> (entries, coefficients)

The number of monomials `z_0^a ∏ f_u^{e_u}` of F-weight `a + n Σ e_u p^u < L = n·i` in the
f-envelope of `O_K/ϖ^n` ([AKN, §7.1, Step 3, p. 55]; [AKN, Prop. 3.34(2), p. 24]), whose normal
forms the memo table of the f-envelope holds, and the number of coefficients of these normal forms
in F-weights `≥` their own (the others are zero). With `b(w)` the number of ways of writing
`w = Σ_u e_u p^u` with `u < M_f + 1`, these are `Σ_{w<i} b(w) m_w` and `Σ_{w<i} b(w) m_w(m_w+1)/2`,
`m_w = L - n·w`.
"""
function f_table_entries(p::Integer, n::Integer, i::Integer)
    L = n * i
    reps = _digit_representations(p, num_f_variables(p, i), i, nothing)
    entries = sum(reps[w + 1] * (L - n * w) for w in 0:(i - 1); init=0.0)
    coefficients = sum(
        reps[w + 1] * (L - n * w) * (L - n * w + 1) / 2 for w in 0:(i - 1); init=0.0
    )
    return _bytes(entries), _bytes(coefficients)
end

# Calibrated constants (see benchmark/memory.jl).
const MEMORY_G = 1.0                  # per coefficient of the g-envelope tables
const MEMORY_WORK = 2.0               # per coefficient of the fc·L³ handled by the f-side of a pair
const MEMORY_F = 0.6                  # per coefficient of the f-envelope table
const MEMORY_S = 0.3                  # per coefficient of the square and Smith normal forms
const MEMORY_Q = 40.0                 # per entry of the f(L-1) × f(L-1) Z_p-matrices of a pair
const MEMORY_CONCURRENT = 2.0         # for several pairs at a time (the heap grows between GCs)
const MEMORY_BASE = 64 * 2^20         # allocations independent of the size

# Estimates are capped (at 8 PiB), so that sums of them cannot overflow.
const MEMORY_CAP = typemax(Int) >> 10
_bytes(x::Real) = x >= MEMORY_CAP ? MEMORY_CAP : round(Int, x)

"""
    shared_memory_estimate(p, f, nweights, L, N) -> Int

The estimated peak memory (bytes) of Phases A and B of a sweep: the g-envelope at F-precision `L`
and precision `N` over `W(F_q)`, `q = p^f` (`f` of the ring used for the computation), and the
matrices `∇_{O_K}` for `nweights` weights. The model is

    64 MiB + f · g_table_entries(p, L) · L² · B + nablaOK_memory_estimate(p, f, nweights, L, N),

with `B = coefficient_bytes(p, N)` (see [`g_table_entries`](@ref), [`coefficient_bytes`](@ref)).
"""
function shared_memory_estimate(
    p::Integer, f::Integer, nweights::Integer, L::Integer, N::Integer
)
    L <= 1 && return 0
    g = MEMORY_G * f * g_table_entries(p, L) * Float64(L)^2 * coefficient_bytes(p, N)
    return _bytes(MEMORY_BASE + g + nablaOK_memory_estimate(p, f, nweights, L, N))
end

"""
    nablaOK_memory_estimate(p, f, nweights, L, N) -> Int

The memory (bytes) of the matrices `∇_{O_K}` (`(L-1) × (L-1)` over `W(F_q)/p^N`) for `nweights`
weights, kept as plain integers ([`PlainWMatrix`](@ref)) while the pairs of a sweep run:
`nweights · f · (L-1)² · (coefficient_bytes(p, N) + 40)`.
"""
function nablaOK_memory_estimate(
    p::Integer, f::Integer, nweights::Integer, L::Integer, N::Integer
)
    L <= 1 && return 0
    return _bytes(nweights * f * Float64(L - 1)^2 * (coefficient_bytes(p, N) + 40))
end

"""
    fside_memory_estimate(p, fc, n, i, N) -> Int

The estimated peak memory (bytes) of the f-envelope of `O_K/ϖ^n` at F-precision `L = n·i` and
precision `N` with `fc`-dimensional arithmetic, and of the matrices built from it:

    max(2 · fc · L³ · B,  0.6 · fc · (c · B + 8 · e · L)),

where `(e, c) = f_table_entries(p, n, i)` and `B = coefficient_bytes(p, N)` (zero coefficients
take 8 bytes).
"""
function fside_memory_estimate(p::Integer, fc::Integer, n::Integer, i::Integer, N::Integer)
    L = n * i
    L <= 1 && return 0
    B = coefficient_bytes(p, N)
    entries, coefficients = f_table_entries(p, n, i)
    work = MEMORY_WORK * fc * Float64(L)^3 * B
    table = MEMORY_F * fc * (Float64(coefficients) * B + 8.0 * entries * L)
    return _bytes(max(work, table))
end

"""
    pair_memory_estimate(p, fc, f, n, i, N) -> Int

The estimated peak memory (bytes) of one pair `(n, i)` of a sweep at precision `N`, computed with
`fc`-dimensional arithmetic (the residue degree of the ring used for the computation) for a target
ring of residue degree `f`: [`fside_memory_estimate`](@ref) plus the `Z_p`-matrices of the square
and of the Smith normal forms and the memory allocated while they are computed,
`0.3 · ((fL)³ + 40 · (f(L-1))²) · coefficient_bytes(p, N)`.
"""
function pair_memory_estimate(
    p::Integer, fc::Integer, f::Integer, n::Integer, i::Integer, N::Integer
)
    L = n * i
    L <= 1 && return 0
    square =
        MEMORY_S *
        (Float64(f * L)^3 + MEMORY_Q * Float64(f * (L - 1))^2) *
        coefficient_bytes(p, N)
    return _bytes(fside_memory_estimate(p, fc, n, i, N) + square)
end

"""
    estimated_working_precision(p, n, i, precision_policy) -> Int

The working precision assumed by the memory estimates before any computation: the precision of
[`apriori_working_precision`](@ref) for `:apriori`, and for `:exponent` the one of
[`exponent_working_precision`](@ref) with both exponents replaced by `n(i-1)` (the first guess of
[`exponent_phase`](@ref)).
"""
function estimated_working_precision(
    p::Integer, n::Integer, i::Integer, precision_policy::Symbol
)
    precision_policy == :apriori && return apriori_working_precision(p, n, i)
    return exponent_working_precision(p, n, i, n * (i - 1), n * (i - 1))
end

"""
    sweep_memory_estimate(Xc, f, ns, is; precision_policy=:exponent, concurrency=1) -> NamedTuple

The estimated peak memory (bytes) of a sweep over `n ∈ ns`, `i ∈ is` computed with the ring `Xc`
(see [`computing_ring`](@ref)) for target rings of residue degree at most `f`, when
`concurrency` pairs run at the same time in this process (the largest ones first, as in
[`syntomic_sweep`](@ref)); the working precisions are those of
[`estimated_working_precision`](@ref). The fields are

- `phase0`: the exponents of all pairs (`:exponent` only), `concurrency` at a time
  ([`fside_memory_estimate`](@ref) at the first precision of [`exponent_phase`](@ref));
- `shared`: Phases A and B ([`shared_memory_estimate`](@ref));
- `nablas`: the matrices `∇_{O_K}` kept during Phase C ([`nablaOK_memory_estimate`](@ref));
- `pairs`: the estimate for each pair, largest first ([`pair_memory_estimate`](@ref));
- `total`: `max(phase0, shared, nablas + the sum of the concurrency largest pairs)`.

When several pairs run at the same time, the sums for Phase 0 and for the pairs are multiplied by
2: the heap grows further between two garbage collections when more memory is live.
"""
function sweep_memory_estimate(
    Xc::FiniteChainRing,
    f::Integer,
    ns,
    is;
    precision_policy::Symbol=:exponent,
    concurrency::Integer=1,
)
    ns, is = _normalize_range(ns), _normalize_range(is)
    (isempty(ns) || isempty(is)) &&
        return (phase0=0, shared=0, nablas=0, pairs=Int[], total=0)
    p, fc = Xc.p, Xc.f
    pairs = [(n, i) for n in ns for i in is]
    Ns = Dict(q => estimated_working_precision(p, q..., precision_policy) for q in pairs)
    k = max(concurrency, 1)
    factor = k > 1 ? MEMORY_CONCURRENT : 1.0
    top(v) = _bytes(factor * sum(sort(v; rev=true)[1:min(k, length(v))]; init=0))
    phase0 = if precision_policy == :exponent
        top([
            fside_memory_estimate(p, fc, n, i, first(exponent_phase_precisions(p, n, i))) for (n, i) in pairs
        ])
    else
        0
    end
    Lmax = last(ns) * last(is)
    Nshare = maximum(values(Ns)) + num_g_variables(p, Lmax) + 1
    shared = shared_memory_estimate(p, fc, length(is), Lmax, Nshare)
    nablas = nablaOK_memory_estimate(p, fc, length(is), Lmax, Nshare)
    est = sort([pair_memory_estimate(p, fc, f, q..., Ns[q]) for q in pairs]; rev=true)
    total = max(phase0, shared, nablas + top(est))
    return (phase0=phase0, shared=shared, nablas=nablas, pairs=est, total=total)
end

# A number of bytes for messages.
function _format_bytes(b::Real)
    b >= 2^30 && return string(round(b / 2^30; digits=2), " GiB")
    return string(round(b / 2^20; digits=1), " MiB")
end

"""
    available_memory() -> Int

The memory (bytes) available to this process for new data: `Sys.free_memory()`, and on macOS at
least the share `kern.memorystatus_level` (in percent) of the total memory that the system
considers available. (On macOS `Sys.free_memory()` counts only free pages, not the inactive and
purgeable pages that are reclaimed on demand; on Linux it is `MemAvailable`, which includes them.)
"""
function available_memory()
    free = Int(Sys.free_memory())
    Sys.isapple() || return free
    level = _macos_memory_level()
    level === nothing && return free
    return max(free, Int(Sys.total_memory()) ÷ 100 * level)
end

# The percentage of memory that macOS considers available (`sysctl kern.memorystatus_level`), or
# `nothing` if it cannot be read.
function _macos_memory_level()
    level = Ref{Cint}(0)
    size = Ref{Csize_t}(sizeof(Cint))
    rc = ccall(
        :sysctlbyname,
        Cint,
        (Cstring, Ptr{Cint}, Ptr{Csize_t}, Ptr{Cvoid}, Csize_t),
        "kern.memorystatus_level",
        level,
        size,
        C_NULL,
        0,
    )
    return (rc == 0 && 0 <= level[] <= 100) ? Int(level[]) : nothing
end

# The fraction of its host's free memory that each Distributed worker may use: 1/(number of
# workers on the host).
function _worker_memory_shares()
    ids = Distributed.workers()
    hosts = Dict(w => Distributed.remotecall_fetch(gethostname, w) for w in ids)
    count = Dict{String,Int}()
    for h in values(hosts)
        count[h] = get(count, h, 0) + 1
    end
    return Dict(w => 1 / count[hosts[w]] for w in ids)
end

# The free memory of each host of the Distributed workers, and the number of workers on it.
function _host_memory()
    out = Dict{String,Tuple{Int,Int}}()
    for w in Distributed.workers()
        h, m = Distributed.remotecall_fetch(() -> (gethostname(), available_memory()), w)
        _, k = get(out, h, (m, 0))
        out[h] = (m, k + 1)
    end
    return out
end

function _memory_error(what, need, avail, est)
    largest = isempty(est.pairs) ? 0 : first(est.pairs)
    return ArgumentError(
        "the sweep needs an estimated $(_format_bytes(need)) of memory $what (shared data " *
        "$(_format_bytes(est.shared)), largest pair $(_format_bytes(largest))), but only " *
        "$(_format_bytes(avail)) are available; use smaller ranges of n and i, fewer threads " *
        "or workers, or pass check_memory=false to skip this check",
    )
end

"""
    check_sweep_memory(est; distributed=false, share=1.0)

Throws an `ArgumentError` if the estimate `est` of [`sweep_memory_estimate`](@ref) exceeds the
available memory. For a computation in this process the estimate `est.total` is compared with
`share` times [`available_memory`](@ref). With `distributed = true` (the pairs run on the
Distributed workers, one at a time per worker), this process needs `max(est.phase0, est.shared)`,
and each host needs the sum of the `k` largest pair estimates for its `k` workers (plus
`est.nablas` for the host of this process), compared with the free memory of the host.
"""
function check_sweep_memory(est::NamedTuple; distributed::Bool=false, share::Real=1.0)
    if !distributed
        avail = floor(Int, share * available_memory())
        est.total <= avail || throw(_memory_error("in this process", est.total, avail, est))
        return nothing
    end
    avail = available_memory()
    here = max(est.phase0, est.shared)
    here <= avail || throw(_memory_error("in this process", here, avail, est))
    me = gethostname()
    for (h, (free, k)) in _host_memory()
        need = sum(est.pairs[1:min(k, length(est.pairs))]; init=0)
        h == me && (need += est.nablas)
        need <= free || throw(_memory_error("on host $h ($k workers)", need, free, est))
    end
    return nothing
end

"""
    check_worker_memory(ests)

For a sweep over several groups of rings, one group at a time per Distributed worker: throws an
`ArgumentError` if the largest total estimate `est.total` (see [`sweep_memory_estimate`](@ref))
exceeds the available memory of a host divided by the number of workers on it.
"""
function check_worker_memory(ests::AbstractVector{<:NamedTuple})
    isempty(ests) && return nothing
    est = ests[argmax([e.total for e in ests])]
    for (h, (free, k)) in _host_memory()
        avail = free ÷ k
        est.total <= avail || throw(
            _memory_error("on a worker of host $h ($k workers)", est.total, avail, est)
        )
    end
    return nothing
end
