# The algorithm of [AKN, §7.1, pp. 53–57], assembled from the steps.
#
#   Step 1  working precision                      precision.jl, `working_precision`
#   Step 2  linear bases, W(F_q) = Z_p[x]/(h̃)       params.jl, arith/galois_ring.jl
#   Step 3  prismatic envelopes                    step3/*.jl
#   Step 4  the unit w(v)                          step4_orientation.jl
#   Step 5  ∇ for O_K                              step5_nabla_ok.jl
#   Step 6  red and red^∇                          step6_reduction.jl
#   Step 7  ∇ for R                                step7_nabla_r.jl
#   Step 8  N^{≥i}∇                                step8_nabla_n.jl, step10_can_phi_nabla.jl
#   Step 9  can - φ                                step9_can_phi.jl, step10_can_phi_nabla.jl
#   Step 10 can - φ^∇                              step10_can_phi_nabla.jl
#   Return  syn^0, syn^1 and their cohomology      cohomology.jl

"""
    descent_ring(X::FiniteChainRing) -> Union{Nothing,FiniteChainRing}

A presentation of `X` over `Z_p`, if there is an evident one. Let `m = max(⌈n/e⌉, 2)`. If `f > 1`
and the normalized Eisenstein polynomial `E/u` (`u = E(0)/p`) of `X` is congruent modulo `p^m` to
a polynomial `E'` with coefficients in `Z_p`, returns `X_0 = Z_p[z]/(E', z^n)` (with `f = 1`);
otherwise `nothing`. `E'` is again Eisenstein, and
`W(F_q) ⊗ X_0 = W(F_q)[z]/(E', z^n) = W(F_q)[z]/(E, z^n) = X`, because `p^m ∈ (E', z^n)` and `R`
depends only on `E` up to a unit ([AKN, §7.1, Input, p. 53]). The W-matrices of the algorithm for
`E'` then have entries in `Z_p`, so they can be computed with `f = 1` arithmetic for `X_0` and
expanded afterwards (see [`zp_expand_descent`](@ref)).

The answer depends only on `X`, not on a working precision.
"""
function descent_ring(X::FiniteChainRing)
    X.f == 1 && return nothing
    m = required_input_precision(X.n, X.e)
    R = GaloisRing(X.p, X.htilde, m)
    coeffs = ws_coefficients(normalize_eisenstein(R, X, X.e + 1))
    all(all(iszero, c[2:end]) for c in coeffs) || return nothing
    E1 = [BigInt[_balanced_residue(c[1], R.pN)] for c in coeffs]
    return finite_chain_ring(X.p, [0, 1], E1, X.n; Eprec=m, label=X.label)
end

"""
    computing_ring(X, descent::Bool) -> FiniteChainRing

The ring whose W-matrices are computed for `X`: [`descent_ring`](@ref)`(X)` if `descent` is true
and it exists, and `X` itself otherwise.
"""
computing_ring(X::FiniteChainRing, descent::Bool) =
    descent ? something(descent_ring(X), X) : X

"""
    syntomic_wmatrices(X, i, N; threaded=true, nablaOK=nothing) -> NamedTuple

Steps 3–6, 8 and 9 over `W(F_q)/p^N` ([AKN, §7.1, pp. 55–57]): the W-matrices `red`, `redN`
(Step 6), `can0`, `Phi` (Step 9), `can1` (the `can` on the `∇`-terms, Step 8) and `nablaOK`
(Steps 4–5). The f-envelope and the g-envelope are independent and are built concurrently when
`threaded` is true. A precomputed `∇_{O_K}` ([`PlainWMatrix`](@ref), e.g. from a sweep) can be
passed as `nablaOK`.
"""
function syntomic_wmatrices(
    X::FiniteChainRing,
    i::Integer,
    N::Integer;
    threaded::Bool=true,
    nablaOK::Union{Nothing,PlainWMatrix}=nothing,
)
    prm = SyntomicParams(X.p, X.f, X.n, i)
    R = GaloisRing(X.p, X.htilde, N)
    ed = EisensteinData(R, X, prm.L)                                  # Input normalization
    lam = LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)                  # Step 3, Lemma 3.24
    gside = if nablaOK === nothing
        () -> _g_side(prm, R, ed, lam, i; threaded=threaded)          # Steps 3–5 (O_K)
    else
        size(nablaOK) == (prm.L - 1, prm.L - 1) ||
            throw(DimensionMismatch("the given ∇_OK has size $(size(nablaOK))"))
        () -> from_plain(R, nablaOK)
    end
    nab, fmats = concurrently(
        gside,
        () -> _f_side(prm, R, ed, lam, i; threaded=threaded);        # Steps 3, 6, 8, 9 (R)
        threaded=threaded && nablaOK === nothing,
    )
    return (; prm, R, fmats..., nablaOK=nab)
end

# The W-matrices of the f-side (Steps 3, 6, 8 and 9 for R = O_K/ϖ^n) of the ring X at precision N,
# with the parameters and the Galois ring.
function _f_wmatrices(X::FiniteChainRing, i::Integer, N::Integer; threaded::Bool=true)
    prm = SyntomicParams(X.p, X.f, X.n, i)
    R = GaloisRing(X.p, X.htilde, N)
    ed = EisensteinData(R, X, prm.L)
    lam = LambdaData(ed, max(prm.nf, 2) - 2)
    return (; prm, R, _f_side(prm, R, ed, lam, i; threaded=threaded)...)
end

# ∇_{O_K} (Steps 3–5 for O_K) of the ring X at precision N, as plain data.
function _plain_nabla_OK(X::FiniteChainRing, i::Integer, N::Integer; threaded::Bool=true)
    prm = SyntomicParams(X.p, X.f, X.n, i)
    R = GaloisRing(X.p, X.htilde, N)
    ed = EisensteinData(R, X, prm.L)
    lam = LambdaData(ed, max(prm.ng, 2) - 2)
    return to_plain(_g_side(prm, R, ed, lam, i; threaded=threaded))
end

# Steps 3 (O_K), 4 and 5. The transposed chain of Step 5 does not depend on w(v^i), so the two are
# computed concurrently when `threaded` (the chain is short of parallel work, a sequence of L
# small steps).
function _g_side(prm, R, ed, lam, i; threaded::Bool=true)
    genv = GEnvelope(prm, R, ed, lam; threaded=threaded)              # Step 3 (O_K)
    wu = bk_unit_u(genv)                                              # Step 4: w(u)
    vi, chain = concurrently(
        () -> bk_unit_v_power(genv, wu, i),                           # Step 4: w(v^i)
        () -> theta_chain(genv; threaded=threaded);                   # Step 5
        threaded=threaded,
    )
    return nabla_OK_matrix(genv, vi; threaded=threaded, chain=chain)  # Step 5
end

# Steps 3 (R = O_K/ϖ^n), 6, 8 (can on the ∇-terms) and 9.
function _f_side(prm, R, ed, lam, i; threaded::Bool=true)
    fenv = FEnvelope(prm, R, ed, lam)                                 # Step 3 (R)
    red = reduction_matrix(fenv; threaded=threaded)                   # Step 6
    redN = reduction_matrix_nabla(fenv; threaded=threaded)            # Step 6
    can0 = can_matrix(fenv, i; threaded=threaded)                     # Step 9
    can1 = can_nabla_matrix(fenv, i; threaded=threaded)               # Step 8
    Phi = phi_matrix(fenv; threaded=threaded)                         # Step 9
    return (; red, redN, can0, can1, Phi)
end

"""
    syntomic_square(W, Rf; descent=false, threaded=true) -> (syn0, syn1, exponents, intermediates)

Steps 7–10 and the assembly of `syn^0`, `syn^1` over `Z_p` ([AKN, §7.1, pp. 56–57]) from the
W-matrices `W` of [`syntomic_wmatrices`](@ref). `Rf` is the Galois ring of the target residue field;
with `descent = true` the W-matrices were computed with `f = 1` arithmetic (see
[`descent_ring`](@ref)) and are expanded at the end. The intermediates are the `Z_p`-matrices of
Steps 5–10 for the target residue field. With `threaded = true` the expansions, the exponents
and the triangular solves are computed in parallel (with the same result).
"""
function syntomic_square(
    W::NamedTuple, Rf::GaloisRing; descent::Bool=false, threaded::Bool=true
)
    p = Rf.p
    fW = W.R.f                                            # f used for the W-matrices
    ex(A; semilinear=false) = zp_expand(A; semilinear=semilinear, threaded=threaded)
    red, redN, nablaOK = ex(W.red), ex(W.redN), ex(W.nablaOK)
    can0, can1, phi = ex(W.can0), ex(W.can1), ex(W.Phi; semilinear=true)
    # Step 7 and the exponent of can1 (needed in Step 8) are independent. The exponents are
    # computed by serial Smith normal forms; for large matrices they run concurrently with other
    # work (for small ones this does not pay, see `PARALLEL_SOLVE_WORK`).
    function step7()                                                  # Step 7
        e = local_exponent(red.M, p, red.prec)
        return nabla_R_matrix(red, redN, nablaOK, fW, p; e_red=e, threaded=threaded)
    end
    n = size(can1, 1)
    (nablaR, e_red), e_can1 = concurrently(
        step7,
        () -> local_exponent(can1.M, p, can1.prec);
        threaded=threaded && n^3 >= PARALLEL_SOLVE_WORK,
    )
    nablaN, _ = nabla_N_matrix(                                        # Step 8
        can0,
        can1,
        nablaR,
        fW,
        p;
        e_can1=e_can1,
        threaded=threaded,
    )
    e_N = local_exponent(nablaN.M, p, nablaN.prec)
    phiN, _ = phi_nabla_matrix(                                        # Step 10
        nablaR,
        phi,
        nablaN,
        fW,
        p;
        e_nablaN=e_N,
        threaded=threaded,
    )
    if descent
        function expand(A; semilinear=false)
            return zp_expand_descent(Rf, A; semilinear=semilinear, threaded=threaded)
        end
        red, redN, nablaOK = expand(red), expand(redN), expand(nablaOK)
        nablaR, nablaN = expand(nablaR), expand(nablaN)
        can0, can1 = expand(can0), expand(can1)
        phi, phiN = expand(phi; semilinear=true), expand(phiN; semilinear=true)
    end
    cmp = can_minus_phi(can0, phi)                                     # Step 9
    cmpN = can_minus_phi_nabla(can1, phiN)                             # Step 10
    syn0 = pm_vcat(cmp, nablaN)                                        # Return
    syn1 = pm_hcat(nablaR, pm_neg(cmpN))
    inter = Dict{Symbol,Any}(
        :red => red,
        :redN => redN,
        :nablaOK => nablaOK,
        :can0 => can0,
        :can1 => can1,
        :phi => phi,
        :nablaR => nablaR,
        :nablaN => nablaN,
        :canmphi => cmp,
        :canmphiN => cmpN,
    )
    return syn0, syn1, (red=e_red, can1=e_can1, nablaN=e_N), inter
end

"""
    exponent_phase(Xc, i; threaded=true) -> (e_red, e_can1)

The p-exponents of the cokernels of `red` and of `can` on the `∇`-terms (Steps 6 and 8) for the
ring `Xc` used in the computation (see [`computing_ring`](@ref)), computed from the f-envelope
alone ([AKN, Lemmas 4.63–4.64, p. 45]). They determine the precision lost in Steps 7 and 8
exactly.

The f-envelope is built at the precisions of `exponent_phase_precisions`, starting from a guess
and doubling it until both exponents are certified, i.e. every elementary divisor is nonzero
modulo the precision to which the matrix is known (then the valuations are exact, as in
[AKN, Lemma 7.1, p. 57]). The last precision exceeds `max(v_p det red, v_p det can1)` by the number
of digits lost in the f-envelope, so the exponents are always certified there.
"""
function exponent_phase(Xc::FiniteChainRing, i::Integer; threaded::Bool=true)
    prm = SyntomicParams(Xc.p, Xc.f, Xc.n, i)
    prm.L <= 1 && return (0, 0)
    precisions = exponent_phase_precisions(Xc.p, Xc.n, i; e=Xc.e)
    for N0 in precisions[1:(end - 1)]
        e = try
            _exponents_at_precision(Xc, prm, i, N0; threaded=threaded)
        catch err
            err isa PrecisionExhausted || rethrow()
            nothing
        end
        e === nothing || return e
    end
    return _exponents_at_precision(Xc, prm, i, last(precisions); threaded=threaded)
end

"""
    exponent_phase_precisions(p, n, i; e=1) -> Vector{Int}

The precisions at which [`exponent_phase`](@ref) builds the f-envelope of a ring with ramification
index `e`, in order: the guess `max(ê_red, ê_can1 + 1) + n_f + 4` from the estimates of
[`predicted_exponents`](@ref), doubled until it reaches `D + n_f + 3`, where
`D = max(v_p det red, v_p det can1)` bounds both exponents and `n_f` is the number of
f-variables ([AKN, Lemmas 4.63–4.64, p. 45]).
"""
function exponent_phase_precisions(p::Integer, n::Integer, i::Integer; e::Integer=1)
    nf = num_f_variables(p, i)
    Nmax = max(det_valuation_red(p, n, i), det_valuation_can1(n, i)) + nf + 3
    e_red, e_can1 = predicted_exponents(p, n, e, i)
    N0 = min(max(e_red, e_can1 + 1) + nf + 4, Nmax)
    out = [N0]
    while last(out) < Nmax
        push!(out, min(2 * last(out), Nmax))
    end
    return out
end

# The exponents of red and can1 from the f-envelope at precision N0; throws PrecisionExhausted if
# an elementary divisor is ≡ 0 modulo the precision of the matrix.
function _exponents_at_precision(
    Xc::FiniteChainRing, prm::SyntomicParams, i::Integer, N0::Integer; threaded::Bool=true
)
    p = Xc.p
    R = GaloisRing(p, Xc.htilde, N0)
    ed = EisensteinData(R, Xc, prm.L)
    lam = LambdaData(ed, max(prm.nf, 2) - 2)
    fenv = FEnvelope(prm, R, ed, lam)
    red = zp_expand(reduction_matrix(fenv; threaded=threaded))
    can1 = zp_expand(can_nabla_matrix(fenv, i; threaded=threaded))
    return local_exponent(red.M, p, red.prec), local_exponent(can1.M, p, can1.prec)
end

const PRECISION_POLICIES = (:apriori, :exponent, :adaptive)

function _check_policy(policy::Symbol, allowed=PRECISION_POLICIES)
    policy in allowed || throw(
        ArgumentError(
            "unknown or unsupported precision policy :$policy (use one of $(join(allowed, ", ")))",
        ),
    )
    return policy
end

"""
    working_precision(X, i; precision_policy=:apriori, Tguess=nothing, threaded=true,
                      descent=true) -> Int

The p-adic working precision N of [AKN, §7.1, Step 1, p. 54] for the given policy
(see [`apriori_working_precision`](@ref), [`exponent_working_precision`](@ref) and
[`adaptive_working_precision`](@ref)). The exponents needed by `:exponent` are computed for
[`computing_ring`](@ref)`(X, descent)`; for `:adaptive` this is the precision of the first attempt,
which uses the estimates of [`predicted_exponents`](@ref) instead
([`predicted_adaptive_working_precision`](@ref)).
"""
function working_precision(
    X::FiniteChainRing,
    i::Integer;
    precision_policy::Symbol=:apriori,
    Tguess::Union{Nothing,Integer}=nothing,
    threaded::Bool=true,
    descent::Bool=true,
)
    _check_policy(precision_policy)
    p, n = X.p, X.n
    precision_policy == :apriori && return apriori_working_precision(p, n, i)
    Xc = computing_ring(X, descent)
    if precision_policy == :exponent
        e_red, e_can1 = exponent_phase(Xc, i; threaded=threaded)
        return exponent_working_precision(p, n, i, e_red, e_can1)
    end
    Tg = Tguess === nothing ? initial_adaptive_guess(p, n, i) : Tguess
    Tg >= target_precision(p, n, i) && return working_precision(
        X, i; precision_policy=:exponent, threaded=threaded, descent=descent
    )
    return predicted_adaptive_working_precision(p, n, Xc.e, i, Tg)
end

"""
    syntomic_complex(X, i; N=nothing, precision_policy=:exponent, threaded=true, descent=true,
                     nablaOK=nothing) -> SyntomicComplex

The syntomic complex `syn^0`, `syn^1` computing `Z_p(i)(X)` for the finite chain ring
`X = O_K/ϖ^n` ([AKN, §7.1, pp. 53–57]). The working precision `N` defaults to the one of the given
policy. With `descent = true` (default), `f = 1` arithmetic is used when `X` has a presentation
over `Z_p` (see [`descent_ring`](@ref)); the matrices then refer to that presentation.
"""
function syntomic_complex(
    X::FiniteChainRing,
    i::Integer;
    N::Union{Nothing,Integer}=nothing,
    precision_policy::Symbol=:exponent,
    threaded::Bool=true,
    descent::Bool=true,
    timings::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    nablaOK::Union{Nothing,PlainWMatrix}=nothing,
)
    _check_policy(precision_policy)
    Xc = computing_ring(X, descent)
    if N === nothing && precision_policy == :exponent && nablaOK === nothing
        C, t = only(_exponent_complexes([X], Xc, i; policy=:exponent, threaded=threaded))
        merge!(timings, t)
        return C
    end
    NN, policy = if N === nothing
        NP = working_precision(
            Xc, i; precision_policy=precision_policy, threaded=threaded, descent=false
        )
        (NP, precision_policy)
    else
        (Int(N), :fixed)
    end
    return _syntomic_complex(
        X, Xc, i, NN; policy=policy, threaded=threaded, timings=timings, nablaOK=nablaOK
    )
end

# The syntomic complex of X, computed at precision N from the W-matrices of the ring Xc (X itself
# or its descent, see `computing_ring`).
function _syntomic_complex(
    X::FiniteChainRing,
    Xc::FiniteChainRing,
    i::Integer,
    N::Integer;
    policy::Symbol,
    threaded::Bool=true,
    timings::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    nablaOK::Union{Nothing,PlainWMatrix}=nothing,
)
    C, t = only(
        _syntomic_complexes(
            [X], Xc, i, N; policy=policy, threaded=threaded, nablaOK=nablaOK
        ),
    )
    merge!(timings, t)
    return C
end

"""
    _syntomic_complexes(Xs, Xc, i, N; policy, threaded=true, nablaOK=nothing)
        -> Vector{Tuple{SyntomicComplex,Dict{Symbol,Float64}}}

The syntomic complexes, with their timings, of the rings `Xs` whose computations all use the ring
`Xc` (see [`computing_ring`](@ref)), e.g. `W_n(F_q)` for several `q = p^f` and the same `n`: the
W-matrices of `Xc` at precision `N` (Steps 3–6, 8 and 9) are computed once, and only Steps 7–10
and the expansion to `Z_p`-matrices ([`syntomic_square`](@ref)) are repeated for each ring of
`Xs` ([AKN, §7.1, pp. 53–57]).
"""
function _syntomic_complexes(
    Xs::AbstractVector{FiniteChainRing},
    Xc::FiniteChainRing,
    i::Integer,
    N::Integer;
    policy::Symbol,
    threaded::Bool=true,
    nablaOK::Union{Nothing,PlainWMatrix}=nothing,
)
    i >= 1 || throw(ArgumentError("the weight i must be at least 1, got $i"))
    all(X -> X.n == Xc.n, Xs) || throw(
        ArgumentError("the rings must have the power n = $(Xc.n) of the computing ring")
    )
    if Xc.n * i == 1                                      # F^{[1,0]} = 0: the complex is zero
        return map(Xs) do X
            Zn, _ = residue_ring(ZZ, ZZ(X.p)^max(N, 1))
            z = PMatrix(zero_matrix(Zn, 0, 0), N)
            exps = (red=0, can1=0, nablaN=0)
            (
                SyntomicComplex(X, i, N, z, z, exps, policy, Dict{Symbol,Any}()),
                Dict{Symbol,Float64}(),
            )
        end
    end
    N >= 2 || throw(PrecisionExhausted("the working precision N = $N is too small"))
    t0 = time()
    W = syntomic_wmatrices(Xc, i, N; threaded=threaded, nablaOK=nablaOK)
    return _squares(Xs, Xc, i, N, W, Dict(:wmatrices => time() - t0); policy, threaded)
end

# Steps 7–10 and the assembly for each ring of Xs from the W-matrices W of Xc at precision N.
function _squares(Xs, Xc, i, N, W, timings0::Dict{Symbol,Float64}; policy, threaded)
    return map(Xs) do X
        timings = copy(timings0)
        Rf = GaloisRing(X.p, X.htilde, N)
        t0 = time()
        syn0, syn1, exps, inter = syntomic_square(
            W, Rf; descent=Xc.f != X.f, threaded=threaded
        )
        timings[:square] = time() - t0
        (SyntomicComplex(X, i, N, syn0, syn1, exps, policy, inter), timings)
    end
end

# f() computed speculatively: its value, or the exception it raised (to be rethrown only if the
# value is needed). Interrupts are not deferred.
function _speculatively(f)
    try
        return f()
    catch err
        err isa InterruptException && rethrow()
        return _task_exception(err, catch_backtrace())
    end
end

"""
    _exponent_complexes(Xs, Xc, i; policy=:exponent, threaded=true, Nspec=nothing)
        -> Vector{Tuple{SyntomicComplex,Dict{Symbol,Float64}}}

The syntomic complexes of the rings `Xs` (with the computing ring `Xc`) at the working precision
`N` of policy `:exponent`, which needs the exponents of [`exponent_phase`](@ref). The envelope of
`O_K` (Steps 3–5), the larger part of the work, is computed without waiting for them: at the
precision `Nspec` of [`predicted_exponent_working_precision`](@ref), concurrently with the
exponents and the f-side W-matrices at `N` (with `threaded`; the envelope of `O_K` is built on the
calling task, so that an interrupt reaches it directly). If `N ≤ Nspec`, its `∇_{O_K}` is used,
reduced to precision `N`; otherwise it is discarded (together with any error it raised) and
`∇_{O_K}` is recomputed at `N`. Without threads the same rule is applied (the envelope of `O_K` is
computed at `Nspec` when `N ≤ Nspec`), so the results do not depend on `threaded`.
"""
function _exponent_complexes(
    Xs::AbstractVector{FiniteChainRing},
    Xc::FiniteChainRing,
    i::Integer;
    policy::Symbol=:exponent,
    threaded::Bool=true,
    Nspec::Union{Nothing,Integer}=nothing,
)
    i >= 1 || throw(ArgumentError("the weight i must be at least 1, got $i"))
    p, n = Xc.p, Xc.n
    if n * i <= 1
        N = exponent_working_precision(p, n, i, 0, 0)
        return _syntomic_complexes(Xs, Xc, i, N; policy=policy, threaded=threaded)
    end
    Ns = something(Nspec, predicted_exponent_working_precision(p, n, Xc.e, i))
    function gside(N)
        return (t = time();
        nab = _plain_nabla_OK(Xc, i, N; threaded=threaded);
        (nab, time() - t))
    end
    function fside()
        t0 = time()
        N = exponent_working_precision(p, n, i, exponent_phase(Xc, i; threaded=threaded)...)
        t1 = time()
        N >= 2 || throw(PrecisionExhausted("the working precision N = $N is too small"))
        return N, _f_wmatrices(Xc, i, N; threaded=threaded), (t1 - t0, time() - t1)
    end
    t0 = time()
    if threaded && threads_available()
        (N, F, tf), spec = concurrently(fside, () -> _speculatively(() -> gside(Ns)))
        if N > Ns                                         # the estimate was too small
            nab, tg = gside(N)
        else
            spec isa Exception && throw(spec)
            nab, tg = spec
        end
    else
        N, F, tf = fside()
        nab, tg = gside(N <= Ns ? Ns : N)
    end
    W = (; F..., nablaOK=from_plain(F.R, nab))
    timings = Dict(
        :wmatrices => time() - t0, :exponents => tf[1], :fside => tf[2], :gside => tg
    )
    return _squares(Xs, Xc, i, N, W, timings; policy=policy, threaded=threaded)
end

"""
    syntomic_cohomology(X::FiniteChainRing, i; precision_policy=:adaptive, N=nothing,
                        threaded=true, check=true, descent=true) -> SyntomicCohomology

The cohomology `H^1`, `H^2` of `Z_p(i)(X)` for the finite chain ring `X = O_K/ϖ^n`
([AKN, §7.1, pp. 53–57; §7.2, pp. 57–58]); `H^0 = 0`.

Precision policies (see [`working_precision`](@ref)); every result is certified by
[AKN, Lemma 7.1, p. 57] and [AKN, Remark 7.3, p. 58]:
- `:adaptive` (default): start from a guess for the target precision
  ([`initial_adaptive_guess`](@ref)) and double it until the result is certified; the attempts
  use estimates of the exponents of `red` and `can` ([`predicted_exponents`](@ref)), and the last
  attempt uses the `:exponent` precision;
- `:exponent`: the bound of [AKN, §7.1, Step 1] with the exact exponents of `red` and `can`
  (computed concurrently with the envelope of `O_K`, see the [Precision](@ref precision) page);
- `:apriori`: the working precision of [AKN, §7.1, Step 1].
If `N` is given, the computation runs at that precision and the policy is recorded as `:fixed`.
"""
function syntomic_cohomology(
    X::FiniteChainRing,
    i::Integer;
    precision_policy::Symbol=:adaptive,
    N::Union{Nothing,Integer}=nothing,
    threaded::Bool=true,
    check::Bool=true,
    descent::Bool=true,
)
    _check_policy(precision_policy)
    timings = Dict{Symbol,Float64}()
    if precision_policy != :adaptive || N !== nothing
        C = syntomic_complex(
            X,
            i;
            N=N,
            precision_policy=precision_policy,
            threaded=threaded,
            descent=descent,
            timings=timings,
        )
        return syntomic_cohomology(C; check=check, threaded=threaded, timings=timings)
    end
    return _adaptive_cohomology(
        X,
        computing_ring(X, descent),
        i;
        Tguess=initial_adaptive_guess(X.p, X.n, i),
        threaded=threaded,
        check=check,
    )
end

# The loop of policy :adaptive, from the guess Tguess of the target precision: the attempts use
# the estimated exponents of `predicted_exponents`; the last one (Tguess = T) uses the exact
# exponents, computed concurrently with the envelope of O_K (see `_exponent_complexes`).
function _adaptive_cohomology(
    X::FiniteChainRing,
    Xc::FiniteChainRing,
    i::Integer;
    Tguess::Integer,
    threaded::Bool=true,
    check::Bool=true,
)
    p, n = X.p, X.n
    timings = Dict{Symbol,Float64}()
    T = target_precision(p, n, i)
    Tg = min(max(Tguess, 1), T)
    restarts = 0
    while true
        final = Tg >= T
        try
            C = if final
                Cf, t = only(
                    _exponent_complexes([X], Xc, i; policy=:adaptive, threaded=threaded)
                )
                merge!(timings, t)
                Cf
            else
                NN = predicted_adaptive_working_precision(p, n, Xc.e, i, Tg)
                _syntomic_complex(
                    X, Xc, i, NN; policy=:adaptive, threaded=threaded, timings=timings
                )
            end
            return syntomic_cohomology(
                C; check=check, threaded=threaded, restarts=restarts, timings=timings
            )
        catch err
            (err isa PrecisionExhausted && !final) || rethrow()
            restarts += 1
            Tg = min(2 * Tg, T)
        end
    end
end
