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

# Steps 3 (O_K), 4 and 5.
function _g_side(prm, R, ed, lam, i; threaded::Bool=true)
    genv = GEnvelope(prm, R, ed, lam; threaded=threaded)              # Step 3 (O_K)
    wu = bk_unit_u(genv)                                              # Step 4: w(u)
    wv = bk_unit_v(genv, wu)                                          # Step 4: w(v)
    vi = bk_unit_power(genv, wv, i)                                   # w(v^i)
    return nabla_OK_matrix(genv, vi; threaded=threaded)               # Step 5
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
    syntomic_square(W, Rf; descent=false) -> (syn0, syn1, exponents, intermediates)

Steps 7–10 and the assembly of `syn^0`, `syn^1` over `Z_p` ([AKN, §7.1, pp. 56–57]) from the
W-matrices `W` of [`syntomic_wmatrices`](@ref). `Rf` is the Galois ring of the target residue field;
with `descent = true` the W-matrices were computed with `f = 1` arithmetic (see
[`descent_ring`](@ref)) and are expanded at the end. The intermediates are the `Z_p`-matrices of
Steps 5–10 for the target residue field.
"""
function syntomic_square(W::NamedTuple, Rf::GaloisRing; descent::Bool=false)
    p = Rf.p
    fW = W.R.f                                            # f used for the W-matrices
    red = zp_expand(W.red)
    redN = zp_expand(W.redN)
    nablaOK = zp_expand(W.nablaOK)
    can0 = zp_expand(W.can0)
    can1 = zp_expand(W.can1)
    phi = zp_expand(W.Phi; semilinear=true)
    nablaR, e_red = nabla_R_matrix(red, redN, nablaOK, fW, p)          # Step 7
    nablaN, e_can1 = nabla_N_matrix(can0, can1, nablaR, fW, p)         # Step 8
    phiN, e_N = phi_nabla_matrix(nablaR, phi, nablaN, fW, p)           # Step 10
    if descent
        expand(A; semilinear=false) = zp_expand_descent(Rf, A; semilinear=semilinear)
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
alone at the precision `max(v_p det red, v_p det can1) + n_f + 3`, where `n_f` is the number of
f-variables ([AKN, Lemmas 4.63–4.64, p. 45]). They determine the precision lost in Steps 7 and 8
exactly.
"""
function exponent_phase(Xc::FiniteChainRing, i::Integer; threaded::Bool=true)
    p, n = Xc.p, Xc.n
    prm = SyntomicParams(p, Xc.f, n, i)
    prm.L <= 1 && return (0, 0)
    D = max(det_valuation_red(p, n, i), det_valuation_can1(n, i))
    N0 = D + prm.nf + 3
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
[`adaptive_working_precision`](@ref)). The exponents needed by `:exponent` and `:adaptive` are
computed for [`computing_ring`](@ref)`(X, descent)`.
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
    e_red, e_can1 = exponent_phase(computing_ring(X, descent), i; threaded=threaded)
    precision_policy == :exponent &&
        return exponent_working_precision(p, n, i, e_red, e_can1)
    Tg = Tguess === nothing ? initial_adaptive_guess(p, n, i) : Tguess
    return adaptive_working_precision(p, n, i, Tg, e_red, e_can1)
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
    i >= 1 || throw(ArgumentError("the weight i must be at least 1, got $i"))
    L = X.n * i
    if L == 1                                             # F^{[1,0]} = 0: the complex is zero
        Zn, _ = residue_ring(ZZ, ZZ(X.p)^max(N, 1))
        z = PMatrix(zero_matrix(Zn, 0, 0), N)
        return SyntomicComplex(
            X, i, N, z, z, (red=0, can1=0, nablaN=0), policy, Dict{Symbol,Any}()
        )
    end
    N >= 2 || throw(PrecisionExhausted("the working precision N = $N is too small"))
    Rf = GaloisRing(X.p, X.htilde, N)
    t0 = time()
    W = syntomic_wmatrices(Xc, i, N; threaded=threaded, nablaOK=nablaOK)
    timings[:wmatrices] = time() - t0
    t0 = time()
    syn0, syn1, exps, inter = syntomic_square(W, Rf; descent=Xc.f != X.f)
    timings[:square] = time() - t0
    return SyntomicComplex(X, i, N, syn0, syn1, exps, policy, inter)
end

"""
    syntomic_cohomology(X::FiniteChainRing, i; precision_policy=:adaptive, N=nothing,
                        threaded=true, check=true, descent=true) -> SyntomicCohomology

The cohomology `H^1`, `H^2` of `Z_p(i)(X)` for the finite chain ring `X = O_K/ϖ^n`
([AKN, §7.1, pp. 53–57; §7.2, pp. 57–58]); `H^0 = 0`.

Precision policies (see [`working_precision`](@ref)); every result is certified by
[AKN, Lemma 7.1, p. 57] and [AKN, Remark 7.3, p. 58]:
- `:adaptive` (default): start from a guess for the target precision and double it until the
  result is certified; the last attempt uses the `:exponent` precision;
- `:exponent`: the bound of [AKN, §7.1, Step 1] with the exact exponents of `red` and `can`
  (computed first);
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
    p, n = X.p, X.n
    Xc = computing_ring(X, descent)
    T = target_precision(p, n, i)
    e_red, e_can1 = exponent_phase(Xc, i; threaded=threaded)
    Tg = initial_adaptive_guess(p, n, i)
    restarts = 0
    while true
        final = Tg >= T
        NN = if final
            exponent_working_precision(p, n, i, e_red, e_can1)
        else
            adaptive_working_precision(p, n, i, Tg, e_red, e_can1)
        end
        try
            C = _syntomic_complex(
                X, Xc, i, NN; policy=:adaptive, threaded=threaded, timings=timings
            )
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
