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
    descent_ring(X::FiniteChainRing, N) -> Union{Nothing,FiniteChainRing}

If `f > 1` and the normalized Eisenstein polynomial of `X` has coefficients in `Z_p`, the ring with
the same (normalized) Eisenstein polynomial over `Z_p` (`f = 1`); otherwise `nothing`. In the first
case every W-matrix of the algorithm has entries in `Z_p`, so it can be computed with `f = 1`
arithmetic and expanded afterwards (see [`zp_expand_descent`](@ref)).
"""
function descent_ring(X::FiniteChainRing, N::Integer)
    X.f == 1 && return nothing
    R = GaloisRing(X.p, X.htilde, N)
    E = normalize_eisenstein(eisenstein_series(R, X, X.e + 1))
    coeffs = ws_coefficients(E)
    all(all(iszero, c[2:end]) for c in coeffs) || return nothing
    E1 = [BigInt[_balanced_residue(c[1], R.pN)] for c in coeffs]
    return finite_chain_ring(X.p, [0, 1], E1, X.n; label=X.label)
end

"""
    syntomic_wmatrices(X, i, N; threaded=true) -> NamedTuple

Steps 3–6, 8 and 9 over `W(F_q)/p^N`: the W-matrices `red`, `redN` (Step 6), `can0`, `Phi`
(Step 9), `can1` (the `can` on the `∇`-terms, Step 8) and `nablaOK` (Steps 4–5). The f-envelope
and the g-envelope are independent and are built concurrently when `threaded` is true.
"""
function syntomic_wmatrices(X::FiniteChainRing, i::Integer, N::Integer; threaded::Bool=true)
    prm = SyntomicParams(X.p, X.f, X.n, i)
    R = GaloisRing(X.p, X.htilde, N)
    ed = EisensteinData(R, X, prm.L)                                  # Input normalization
    lam = LambdaData(ed, max(prm.nf, prm.ng, 2) - 2)                  # Step 3, Lemma 3.24
    gtask = spawn_if(threaded, () -> _g_side(prm, R, ed, lam, i; threaded=threaded))
    fenv = FEnvelope(prm, R, ed, lam)                                 # Step 3 (R = O_K/ϖ^n)
    red = reduction_matrix(fenv; threaded=threaded)                   # Step 6
    redN = reduction_matrix_nabla(fenv; threaded=threaded)            # Step 6
    can0 = can_matrix(fenv, i; threaded=threaded)                     # Step 9
    can1 = can_nabla_matrix(fenv, i; threaded=threaded)               # Step 8
    Phi = phi_matrix(fenv; threaded=threaded)                         # Step 9
    nablaOK = fetch(gtask)                                            # Steps 3–5 (O_K)
    return (; prm, R, red, redN, can0, can1, Phi, nablaOK)
end

# Steps 3 (O_K), 4 and 5.
function _g_side(prm, R, ed, lam, i; threaded::Bool=true)
    genv = GEnvelope(prm, R, ed, lam)                                 # Step 3 (O_K)
    wu = bk_unit_u(genv)                                              # Step 4: w(u)
    wv = bk_unit_v(genv, wu)                                          # Step 4: w(v)
    vi = bk_unit_power(genv, wv, i)                                   # w(v^i)
    return nabla_OK_matrix(genv, vi; threaded=threaded)               # Step 5
end

"""
    syntomic_square(W, Rf; descent=false) -> (syn0, syn1, exponents, intermediates)

Steps 7–10 and the assembly of `syn^0`, `syn^1` over `Z_p` ([AKN, §7.1, pp. 56–57]) from the
W-matrices `W` of [`syntomic_wmatrices`](@ref). `Rf` is the Galois ring of the target residue field;
with `descent = true` the W-matrices were computed with `f = 1` arithmetic (see
[`descent_ring`](@ref)) and are expanded at the end.
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
    exponent_phase(X, i; threaded=true) -> (e_red, e_can1)

The p-exponents of the cokernels of `red` and of `can` on the `∇`-terms (Steps 6 and 8), computed
from the f-envelope alone at the precision `max(v_p det red, v_p det can1) + M_f + 3`
([AKN, Lemmas 4.63–4.64, p. 45]). They determine the precision lost in Steps 7 and 8 exactly.
"""
function exponent_phase(X::FiniteChainRing, i::Integer; threaded::Bool=true)
    p, n = X.p, X.n
    prm = SyntomicParams(p, X.f, n, i)
    prm.L <= 1 && return (0, 0)
    D = max(det_valuation_red(p, n, i), det_valuation_can1(n, i))
    N0 = D + prm.nf + 3
    Xc = something(descent_ring(X, N0), X)
    prm = SyntomicParams(p, Xc.f, n, i)
    R = GaloisRing(p, Xc.htilde, N0)
    ed = EisensteinData(R, Xc, prm.L)
    lam = LambdaData(ed, max(prm.nf, 2) - 2)
    fenv = FEnvelope(prm, R, ed, lam)
    red = zp_expand(reduction_matrix(fenv; threaded=threaded))
    can1 = zp_expand(can_nabla_matrix(fenv, i; threaded=threaded))
    return local_exponent(red.M, p, red.prec), local_exponent(can1.M, p, can1.prec)
end

"""
    working_precision(X, i; precision_policy=:apriori, Tguess=nothing, threaded=true) -> Int

The p-adic working precision N of [AKN, §7.1, Step 1, p. 54] for the given policy
(see [`apriori_working_precision`](@ref), [`exponent_working_precision`](@ref) and
[`adaptive_working_precision`](@ref)).
"""
function working_precision(
    X::FiniteChainRing,
    i::Integer;
    precision_policy::Symbol=:apriori,
    Tguess::Union{Nothing,Integer}=nothing,
    threaded::Bool=true,
)
    p, n = X.p, X.n
    precision_policy == :apriori && return apriori_working_precision(p, n, i)
    e_red, e_can1 = exponent_phase(X, i; threaded=threaded)
    precision_policy == :exponent &&
        return exponent_working_precision(p, n, i, e_red, e_can1)
    precision_policy == :adaptive || throw(
        ArgumentError(
            "unknown precision policy :$precision_policy (use :apriori, :exponent or :adaptive)",
        ),
    )
    Tg = Tguess === nothing ? initial_adaptive_guess(p, n, i) : Tguess
    return adaptive_working_precision(p, n, i, Tg, e_red, e_can1)
end

"""
    syntomic_complex(X, i; N=nothing, precision_policy=:apriori, threaded=true, descent=true) -> SyntomicComplex

The syntomic complex `syn^0`, `syn^1` computing `Z_p(i)(X)` for the finite chain ring
`X = O_K/ϖ^n` ([AKN, §7.1, pp. 53–57]). The working precision `N` defaults to the one of the given
policy. With `descent = true` (default), `f = 1` arithmetic is used when the Eisenstein polynomial
has coefficients in `Z_p`.
"""
function syntomic_complex(
    X::FiniteChainRing,
    i::Integer;
    N::Union{Nothing,Integer}=nothing,
    precision_policy::Symbol=:apriori,
    threaded::Bool=true,
    descent::Bool=true,
    timings::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
)
    i >= 1 || throw(ArgumentError("the weight i must be at least 1, got $i"))
    L = X.n * i
    NN = if N === nothing
        working_precision(X, i; precision_policy=precision_policy, threaded=threaded)
    else
        Int(N)
    end
    Rf = GaloisRing(X.p, X.htilde, NN)
    if L == 1                                             # F^{[1,0]} = 0: the complex is zero
        z = PMatrix(zero_matrix(Rf.Zn, 0, 0), NN)
        return SyntomicComplex(
            X,
            i,
            NN,
            PMatrix(zero_matrix(Rf.Zn, 0, 0), NN),
            z,
            (red=0, can1=0, nablaN=0),
            Dict{Symbol,Any}(),
        )
    end
    Xd = descent ? descent_ring(X, NN) : nothing
    t0 = time()
    W = syntomic_wmatrices(something(Xd, X), i, NN; threaded=threaded)
    timings[:wmatrices] = time() - t0
    t0 = time()
    syn0, syn1, exps, inter = syntomic_square(W, Rf; descent=Xd !== nothing)
    timings[:square] = time() - t0
    return SyntomicComplex(X, i, NN, syn0, syn1, exps, inter)
end

"""
    syntomic_cohomology(X::FiniteChainRing, i; precision_policy=:apriori, N=nothing,
                        threaded=true, check=true, descent=true) -> SyntomicCohomology

The cohomology `H^1`, `H^2` of `Z_p(i)(X)` for the finite chain ring `X = O_K/ϖ^n`
([AKN, §7.1, pp. 53–58]); `H^0 = 0`.

Precision policies (see [`working_precision`](@ref)):
- `:apriori`: the working precision of [AKN, §7.1, Step 1];
- `:exponent`: the same bound with the exact exponents of `red` and `can` (computed first);
- `:adaptive`: start from a guess for the target precision and double it until the result is
  certified ([AKN, Remark 7.3, p. 58]); the last attempt uses the `:exponent` precision.
"""
function syntomic_cohomology(
    X::FiniteChainRing,
    i::Integer;
    precision_policy::Symbol=:apriori,
    N::Union{Nothing,Integer}=nothing,
    threaded::Bool=true,
    check::Bool=true,
    descent::Bool=true,
)
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
        return syntomic_cohomology(
            C; check=check, threaded=threaded, policy=precision_policy, timings=timings
        )
    end
    p, n = X.p, X.n
    T = target_precision(p, n, i)
    e_red, e_can1 = exponent_phase(X, i; threaded=threaded)
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
            C = syntomic_complex(
                X, i; N=NN, threaded=threaded, descent=descent, timings=timings
            )
            return syntomic_cohomology(
                C;
                check=check,
                threaded=threaded,
                policy=:adaptive,
                restarts=restarts,
                timings=timings,
            )
        catch err
            (err isa PrecisionExhausted && !final) || rethrow()
            restarts += 1
            Tg = min(2 * Tg, T)
        end
    end
end
