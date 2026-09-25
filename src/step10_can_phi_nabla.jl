# Steps 8–10 at the level of Z_p-matrices: N^{≥i}∇, can - φ and can - φ^∇.
#
# [AKN, §7.1, Step 8, p. 56]: N^{≥i}∇ = can^{-1} ∘ ∇ ∘ can (the can on the ∇-terms is a rational
#   isomorphism, [AKN, Lemma 4.63, p. 45]; precision loss bounded by [AKN, Lemma 7.8, p. 60]).
# [AKN, §7.1, Step 9, p. 57]: can - φ as the difference of the two matrices.
# [AKN, §7.1, Step 10, p. 57]: "Using that N^{≥i}∇ is a rational isomorphism, compute the unique map
#   can - φ^∇ such that [the square] commutes, as ∇ ∘ (can - φ) ∘ (N^{≥i}∇)^{-1}" (precision loss
#   bounded by [AKN, Lemma 7.9, p. 60]).
#
# Since ∇ ∘ can = can ∘ N^{≥i}∇ by construction, can - φ^∇ = can - ∇ ∘ φ ∘ (N^{≥i}∇)^{-1}, and only
# the second term has to be computed by a solve. W(F_q)-linear maps are expanded as 𝓜(A) and the
# σ-semilinear φ as 𝓜(Φ)(I ⊗ S) (see `zp_expand`).

"""
    nabla_N_matrix(can0, can1, nablaR, f, p; e_can1=nothing, threaded=true) -> (PMatrix, e_can1)

`N^{≥i}∇ = can1^{-1} ∘ ∇_R ∘ can0` ([AKN, §7.1, Step 8, p. 56]), where `can0` and `can1` are the
`Z_p`-matrices of `can` on the terms of level 0 and on the `∇`-terms; also returns the p-exponent
`e_can1` of the cokernel of `can1` (the precision lost, [AKN, Lemma 7.8, p. 60]; computed unless
given). With `threaded = true` the exponent and the solve are computed in parallel.
"""
function nabla_N_matrix(
    can0::PMatrix,
    can1::PMatrix,
    nablaR::PMatrix,
    f::Integer,
    p::Integer;
    e_can1::Union{Nothing,Integer}=nothing,
    threaded::Bool=true,
)
    e = if e_can1 === nothing
        local_exponent(can1.M, p, can1.prec; threaded=threaded)
    else
        e_can1
    end
    X = pm_mul(nablaR, can0)
    return solve_lower(can1, X, f, p; exponent=e, threaded=threaded), Int(e)
end

"""
    can_minus_phi(can0, phi) -> PMatrix

The `Z_p`-matrix of `can - φ: F^{[1,in-1]}N^{≥i}Δ^(1)_{R/A_0}{i} → F^{[1,in-1]}Δ^(1)_{R/A_0}{i}`
([AKN, §7.1, Step 9, p. 57]); `phi` is the `Z_p`-matrix of the σ-semilinear `φ`.
"""
can_minus_phi(can0::PMatrix, phi::PMatrix) = pm_sub(can0, phi)

"""
    phi_nabla_matrix(nablaR, phi, nablaN, f, p; e_nablaN=nothing, threaded=true) -> (PMatrix, e_nablaN)

The `Z_p`-matrix of `φ^∇ = ∇_R ∘ φ ∘ (N^{≥i}∇)^{-1}` ([AKN, §7.1, Step 10, p. 57]) and the p-exponent
of the cokernel of `N^{≥i}∇` (the precision lost, [AKN, Lemma 7.9, p. 60]; computed unless
given). With `threaded = true` the exponent and the solve are computed in parallel.
"""
function phi_nabla_matrix(
    nablaR::PMatrix,
    phi::PMatrix,
    nablaN::PMatrix,
    f::Integer,
    p::Integer;
    e_nablaN::Union{Nothing,Integer}=nothing,
    threaded::Bool=true,
)
    e = if e_nablaN === nothing
        local_exponent(nablaN.M, p, nablaN.prec; threaded=threaded)
    else
        e_nablaN
    end
    X = pm_mul(nablaR, phi)
    return solve_right_lower(X, nablaN, f, p; exponent=e, threaded=threaded), Int(e)
end

"""
    can_minus_phi_nabla(can1, phinabla) -> PMatrix

The `Z_p`-matrix of `can - φ^∇` on the `∇`-terms ([AKN, §7.1, Step 10, p. 57]).
"""
can_minus_phi_nabla(can1::PMatrix, phinabla::PMatrix) = pm_sub(can1, phinabla)
