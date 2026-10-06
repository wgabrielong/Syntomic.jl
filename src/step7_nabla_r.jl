# Step 7: ∇ for R = O_K/ϖ^n.
#
# [AKN24, §7.1, Step 7, p. 56]: "The map ∇ is natural in R, so we may reduce to the O_K case via the
# commutative diagram" formed by ∇_{O_K}, ∇_R and the reduction maps red, red^∇ of Step 6. Since red is
# a rational isomorphism ([AKN24, Lemma 4.64, p. 45]), ∇_R = red^∇ ∘ ∇_{O_K} ∘ red^{-1}; the precision
# lost is bounded by the p-exponent of the cokernel of red ([AKN24, Lemma 7.6, p. 59]).

"""
    nabla_R_matrix(red, redN, nablaOK, f, p; e_red=nothing, threaded=true) -> (PMatrix, e_red)

The `Z_p`-matrix of `∇_R = red^∇ ∘ ∇_{O_K} ∘ red^{-1}` ([AKN24, §7.1, Step 7, p. 56]) from
the `Z_p`-matrices of `red`, `red^∇` and `∇_{O_K}`, together with the p-exponent `e_red` of
the cokernel of `red` (the precision lost, [AKN24, Lemma 7.6, p. 59]; computed unless
given). With `threaded = true` the exponent and the solve are computed in parallel.
"""
function nabla_R_matrix(
    red::PMatrix,
    redN::PMatrix,
    nablaOK::PMatrix,
    f::Integer,
    p::Integer;
    e_red::Union{Nothing,Integer}=nothing,
    threaded::Bool=true,
)
    e = e_red === nothing ? local_exponent(red.M, p, red.prec; threaded=threaded) : e_red
    X = pm_mul(redN, nablaOK)
    return solve_right_lower(X, red, f, p; exponent=e, threaded=threaded), Int(e)
end
