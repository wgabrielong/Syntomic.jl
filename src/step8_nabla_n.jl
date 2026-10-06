# Step 8: ∇ on the Nygaard filtration.
#
# [AKN24, §7.1, Step 8, p. 56]: "Use ∇ and the can maps to compute the unique map N^{≥i}∇ making the
# diagram commute … the can maps are rational isomorphisms by Lemma 4.63, so the top map can be
# computed as can^{-1} ∘ ∇ ∘ can."
#
# The ∇-terms are F^{[1,in-1]}N^{≥i}Δ^(1),∇_{R/A_0}{i} = F^{[0,in-2]}N^{≥i-1}Δ^(1)_{R/A_0}{i-1}
# ([AKN24, §7.1, Step 2, p. 54]); their basis consists of d̃^{i-1-⌊m/n⌋} b_m, 0 ≤ m ≤ in-2, so rows and
# columns of can on the ∇-terms are indexed by F-weights 0, …, L-2.

"""
    can_nabla_matrix(env, i; threaded=true) -> WMatrix

The matrix of `can` on the `∇`-terms, `F^{[0,in-2]}N^{≥i-1}Δ^(1)_{R/A_0}{i-1} →
F^{[0,in-2]}Δ^(1)_{R/A_0}{i-1}`: column `m` is the normal form of `E(z_0)^{i-1-⌊m/n⌋} b_m`
([AKN24, §7.1, Step 8, p. 56]).
"""
function can_nabla_matrix(env::FEnvelope, i::Integer; threaded::Bool=true)
    prm = env.prm
    L = prm.L
    Ep = f_E_powers(env, i - 1)
    cols = tmap(0:(L - 2); threaded=threaded) do m
        j = i - 1 - div(m, prm.n)
        return f_can_image(env, m, j; Ej=Ep[j + 1])
    end
    return f_column_matrix(env, collect(FElem, cols), 0:(L - 2))
end
