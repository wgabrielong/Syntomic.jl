# Step 9: the maps can and φ on F^{[1,in-1]}N^{≥i}Δ^(1)_{R/A_0}{i}.
#
# [AKN, §7.1, Step 9, p. 57]: "Compute a matrix for can by reducing each of the basis elements
# d̃^{i-Σp^u e_u} z_0^k ∏ f_u^{e_u} in Δ^(1)_{R/W(k)[[z_0]]} (replacing d̃ to E(z_0)). Compute a matrix
# for φ by reducing φ_i(d̃^{i-Σp^u e_u} z_0^k ∏ f_u^{e_u}) = z_0^{pk} ∏ φ_{p^u}(f_u)^{e_u}."
#
# The basis of F^{[1,in-1]}N^{≥i}Δ^(1)_{R/A_0} ([AKN, Prop. 3.34(2), p. 24]) consists of
# d̃^{i-s} b_w for 1 ≤ w ≤ in-1, where b_w = z_0^c ∏ f_u^{e_u} is the basis monomial of F-weight w and
# s = ⌊w/n⌋ = Σ p^u e_u ≤ i-1. Rows and columns are indexed by F-weights 1, …, L-1.
#
# can is W(F_q)-linear; φ is σ-semilinear, and the matrix computed here is its matrix Φ on basis
# vectors, so that φ(Σ a_j v_j) = Σ σ(a_j) φ(v_j).

"""
    can_matrix(env, i; threaded=true) -> WMatrix

The matrix of `can: F^{[1,in-1]}N^{≥i}Δ^(1)_{R/A_0}{i} → F^{[1,in-1]}Δ^(1)_{R/A_0}{i}`: column `w`
is the normal form of `E(z_0)^{i-⌊w/n⌋} b_w` ([AKN, §7.1, Step 9, p. 57]).
"""
function can_matrix(env::FEnvelope, i::Integer; threaded::Bool=true)
    prm = env.prm
    L = prm.L
    cols = tmap(w -> f_can_image(env, w, i - div(w, prm.n)), 1:(L - 1); threaded=threaded)
    return f_column_matrix(env, collect(FElem, cols), 1:(L - 1))
end

"""
    phi_matrix(env; threaded=true) -> WMatrix

The matrix `Φ` of the σ-semilinear map `φ_i: F^{[1,in-1]}N^{≥i}Δ^(1)_{R/A_0}{i} →
F^{[1,in-1]}Δ^(1)_{R/A_0}{i}` on basis vectors: column `w` is the normal form of
`z_0^{pc} ∏ (λ_u f_{u+1})^{e_u}` ([AKN, §7.1, Step 9, p. 57]; [AKN, Lemma 3.26(1), p. 21]).
"""
function phi_matrix(env::FEnvelope; threaded::Bool=true)
    L = env.prm.L
    cols = tmap(w -> f_phi_image(env, w), 1:(L - 1); threaded=threaded)
    return f_column_matrix(env, collect(FElem, cols), 1:(L - 1))
end
