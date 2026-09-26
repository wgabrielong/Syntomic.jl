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
    Ep = f_E_powers(env, i)
    cols = tmap(1:(L - 1); threaded=threaded) do w
        j = i - div(w, prm.n)
        return f_can_image(env, w, j; Ej=Ep[j + 1])
    end
    return f_column_matrix(env, collect(FElem, cols), 1:(L - 1))
end

"""
    phi_matrix(env; threaded=true) -> WMatrix

The matrix `Φ` of the σ-semilinear map `φ_i: F^{[1,in-1]}N^{≥i}Δ^(1)_{R/A_0}{i} →
F^{[1,in-1]}Δ^(1)_{R/A_0}{i}` on basis vectors: column `w` is the normal form of
`z_0^{pc} ∏ (λ_u f_{u+1})^{e_u}` ([AKN, §7.1, Step 9, p. 57]; [AKN, Lemma 3.26(1), p. 21]).
"""
function phi_matrix(env::FEnvelope; threaded::Bool=true)
    prm = env.prm
    L = prm.L
    # the columns of F-weights w = c + n·s share the series ∏_u λ_u^{e_u} of B_s (needed modulo
    # z_0^{L - p·w}); only the F-weights w < L/p give nonzero columns
    ss = 0:div(cld(L, prm.p) - 1, prm.n)
    lams = tmap(ss; threaded=threaded) do s
        e = f_basis_exponents(prm, prm.n * s)[2]
        return f_phi_series(env, e; zprec=L - prm.p * prm.n * s)
    end
    cols = tmap(1:(L - 1); threaded=threaded) do w
        s = div(w, prm.n)
        return f_phi_image(env, w; lam=s < length(lams) ? lams[s + 1] : nothing)
    end
    return f_column_matrix(env, collect(FElem, cols), 1:(L - 1))
end
